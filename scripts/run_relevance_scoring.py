import json
import os
import argparse
import yaml
import pandas as pd
import tqdm
from openai import OpenAI
from together import Together
from dotenv import load_dotenv


REQUIRED_KEYS = ("model", "input_csv", "output_csv", "provider", "retrieved_docs_dir")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    return cfg


class SupportScorer:
    """
    Scores whether the retrieved documents contain enough information to
    identify the correct answer to a multiple-choice medical question.

    For each row the judge LLM returns a single integer:
      0 — context contains no relevant evidence for the correct answer
      1 — context is partially relevant but insufficient to identify the correct answer
      2 — context clearly contains enough information to identify the correct answer
    """

    SCORING_SYSTEM_PROMPT = (
        "You are a strict medical evidence evaluator. "
        "You will be given a multiple-choice medical question, the correct answer, "
        "and a set of retrieved context documents. "
        "Your task is to judge whether the retrieved context contains enough information "
        "to identify the correct answer — independent of what any model may have said.\n\n"
        "Respond with ONLY a single integer:\n"
        "  0 — the context contains no relevant evidence for the correct answer\n"
        "  1 — the context is partially relevant but insufficient to identify the correct answer\n"
        "  2 — the context clearly contains enough information to identify the correct answer\n\n"
        "Output the integer and nothing else."
    )

    def __init__(
        self,
        input_csv: str,
        output_csv: str,
        model: str,
        provider: str,
        retrieved_docs_dir: str,
        top_k_docs: int = 5,
        verbose: bool = False,
    ):
        load_dotenv()
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.model = model
        self.retrieved_docs_dir = retrieved_docs_dir
        self.top_k_docs = top_k_docs
        self.verbose = verbose

        provider = provider.lower()
        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "together":
            self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        elif provider == "local":
            self.client = OpenAI(
                base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"),
                api_key=os.getenv("LOCAL_API_KEY", "local-dev-key"),
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
        self.provider = provider

    def _call_llm(self, prompt: str) -> str:
        extra = {}
        if self.provider == "openai" and "gpt-5" in self.model:
            extra["reasoning_effort"] = "minimal"
        else:
            extra["temperature"] = 0

        response = self.client.chat.completions.create(
            model=self.model,
            **extra,
            messages=[
                {"role": "system", "content": self.SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _parse_score(self, raw: str) -> int | None:
        """Extract the first digit 0/1/2 from the LLM response."""
        for ch in raw:
            if ch in ("0", "1", "2"):
                return int(ch)
        return None

    def _load_docs(self, question_id) -> list[str]:
        path = os.path.join(self.retrieved_docs_dir, f"{question_id}.json")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("retrieval_result", {}).get("documents", [])

    def _build_scoring_prompt(self, row: pd.Series) -> str | None:
        """Return None if there are no retrieved documents to score against."""
        docs = self._load_docs(row["question_id"])
        if not docs:
            return None

        docs = docs[: self.top_k_docs]

        try:
            answer_options = json.loads(row["answer_options"])
        except (json.JSONDecodeError, KeyError):
            answer_options = {}

        correct_letter = str(row.get("correct_choice", "")).strip().upper()
        correct_text = answer_options.get(correct_letter, "")
        correct_display = (
            f"{correct_letter}. {correct_text}" if correct_text else correct_letter
        )

        options_block = "\n".join(f"{k}. {v}" for k, v in answer_options.items())
        context_block = "\n\n".join(
            f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(docs)
        )

        return (
            f"QUESTION:\n{row['question']}\n\n"
            f"ANSWER OPTIONS:\n{options_block}\n\n"
            f"CORRECT ANSWER: {correct_display}\n\n"
            f"RETRIEVED CONTEXT:\n{context_block}"
        )

    def _load_checkpoint(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (already_scored_df, todo_df). Skips rows with an existing score."""
        if os.path.exists(self.output_csv):
            existing = pd.read_csv(self.output_csv)
            scored_ids = set(
                existing.loc[existing["context_sufficient"].notna(), "question_id"].astype(str)
            )
            todo = df[~df["question_id"].astype(str).isin(scored_ids)].copy()
            print(f"Resuming: {len(scored_ids)} already scored, {len(todo)} remaining.")
            return existing, todo
        return pd.DataFrame(), df.copy()

    def _save_checkpoint(self, existing: pd.DataFrame, new_rows: list[dict]) -> None:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        # keep the latest score per question; new_rows are appended last so they win
        combined = combined.drop_duplicates(subset=["question_id"], keep="last")
        combined[["question_id", "context_sufficient"]].to_csv(self.output_csv, index=False)

    def process(self):
        df = pd.read_csv(self.input_csv)
        existing, todo = self._load_checkpoint(df)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"INPUT:   {self.input_csv}  ({len(df)} rows)")
            print(f"DOCS:    {self.retrieved_docs_dir}")
            print(f"MODEL:   {self.model}")
            print(f"TOP_K:   {self.top_k_docs} docs per question")
            print(f"OUTPUT:  {self.output_csv}")
            print(f"{'='*60}\n")

        new_rows: list[dict] = []

        for _, row in tqdm.tqdm(todo.iterrows(), total=len(todo), desc="Scoring context"):
            prompt = self._build_scoring_prompt(row)
            if prompt is None:
                score = None
                if self.verbose:
                    print(f"  [{row.get('question_id', '?')}] SKIPPED — no retrieved documents")
            else:
                raw = self._call_llm(prompt)
                score = self._parse_score(raw)
                if self.verbose:
                    correct = row.get("correct_choice", "?")
                    print(
                        f"  [{row.get('question_id', '?')}] "
                        f"correct={correct}  context_sufficient={score}  (raw={raw!r})"
                    )

            new_rows.append({"question_id": row["question_id"], "context_sufficient": score})

            if len(new_rows) % 50 == 0:
                self._save_checkpoint(existing, new_rows)
                print(f"  [checkpoint] saved {len(existing) + len(new_rows)} rows to {self.output_csv}")

        self._save_checkpoint(existing, new_rows)
        df = pd.read_csv(self.output_csv)

        scored = df["context_sufficient"].notna()
        print(f"\nResults saved to {self.output_csv}")
        print(f"Rows scored: {scored.sum()} / {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)

    scorer = SupportScorer(
        input_csv=cfg["input_csv"],
        output_csv=cfg["output_csv"],
        model=cfg["model"],
        provider=cfg["provider"],
        retrieved_docs_dir=cfg["retrieved_docs_dir"],
        top_k_docs=cfg.get("top_k_docs", 5),
        verbose=cfg.get("verbose", False),
    )
    scorer.process()
