import json
import os
import re
import argparse
import yaml
import asyncio
import pandas as pd
import tqdm.asyncio
from openai import AsyncOpenAI
from together import AsyncTogether
from dotenv import load_dotenv


REQUIRED_KEYS = ("model", "output_file", "provider", "rag_mode", "retrieved_docs_dir")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    return cfg


class AnswerGenerator:
    """
    Generates answers for multiple-choice medical questions with async concurrency
    for high-throughput inference on multi-GPU setups.

    - rag_mode=True:  reads agent retrieval records from retrieved_docs_dir and
                      passes retrieved documents as context to the LLM.
    - rag_mode=False: reads questions from retrieved_docs_dir but calls the LLM
                      without any retrieval context (baseline).
    """

    BASELINE_SYSTEM_PROMPT = (
        "You are a medical expert assistant. "
        "Answer the multiple choice question by selecting the best answer "
        "based on your medical knowledge. "
        "Respond with only the answer choice letter and nothing else."
        "Do NOT explain your reasoning."
        "Do NOT add any text before or after your answer."
    )

    RAG_SYSTEM_PROMPT = (
        "You are a medical expert assistant. "
        "You must answer based on the context from the guidelines provided by a database.\n"
        "You are provided with the following:\n"
        "- Multiple choice question\n"
        "- Database context knowledge\n\n"
        "Do NOT use information that is not relevant for answering the question.\n"
        "Ensure your answer is annotated with the Document IDs of the context that were used to answer the question.\n\n"
        "STRICT OUTPUT FORMAT — no other text allowed:\n"
        "Answer: 'X'\n"
        "Used sources: [Document ID N], [Document ID M], [Document ID O], ... [Document ID Z]\n\n"
        "If no sources are relevant:\n"
        "Answer: 'X'\n"
        "Used sources: None\n\n"
        "RULES:\n"
        "- Do NOT explain your reasoning\n"
        "- Do NOT add any text before or after the format"
        "- The answer MUST be only one of the provided answer choices."
    )

    OPEN_ENDED_BASELINE_SYSTEM_PROMPT = (
        "You are a medical expert assistant. "
        "Answer the medical question concisely and accurately based on your medical knowledge. "
        "Give a single, final answer and do not second-guess or restate it. "
        "Do NOT explain your reasoning. "
        "Do NOT add any text before or after your answer."
    )

    OPEN_ENDED_RAG_SYSTEM_PROMPT = (
        "You are a medical expert assistant. "
        "You must answer based on the context from the guidelines provided by a database.\n"
        "Give a single, final answer and do not second-guess or restate it. \n"
        "You are provided with the following:\n"
        "- A medical question\n"
        "- Database context knowledge\n\n"
        "Do NOT use information that is not relevant for answering the question.\n"
        "Ensure your answer is annotated with the Document IDs of the context that were used to answer the question.\n\n"
        "STRICT OUTPUT FORMAT — no other text allowed:\n"
        "Answer: <your concise answer>\n"
        "Used sources: [Document ID N], [Document ID M], [Document ID O], ... [Document ID Z]\n\n"
        "If no sources are relevant:\n"
        "Answer: <your concise answer>\n"
        "Used sources: None\n\n"
        "RULES:\n"
        "- Do NOT explain your reasoning\n"
        "- Do NOT add any text before or after the format\n"
        "- Answer concisely in your own words"
    )

    def __init__(
        self,
        output_csv: str,
        model: str,
        provider: str,
        rag_mode: bool,
        retrieved_docs_dir: str,
        local_base_url: str = "http://localhost:8000/v1",
        verbose: bool = False,
        retry_nan_responses: bool = False,
        rerank_top_k: int = None,
        min_confidence: float = None,
        sufficient_only: bool = True,
        filter_sufficient_from_dir: str = None,
        open_ended: bool = False,
        request_timeout: float = 120.0,
        max_tokens: int = 1024,
        max_retries: int = 2,
        concurrency: int = 16,
    ):
        load_dotenv()
        self.output_csv = output_csv
        self.model = model
        self.rag_mode = rag_mode
        self.retrieved_docs_dir = retrieved_docs_dir
        self.verbose = verbose
        self.retry_nan_responses = retry_nan_responses
        self.rerank_top_k = rerank_top_k
        self.min_confidence = min_confidence
        self.sufficient_only = sufficient_only
        self.filter_sufficient_ids = self._load_sufficient_ids(filter_sufficient_from_dir)
        self.open_ended = open_ended
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.provider = provider.lower()
        self._local_base_url = local_base_url

        # Async clients — instantiated lazily inside the async context
        self._async_client = None

    # ------------------------------------------------------------------ #
    #  Client setup (lazy, inside async context)                           #
    # ------------------------------------------------------------------ #

    def _get_client(self):
        if self._async_client is not None:
            return self._async_client

        if self.provider == "openai":
            self._async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "together":
            self._async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))
        elif self.provider == "local":
            self._async_client = AsyncOpenAI(
                base_url=self._local_base_url, api_key="none"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._async_client

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _vprint(self, msg: str = "") -> None:
        if self.verbose:
            print(msg)

    async def _call_llm(self, prompt: str, semaphore: asyncio.Semaphore) -> str:
        if self.open_ended:
            system_prompt = self.OPEN_ENDED_RAG_SYSTEM_PROMPT if self.rag_mode else self.OPEN_ENDED_BASELINE_SYSTEM_PROMPT
        else:
            system_prompt = self.RAG_SYSTEM_PROMPT if self.rag_mode else self.BASELINE_SYSTEM_PROMPT

        extra = {}
        if self.provider == "openai" and self.model in ("gpt-5", "gpt-5-mini"):
            extra["reasoning_effort"] = "minimal"
        else:
            extra["temperature"] = 0
        if self.max_tokens is not None and self.provider != "openai":
            extra["max_tokens"] = self.max_tokens

        async with semaphore:
            response = await self._get_client().chat.completions.create(
                model=self.model,
                timeout=self.request_timeout,
                **extra,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
        return response.choices[0].message.content

    @staticmethod
    def _format_rag_prompt(question_prompt: str, documents: list) -> str:
        context = "\n\n".join(
            [f"Document ID {i+1}:\n{doc}" for i, doc in enumerate(documents)]
        )
        return f"CONTEXT:\n{context}\n\nQUESTION:\n{question_prompt}"

    def _build_final_prompt(self, record: dict) -> str:
        question = record["question"] if self.open_ended else record["question_prompt"]
        retrieval_result = record.get("retrieval_result")
        if retrieval_result and retrieval_result.get("documents"):
            docs = retrieval_result["documents"]
            if self.rerank_top_k is not None:
                docs = docs[: self.rerank_top_k]
            return self._format_rag_prompt(question, docs)
        return question

    @staticmethod
    def _parse_open_ended_rag_response(response: str) -> tuple[str, str]:
        answer = ""
        used_sources = ""
        for line in response.strip().splitlines():
            line = line.strip()
            if line.startswith("Answer:"):
                answer = line[len("Answer:"):].strip()
            elif line.startswith("Used sources:"):
                used_sources = line[len("Used sources:"):].strip()
        return answer, used_sources

    @staticmethod
    def _parse_rag_response(response: str) -> tuple[str, str]:
        answer = ""
        used_sources = ""
        for line in response.strip().splitlines():
            line = line.strip()
            if line.startswith("Answer:"):
                match = re.search(r"['\"]([A-Ja-j])['\"]", line)
                if match:
                    answer = match.group(1).upper()
                else:
                    rest = line[len("Answer:") :].strip()
                    if rest:
                        answer = rest[0].upper()
            elif line.startswith("Used sources:"):
                used_sources = line[len("Used sources:") :].strip()
        return answer, used_sources

    @staticmethod
    def _build_result_row(
        record: dict,
        llm_response: str,
        llm_response_raw: str,
        used_sources: str,
    ) -> dict:
        retrieval_result = record.get("retrieval_result")
        row = {
            "question_id": record["question_id"],
            "question": record["question"],
            "answer_options": json.dumps(record["answer_options"]),
            "correct_choice": record["correct_choice"],
            "llm_response": llm_response,
            "llm_response_raw": llm_response_raw,
            "used_sources": used_sources,
            "answer_source": record["answer_source"],
        }
        if retrieval_result:
            docs = retrieval_result.get("documents", [])
            history = retrieval_result.get("retrieval_history", [])
            last_critique = (history[-1].get("critique") or {}) if history else {}
            row.update(
                {
                    "conditions": json.dumps(retrieval_result.get("conditions", [])),
                    "retrieval_iterations": sum(
                        1 for s in history if s.get("query")
                    ),
                    "final_doc_count": len(docs),
                    "final_sufficient": retrieval_result.get("final_sufficient"),
                    "final_confidence": last_critique.get("confidence"),
                    "retrieval_history": json.dumps(history),
                    "retrieved_documents": json.dumps(docs),
                }
            )
        else:
            row.update(
                {
                    "conditions": None,
                    "retrieval_iterations": 0,
                    "final_doc_count": 0,
                    "final_sufficient": None,
                    "final_confidence": None,
                    "retrieval_history": None,
                    "retrieved_documents": None,
                }
            )
        return row

    # ------------------------------------------------------------------ #
    #  Data loading                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_sufficient_ids(directory: str) -> set | None:
        if not directory:
            return None
        ids = set()
        for filename in os.listdir(directory):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                record = json.load(f)
            if (record.get("retrieval_result") or {}).get("final_sufficient"):
                ids.add(str(record["question_id"]))
        print(f"filter_sufficient_from_dir: {len(ids)} sufficient question IDs loaded from {directory}")
        return ids

    def _is_sufficient(self, retrieval_result: dict) -> bool:
        if self.min_confidence is not None:
            history = retrieval_result.get("retrieval_history", [])
            if not history:
                return False
            last = history[-1].get("critique") or {}
            return (
                last.get("sufficient", False)
                and last.get("confidence", 0) >= self.min_confidence
            )
        return retrieval_result.get("final_sufficient", False)

    def _load_rag_records(self) -> list:
        records = []
        for filename in sorted(os.listdir(self.retrieved_docs_dir)):
            if not filename.endswith(".json"):
                continue
            with open(
                os.path.join(self.retrieved_docs_dir, filename), "r", encoding="utf-8"
            ) as f:
                record = json.load(f)
            rr = record.get("retrieval_result") or {}
            if self.sufficient_only and not self._is_sufficient(rr):
                continue
            if self.filter_sufficient_ids is not None and str(record["question_id"]) not in self.filter_sufficient_ids:
                continue
            records.append(record)
        return records

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                       #
    # ------------------------------------------------------------------ #

    def _load_checkpoint(self, records: list) -> tuple[list, list]:
        """Return (already_done_rows, todo_records)."""
        if os.path.exists(self.output_csv):
            try:
                existing_df = pd.read_csv(self.output_csv)
            except pd.errors.EmptyDataError:
                return [], records
            if self.retry_nan_responses:
                if self.open_ended:
                    valid = existing_df["llm_response"].astype(str).str.strip().ne("")
                else:
                    valid = (
                        existing_df["llm_response"].astype(str).str.match(r"^[A-Ea-e]$")
                    )
                done_df = existing_df[valid]
            else:
                done_df = existing_df
            done_ids = set(done_df["question_id"].astype(str))
            todo = [r for r in records if str(r["question_id"]) not in done_ids]
            print(
                f"Resuming: {len(done_ids)} already answered, {len(todo)} remaining."
            )
            return done_df.to_dict("records"), todo
        return [], records

    def _save_checkpoint(self, rows: list) -> None:
        """Atomic write — safe against crashes mid-save."""
        tmp = self.output_csv + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, self.output_csv)

    # ------------------------------------------------------------------ #
    #  Per-record async handler                                            #
    # ------------------------------------------------------------------ #

    async def _handle_record(
        self, record: dict, semaphore: asyncio.Semaphore
    ) -> dict | None:
        if self.verbose:
            rr = record.get("retrieval_result") or {}
            docs = rr.get("documents", [])
            history = rr.get("retrieval_history", [])
            iterations = sum(1 for s in history if s.get("query"))
            print(
                f"\n[{record['question_id']}] source={record['answer_source']} "
                f"docs={len(docs)} iters={iterations}"
            )

        prompt = self._build_final_prompt(record)

        llm_response_raw = None
        for attempt in range(1, self.max_retries + 2):
            try:
                llm_response_raw = await self._call_llm(prompt, semaphore)
                break
            except Exception as e:
                if attempt <= self.max_retries:
                    print(f"\n  [WARN] [{record['question_id']}] LLM call failed "
                          f"(attempt {attempt}/{self.max_retries + 1}), retrying: {e}")
                    await asyncio.sleep(2 * attempt)
                else:
                    print(f"\n  [ERROR] [{record['question_id']}] LLM call failed "
                          f"after {self.max_retries + 1} attempts, skipping: {e}")
        if llm_response_raw is None:
            return None

        if self.rag_mode:
            if self.open_ended:
                llm_response, used_sources = self._parse_open_ended_rag_response(llm_response_raw)
            else:
                llm_response, used_sources = self._parse_rag_response(llm_response_raw)
        else:
            llm_response = llm_response_raw.strip()
            used_sources = None

        self._vprint(f"  response: {llm_response_raw.strip()}")

        return self._build_result_row(record, llm_response, llm_response_raw, used_sources)

    # ------------------------------------------------------------------ #
    #  Main async entrypoint                                               #
    # ------------------------------------------------------------------ #

    async def process(self):
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"MODE:        {'RAG' if self.rag_mode else 'baseline (no retrieval)'}")
            print(f"MODEL:       {self.model}")
            print(f"PROVIDER:    {self.provider}")
            print(f"INPUT:       {self.retrieved_docs_dir}")
            print(f"OUTPUT:      {self.output_csv}")
            print(f"CONCURRENCY: {self.concurrency}")
            print(f"{'='*60}\n")

        raw_records = self._load_rag_records()

        if self.rag_mode:
            records = raw_records
            n_before = len(records)
            records = [
                r
                for r in records
                if r["answer_source"] in ("agentic_rag", "agentic_rag_insufficient", "naive_rag")
            ]
            skipped = n_before - len(records)
            if skipped:
                self._vprint(
                    f"Skipping {skipped} records with non-agentic answer_source."
                )
        else:
            records = [
                {**r, "retrieval_result": None, "answer_source": "non_rag"}
                for r in raw_records
            ]

        self._vprint(f"Loaded {len(records)} records\n")

        existing_rows, todo = self._load_checkpoint(records)

        # Thread-safe accumulator protected by a lock (multiple coroutines append)
        all_rows = list(existing_rows)
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(self.concurrency)
        checkpoint_counter = [0]  # mutable int in a list so closure can write it

        async def run_and_collect(record: dict):
            row = await self._handle_record(record, semaphore)
            if row is None:
                return
            async with lock:
                all_rows.append(row)
                checkpoint_counter[0] += 1
                if checkpoint_counter[0] % 20 == 0:
                    self._save_checkpoint(all_rows)
                    print(
                        f"  [checkpoint] saved {len(all_rows)} rows → {self.output_csv}"
                    )

        tasks = [run_and_collect(r) for r in todo]
        await tqdm.asyncio.tqdm.gather(*tasks, desc="Generating answers")

        self._save_checkpoint(all_rows)
        df = pd.DataFrame(all_rows)

        print(f"\nResults saved to {self.output_csv}")
        print(f"Total questions answered: {len(all_rows)}")

        if self.verbose and not df.empty:
            if (
                "retrieval_iterations" in df.columns
                and df["retrieval_iterations"].gt(0).any()
            ):
                print(
                    f"Average retrieval iterations: {df['retrieval_iterations'].mean():.2f}"
                )
                print(
                    f"Average documents retrieved:  {df['final_doc_count'].mean():.2f}"
                )
            source_counts = df["answer_source"].value_counts().to_dict()
            print(f"Answer source breakdown:      {source_counts}")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Truncate reranked docs to top-k at answerer time",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max concurrent LLM requests (overrides config)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    output_file = cfg["output_file"]
    if args.top_k is not None:
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}_k{args.top_k}{ext}"

    concurrency = args.concurrency or cfg.get("concurrency", 16)

    generator = AnswerGenerator(
        output_csv=output_file,
        model=cfg["model"],
        provider=cfg["provider"],
        rag_mode=cfg["rag_mode"],
        retrieved_docs_dir=cfg["retrieved_docs_dir"],
        local_base_url=cfg.get("local_base_url", "http://localhost:8000/v1"),
        verbose=cfg.get("verbose", False),
        retry_nan_responses=cfg.get("retry_nan_responses", False),
        rerank_top_k=args.top_k,
        min_confidence=cfg.get("min_confidence"),
        sufficient_only=cfg.get("sufficient_only", True),
        filter_sufficient_from_dir=cfg.get("filter_sufficient_from_dir"),
        open_ended=cfg.get("open_ended", False),
        request_timeout=cfg.get("request_timeout", 120.0),
        max_tokens=cfg.get("max_tokens", 1024),
        max_retries=cfg.get("max_retries", 2),
        concurrency=concurrency,
    )

    asyncio.run(generator.process())