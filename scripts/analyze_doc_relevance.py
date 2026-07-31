"""
Analyze which retrieved documents are relevant to the LLM's answer for each question.

Usage (config file):
    python analyze_doc_relevance.py --config configs/doc_relevance_gptoss120b_rag.yaml

Usage (CLI flags):
    python analyze_doc_relevance.py \
        --retrieved_docs_dir results/retrieved_docs_medqa_snowflakel \
        --input_csv results/results_gptoss120b_rag.csv \
        --output_csv results/doc_relevance_gptoss120b_rag.csv \
        [--model openai/gpt-oss-120b] \
        [--provider together] \
        [--top_k_docs 10] \
        [--verbose]

Output CSV columns:
    question_id, question, llm_response, correct_choice, is_correct,
    answer_source, n_docs_retrieved, relevant_docs (JSON), relevance_summary, raw_response
"""

import json
import os
import argparse
import re
import yaml
import pandas as pd
import tqdm
from openai import OpenAI
from together import Together
from dotenv import load_dotenv


SYSTEM_PROMPT = (
    "You are a medical document relevance analyst. "
    "You will be given a multiple-choice medical question, the answer chosen by an LLM, "
    "and a set of retrieved documents.\n\n"
    "Your task: identify which retrieved documents are relevant to the LLM's answer and explain why. "
    "Focus on documents that contain information that supports or informs the chosen answer — "
    "not just documents that are broadly about the topic.\n\n"
    "Respond with a JSON object (and nothing else) matching this schema:\n"
    "{\n"
    '  "relevant_documents": [\n'
    '    {"doc_index": <1-based integer>, "relevance": "<brief explanation of why this doc supports the answer>"},\n'
    "    ...\n"
    "  ],\n"
    '  "summary": "<overall summary: what the relevant docs say and how they support or contradict the chosen answer>"\n'
    "}\n\n"
    "If no documents are relevant, return an empty relevant_documents list and explain in summary."
)


def build_prompt(row: pd.Series, docs: list[str], top_k: int) -> str:
    docs = docs[:top_k]

    try:
        answer_options = json.loads(row["answer_options"])
    except (json.JSONDecodeError, KeyError, TypeError):
        answer_options = {}

    llm_response = str(row.get("llm_response", "")).strip()
    letter_match = re.match(r"^([A-Ea-e])", llm_response)
    llm_letter = letter_match.group(1).upper() if letter_match else llm_response
    llm_text = answer_options.get(llm_letter, "")
    llm_display = f"{llm_letter}. {llm_text}" if llm_text else llm_letter

    options_block = "\n".join(f"{k}. {v}" for k, v in answer_options.items())
    docs_block = "\n\n".join(f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(docs))

    return (
        f"QUESTION:\n{row['question']}\n\n"
        f"ANSWER OPTIONS:\n{options_block}\n\n"
        f"LLM'S ANSWER: {llm_display}\n\n"
        f"RETRIEVED DOCUMENTS:\n{docs_block}"
    )


def load_docs(retrieval_dir: str, question_id) -> list[str]:
    path = os.path.join(retrieval_dir, f"{question_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("retrieval_result", {}).get("documents", [])


def call_llm(client, model: str, provider: str, prompt: str) -> str:
    extra = {}
    if provider == "openai" and "gpt-5" in model:
        extra["reasoning_effort"] = "minimal"
    else:
        extra["temperature"] = 0

    response = client.chat.completions.create(
        model=model,
        **extra,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def parse_response(raw: str) -> tuple[str, str]:
    """Parse JSON response into (relevant_docs_json_str, summary). Falls back gracefully."""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(clean)
        rel_docs = parsed.get("relevant_documents", [])
        summary = parsed.get("summary", "")
        return json.dumps(rel_docs), summary
    except json.JSONDecodeError:
        return "[]", raw


def check_correct(row: pd.Series) -> bool:
    resp = str(row.get("llm_response", "")).strip()
    letter_match = re.match(r"^([A-Ea-e])", resp)
    resp_letter = letter_match.group(1).upper() if letter_match else resp.upper()
    correct = str(row.get("correct_choice", "")).strip().upper()
    return resp_letter == correct


def load_checkpoint(output_csv: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        done_ids = set(existing["question_id"].astype(str))
        todo = df[~df["question_id"].astype(str).isin(done_ids)].copy()
        print(f"Resuming: {len(done_ids)} already processed, {len(todo)} remaining.")
        return existing, todo
    return pd.DataFrame(), df.copy()


def save_checkpoint(output_csv: str, existing: pd.DataFrame, new_rows: list[dict]) -> None:
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined = combined.drop_duplicates(subset=["question_id"], keep="last")
    combined.to_csv(output_csv, index=False)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze which retrieved docs are relevant to LLM answers")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--retrieved_docs_dir", help="Dir with per-question JSON files ({question_id}.json)")
    parser.add_argument("--input_csv", help="CSV with LLM answers (needs: question_id, question, llm_response, correct_choice, answer_options)")
    parser.add_argument("--output_csv", help="Output CSV path")
    parser.add_argument("--model", help="LLM model identifier")
    parser.add_argument("--provider", choices=["openai", "together", "local"])
    parser.add_argument("--top_k_docs", type=int, help="Max docs per question to include in prompt")
    parser.add_argument("--verbose", action="store_true", default=None)
    args = parser.parse_args()

    cfg: dict = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    def resolve(key, default=None):
        cli_val = getattr(args, key, None)
        if cli_val is not None and cli_val is not False:
            return cli_val
        return cfg.get(key, default)

    retrieved_docs_dir = resolve("retrieved_docs_dir")
    input_csv = resolve("input_csv")
    output_csv = resolve("output_csv")
    model = resolve("model", "openai/gpt-oss-120b")
    provider = resolve("provider", "together")
    top_k_docs = resolve("top_k_docs", 10)
    verbose = bool(resolve("verbose", False))

    missing = [name for name, val in [("--retrieved_docs_dir", retrieved_docs_dir), ("--input_csv", input_csv), ("--output_csv", output_csv)] if not val]
    if missing:
        parser.error(f"Missing required arguments (provide via --config or CLI): {', '.join(missing)}")

    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "together":
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    else:
        client = OpenAI(
            base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("LOCAL_API_KEY", "local-dev-key"),
        )

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["llm_response"])
    df["question_id"] = df["question_id"].astype(str)

    existing, todo = load_checkpoint(output_csv, df)

    if verbose:
        print(f"\n{'='*60}")
        print(f"LLM RESULTS:    {input_csv}  ({len(df)} rows)")
        print(f"RETRIEVAL DIR:  {retrieved_docs_dir}")
        print(f"MODEL:          {model}")
        print(f"TOP_K DOCS:     {top_k_docs}")
        print(f"OUTPUT:         {output_csv}")
        print(f"{'='*60}\n")

    new_rows: list[dict] = []

    for _, row in tqdm.tqdm(todo.iterrows(), total=len(todo), desc="Analyzing relevance"):
        qid = row["question_id"]
        docs = load_docs(retrieved_docs_dir, qid)

        if not docs:
            new_rows.append({
                "question_id": qid,
                "question": row.get("question", ""),
                "llm_response": row.get("llm_response", ""),
                "correct_choice": row.get("correct_choice", ""),
                "is_correct": check_correct(row),
                "answer_source": row.get("answer_source", ""),
                "n_docs_retrieved": 0,
                "relevant_docs": "[]",
                "relevance_summary": "No retrieved documents found for this question.",
                "raw_response": "",
            })
            if verbose:
                print(f"  [{qid}] SKIPPED — no retrieved docs")
            continue

        prompt = build_prompt(row, docs, top_k_docs)
        raw = call_llm(client, model, provider, prompt)
        relevant_docs_json, summary = parse_response(raw)

        new_rows.append({
            "question_id": qid,
            "question": row.get("question", ""),
            "llm_response": row.get("llm_response", ""),
            "correct_choice": row.get("correct_choice", ""),
            "is_correct": check_correct(row),
            "answer_source": row.get("answer_source", ""),
            "n_docs_retrieved": len(docs),
            "relevant_docs": relevant_docs_json,
            "relevance_summary": summary,
            "raw_response": raw,
        })

        if verbose:
            correct_str = "CORRECT" if check_correct(row) else "WRONG"
            try:
                n_relevant = len(json.loads(relevant_docs_json))
            except Exception:
                n_relevant = "?"
            print(f"  [{qid}] {correct_str} | {n_relevant} relevant docs of {min(len(docs), top_k_docs)}")

        if len(new_rows) % 10 == 0:
            save_checkpoint(output_csv, existing, new_rows)
            print(f"  [checkpoint] saved {len(existing) + len(new_rows)} rows to {output_csv}")

    save_checkpoint(output_csv, existing, new_rows)

    result = pd.read_csv(output_csv)
    print(f"\nDone. Results saved to {output_csv}")
    print(f"Total rows: {len(result)}")
    print(f"Correct answers: {result['is_correct'].sum()} / {len(result)} ({result['is_correct'].mean():.1%})")
    print(f"Avg relevant docs per question: {result['relevant_docs'].apply(lambda x: len(json.loads(x))).mean():.1f}")


if __name__ == "__main__":
    main()
