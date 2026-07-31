"""
Count tokens that would be burned per JSON file in a retrieved_docs directory.

Tokens are counted for the actual prompt sent to the answerer LLM:
  - system prompt
  - RAG context (documents) + question  (for agentic_rag records)
  - question only                        (for non-rag / insufficient records)

Usage:
    python count_tokens.py [--dir results/retrieved_docs_medqa_snowflakev2]
                           [--model gpt-4o]
                           [--sort {tokens,id}]
                           [--top N]
"""

import argparse
import json
import os

import tiktoken

SYSTEM_PROMPT = (
    "You are a medical expert assistant. "
    "Answer the multiple choice question by selecting the best answer "
    "based on the provided context and your medical knowledge. "
    "Respond with only the answer choice letter and nothing else."
)


def build_final_prompt(record: dict) -> str:
    answer_source = record.get("answer_source", "")
    retrieval_result = record.get("retrieval_result") or {}
    documents = retrieval_result.get("documents") or []
    question_prompt = record.get("question_prompt", "")

    if answer_source == "agentic_rag" and documents:
        context = "\n\n".join(f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents))
        return f"CONTEXT:\n{context}\n\nQUESTION:\n{question_prompt}"
    return question_prompt


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def process_dir(docs_dir: str, model: str, sort_by: str, top_n: int | None):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    system_tokens = count_tokens(SYSTEM_PROMPT, enc)
    results = []

    files = sorted(f for f in os.listdir(docs_dir) if f.endswith(".json"))
    print(f"Processing {len(files)} files in '{docs_dir}' using encoding for '{model}'...")

    for fname in files:
        path = os.path.join(docs_dir, fname)
        with open(path, encoding="utf-8") as f:
            record = json.load(f)

        prompt = build_final_prompt(record)
        prompt_tokens = count_tokens(prompt, enc)
        total = system_tokens + prompt_tokens

        results.append({
            "file": fname,
            "question_id": record.get("question_id", fname),
            "answer_source": record.get("answer_source", "unknown"),
            "doc_count": len((record.get("retrieval_result") or {}).get("documents") or []),
            "system_tokens": system_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total,
        })

    if sort_by == "tokens":
        results.sort(key=lambda r: r["total_tokens"], reverse=True)
    # default sort is by file name (already sorted above)

    if top_n:
        results = results[:top_n]

    # Print table
    col_w = [12, 30, 6, 10, 12, 12]
    header = f"{'question_id':<12}  {'answer_source':<30}  {'docs':>6}  {'sys_tok':>10}  {'prompt_tok':>12}  {'total_tok':>12}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{str(r['question_id']):<12}  {r['answer_source']:<30}  "
            f"{r['doc_count']:>6}  {r['system_tokens']:>10,}  "
            f"{r['prompt_tokens']:>12,}  {r['total_tokens']:>12,}"
        )

    totals = [r["total_tokens"] for r in results]
    all_results_for_stats = results  # already sliced to top_n for display, recalc on full if needed
    print(f"\nShowing {len(results)} records")
    print(f"  min  : {min(totals):,}")
    print(f"  max  : {max(totals):,}")
    print(f"  mean : {sum(totals)/len(totals):,.0f}")
    print(f"  total: {sum(totals):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count answerer LLM tokens per retrieved-docs JSON file.")
    parser.add_argument(
        "--dir",
        default="results/retrieved_docs_medqa_snowflakev2",
        help="Directory of per-question JSON files",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model name used to select the tokenizer (default: gpt-4o)",
    )
    parser.add_argument(
        "--sort",
        choices=["tokens", "id"],
        default="id",
        help="Sort output by total tokens (desc) or question id (asc)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only the top N records after sorting",
    )
    args = parser.parse_args()
    process_dir(args.dir, args.model, args.sort, args.top)
