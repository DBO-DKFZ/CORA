"""
Convert old snowflake-format retrieved-docs JSON files to the new agentic-RAG
format expected by run_answerer.py.

Old retrieval_result shape:
  {
    "documents": [...],
    "strategy": "...",
    "retrieval_history": [
      {"iteration": N, "docs_retrieved": M, "total_docs": T,
       "critique": {"sufficient": bool, "confidence": float, "gaps": [...], ...}},
      ...
    ],
    "total_iterations": N,
    "final_doc_count": N,
    ["reranker_scores": [...]]   # reranked variant only
  }

New retrieval_result shape (required by run_answerer.py):
  {
    "conditions": [],
    "retrieval_history": [
      {"step": "iter_N", "query": "", "sufficient": bool, "gap_query": ""},
      ...
    ],
    "documents": [...],
    "final_sufficient": true
  }
"""

import argparse
import json
import os


def convert_history(old_history: list) -> list:
    new_history = []
    for h in (old_history or []):
        if not isinstance(h, dict):
            continue
        iteration = h.get("iteration", 0)
        critique = h.get("critique") or {}
        new_history.append({
            "step": f"iter_{iteration}",
            "query": critique.get("reasoning", "")[:120] if critique.get("reasoning") else "",
            "sufficient": bool(critique.get("sufficient", False)),
            "gap_query": "",
        })
    return new_history


def convert_record(record: dict) -> dict:
    rr = record.get("retrieval_result") or {}

    old_history = rr.get("retrieval_history") or []
    new_history = convert_history(old_history)

    documents = rr.get("documents") or []

    # A record is "sufficient" if any iteration was sufficient, or if there
    # are documents and no iteration history (should not happen in practice).
    any_suf = any(h.get("sufficient") for h in new_history)
    final_sufficient = any_suf or (len(documents) > 0 and len(new_history) == 0)

    new_rr = {
        "conditions": [],
        "retrieval_history": new_history,
        "documents": documents,
        "final_sufficient": final_sufficient,
    }

    return {
        "question_id": record.get("question_id"),
        "question": record.get("question"),
        "question_prompt": record.get("question_prompt", "").rstrip("\n"),
        "answer_options": record.get("answer_options"),
        "correct_choice": record.get("correct_choice"),
        "answer_source": record.get("answer_source"),
        "retrieval_result": new_rr,
    }


def convert_dir(input_dir: str, output_dir: str, include_insufficient: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))

    kept = skipped = 0
    for fname in files:
        in_path = os.path.join(input_dir, fname)
        with open(in_path, "r", encoding="utf-8") as fh:
            record = json.load(fh)

        new_record = convert_record(record)

        if not new_record["retrieval_result"]["final_sufficient"]:
            if not include_insufficient:
                skipped += 1
                continue
            # Keep but mark — run_answerer's _load_rag_records filters these out
            # unless include_insufficient forces final_sufficient=True
            new_record["retrieval_result"]["final_sufficient"] = True

        out_path = os.path.join(output_dir, fname)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(new_record, fh, ensure_ascii=False, indent=2)
        kept += 1

    total = kept + skipped
    print(f"{input_dir} → {output_dir}: {kept}/{total} records written"
          + (f" ({skipped} skipped, insufficient)" if skipped else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert snowflake format to new agentic-RAG format")
    parser.add_argument("input_dir", help="Directory with old-format JSON files")
    parser.add_argument("output_dir", help="Directory to write new-format JSON files")
    parser.add_argument(
        "--include-insufficient",
        action="store_true",
        default=False,
        help="Include records where no retrieval iteration was sufficient "
             "(sets final_sufficient=True so run_answerer.py loads them)",
    )
    args = parser.parse_args()
    convert_dir(args.input_dir, args.output_dir, args.include_insufficient)
