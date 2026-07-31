"""
LLM judge for citation quality and faithfulness evaluation.

For each case with cited sources the judge evaluates:

  Per cited document:
    "Does this source contain information that supports the model's answer?"
    → Supports | Partially supports | Irrelevant

  Per case overall:
    "Is the model's answer derivable from these cited sources alone?"
    → Yes | No

Usage:
    python run_faithfulness_judge.py --config configs/faithfulness_judge.yaml
    python run_faithfulness_judge.py --input_csv results/results_llama4_reranked_rag.csv \
                                     --output_csv results/faithfulness_llama4_reranked_rag.csv \
                                     --model openai/gpt-oss-120b --provider together

Output CSV columns:
    question_id, question, llm_response, correct_choice, is_correct,
    used_sources, n_cited, doc_ratings (JSON), faithfulness,
    faithfulness_explanation, raw_response
"""

import json
import os
import re
import argparse
import yaml
import pandas as pd
import tqdm
from openai import OpenAI
from together import Together
from dotenv import load_dotenv


SYSTEM_PROMPT = """\
You are a medical evidence evaluator assessing the quality of citations in a RAG system.

You will be given:
- A multiple-choice dermatology question
- The answer chosen by an LLM
- The sources the LLM cited to support its answer (document text included)

Your task has two parts:

PART 1 — Per document: for each cited source, rate whether it contains information \
that supports the model's chosen answer.
Use exactly one of: Supports | Partially supports | Irrelevant

PART 2 — Overall faithfulness: judge whether the model's answer is derivable from \
the cited sources alone, without needing outside medical knowledge.
Use exactly one of: Yes | No

Respond with ONLY a JSON object matching this schema (no other text):
{
  "doc_ratings": [
    {
      "doc_id": <integer, 1-based>,
      "rating": "<Supports | Partially supports | Irrelevant>",
      "explanation": "<one sentence>"
    }
  ],
  "faithfulness": "<Yes | No>",
  "faithfulness_explanation": "<one or two sentences>"
}"""

DOC_RATINGS   = ["Supports", "Partially supports", "Irrelevant"]
FAITHFUL_VALS = ["Yes", "No"]


def parse_cited_indices(used_sources: str) -> list[int]:
    return sorted(set(int(m) for m in re.findall(r"Document ID (\d+)", str(used_sources))))


def build_prompt(row: pd.Series, cited_docs: list[dict]) -> str:
    try:
        answer_options = json.loads(row["answer_options"])
    except Exception:
        answer_options = {}

    resp = str(row.get("llm_response", "")).strip().upper()
    letter = resp[0] if resp else "?"
    answer_text = answer_options.get(letter, "")
    answer_display = f"{letter}. {answer_text}" if answer_text else letter

    options_block = "\n".join(f"{k}. {v}" for k, v in answer_options.items())
    docs_block = "\n\n".join(
        f"[Cited Document {d['doc_id']}]\n{d['text']}" for d in cited_docs
    )

    return (
        f"QUESTION:\n{row['question']}\n\n"
        f"ANSWER OPTIONS:\n{options_block}\n\n"
        f"MODEL'S ANSWER: {answer_display}\n\n"
        f"CITED SOURCES:\n{docs_block}"
    )


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
            {"role": "user",   "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def parse_response(raw: str) -> tuple[list, str, str]:
    """Returns (doc_ratings, faithfulness, faithfulness_explanation)."""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(clean)
        doc_ratings = parsed.get("doc_ratings", [])
        faithfulness = parsed.get("faithfulness", "")
        explanation  = parsed.get("faithfulness_explanation", "")
        # Normalise ratings in case the model drifts slightly
        for d in doc_ratings:
            raw_r = d.get("rating", "")
            matched = next((r for r in DOC_RATINGS if r.lower() in raw_r.lower()), raw_r)
            d["rating"] = matched
        if faithfulness not in FAITHFUL_VALS:
            faithfulness = next((v for v in FAITHFUL_VALS if v.lower() in faithfulness.lower()), faithfulness)
        return doc_ratings, faithfulness, explanation
    except json.JSONDecodeError:
        return [], "", raw


def is_correct(row: pd.Series) -> bool:
    resp    = str(row.get("llm_response", "")).strip().upper()
    correct = str(row.get("correct_choice", "")).strip().upper()
    return (resp[0] if resp else "") == (correct[0] if correct else "")


def load_checkpoint(output_csv: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        done_ids = set(existing["question_id"].astype(str))
        todo = df[~df["question_id"].astype(str).isin(done_ids)].copy()
        print(f"Resuming: {len(done_ids)} done, {len(todo)} remaining.")
        return existing, todo
    return pd.DataFrame(), df.copy()


def save_checkpoint(output_csv: str, existing: pd.DataFrame, new_rows: list[dict]) -> None:
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined = combined.drop_duplicates(subset=["question_id"], keep="last")
    combined.to_csv(output_csv, index=False)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     help="YAML config file")
    parser.add_argument("--input_csv",  help="Results CSV to evaluate")
    parser.add_argument("--output_csv", help="Output CSV path")
    parser.add_argument("--model",      help="Judge model identifier")
    parser.add_argument("--provider",   choices=["openai", "together", "local"])
    parser.add_argument("--top_k_docs",      type=int, help="Max cited docs to include per case")
    parser.add_argument("--only_sufficient", action="store_true", default=None,
                        help="Only judge cases where final_sufficient == True")
    parser.add_argument("--sufficient_source",
                        help="Dir of per-question JSONs or a CSV with final_sufficient column")
    parser.add_argument("--verbose",         action="store_true", default=None)
    args = parser.parse_args()

    cfg: dict = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    def resolve(key, default=None):
        val = getattr(args, key, None)
        if val is not None and val is not False:
            return val
        return cfg.get(key, default)

    input_csv  = resolve("input_csv")
    output_csv = resolve("output_csv")
    model      = resolve("model", "openai/gpt-oss-120b")
    provider   = resolve("provider", "together")
    top_k_docs      = resolve("top_k_docs")  # None = use all cited docs
    only_sufficient   = bool(resolve("only_sufficient", False))
    sufficient_source = resolve("sufficient_source")
    verbose           = bool(resolve("verbose", False))

    if not input_csv or not output_csv:
        parser.error("Provide --input_csv and --output_csv (or --config with those keys).")

    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "together":
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    else:
        client = OpenAI(
            base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1"),
            api_key=os.getenv("LOCAL_API_KEY", "local-dev-key"),
        )

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["llm_response"])
    df["question_id"] = df["question_id"].astype(str)

    # Keep only rows where the model cited at least one source
    has_sources = (
        df["used_sources"].notna()
        & (df["used_sources"].astype(str).str.strip().str.lower() != "none")
        & (df["used_sources"].astype(str).str.strip() != "")
    )
    df = df[has_sources].copy()
    print(f"Rows with cited sources: {len(df)}")

    if only_sufficient:
        before = len(df)
        if sufficient_source and os.path.isdir(sufficient_source):
            sufficient_ids = set()
            for fname in os.listdir(sufficient_source):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(sufficient_source, fname)) as fh:
                    rec = json.load(fh)
                if rec.get("retrieval_result", {}).get("final_sufficient"):
                    sufficient_ids.add(str(rec["question_id"]))
            df = df[df["question_id"].isin(sufficient_ids)].copy()
            print(f"Filtered to final_sufficient=True (from {sufficient_source}): {len(df)} / {before} rows")
        elif sufficient_source and os.path.isfile(sufficient_source):
            ref = pd.read_csv(sufficient_source)
            if "final_sufficient" not in ref.columns:
                raise ValueError(f"'final_sufficient' column not found in {sufficient_source}")
            sufficient_ids = set(
                ref.loc[ref["final_sufficient"].astype(str).str.strip().str.lower() == "true", "question_id"].astype(str)
            )
            df = df[df["question_id"].isin(sufficient_ids)].copy()
            print(f"Filtered to final_sufficient=True (from {sufficient_source}): {len(df)} / {before} rows")
        else:
            if "final_sufficient" not in df.columns:
                raise ValueError("only_sufficient=True but no sufficient_source given and 'final_sufficient' column not found in input CSV.")
            df = df[df["final_sufficient"].astype(str).str.strip().str.lower() == "true"].copy()
            print(f"Filtered to final_sufficient=True: {len(df)} / {before} rows")

    existing, todo = load_checkpoint(output_csv, df)

    if verbose:
        print(f"\n{'='*60}")
        print(f"INPUT:    {input_csv}  ({len(df)} rows with citations)")
        print(f"JUDGE:    {model}  [{provider}]")
        print(f"OUTPUT:   {output_csv}")
        print(f"{'='*60}\n")

    new_rows: list[dict] = []

    for _, row in tqdm.tqdm(todo.iterrows(), total=len(todo), desc="Judging faithfulness"):
        qid = row["question_id"]

        # Resolve cited docs from the retrieved_documents column
        try:
            all_docs = json.loads(row["retrieved_documents"])
        except Exception:
            all_docs = []

        cited_indices = parse_cited_indices(str(row["used_sources"]))
        cited_docs = []
        for idx in cited_indices:
            text = all_docs[idx - 1] if 0 < idx <= len(all_docs) else "(document not found)"
            cited_docs.append({"doc_id": idx, "text": text})

        if top_k_docs is not None:
            cited_docs = cited_docs[:top_k_docs]

        if not cited_docs:
            continue

        prompt = build_prompt(row, cited_docs)

        try:
            raw = call_llm(client, model, provider, prompt)
        except Exception as e:
            print(f"\n  [ERROR] [{qid}] LLM call failed: {e}")
            continue

        doc_ratings, faithfulness, faithfulness_explanation = parse_response(raw)

        new_rows.append({
            "question_id":              qid,
            "question":                 row.get("question", ""),
            "llm_response":             row.get("llm_response", ""),
            "correct_choice":           row.get("correct_choice", ""),
            "is_correct":               is_correct(row),
            "used_sources":             row.get("used_sources", ""),
            "n_cited":                  len(cited_docs),
            "doc_ratings":              json.dumps(doc_ratings),
            "faithfulness":             faithfulness,
            "faithfulness_explanation": faithfulness_explanation,
            "raw_response":             raw,
        })

        if verbose:
            precision = (
                sum(1 for d in doc_ratings if d.get("rating") in ("Supports", "Partially supports"))
                / len(doc_ratings)
                if doc_ratings else float("nan")
            )
            print(
                f"  [{qid}] {'CORRECT' if is_correct(row) else 'WRONG':7s} | "
                f"faithful={faithfulness:10s} | citation_precision={precision:.2f} "
                f"({len(cited_docs)} cited docs)"
            )

        if len(new_rows) % 20 == 0:
            save_checkpoint(output_csv, existing, new_rows)
            print(f"  [checkpoint] {len(existing) + len(new_rows)} rows saved")

    save_checkpoint(output_csv, existing, new_rows)

    result = pd.read_csv(output_csv)
    print(f"\nDone. {len(result)} rows saved to {output_csv}")

    # Summary stats
    result["is_correct"] = result["is_correct"].astype(bool)

    def citation_precision(doc_ratings_json: str) -> float:
        try:
            ratings = json.loads(doc_ratings_json)
            if not ratings:
                return float("nan")
            return sum(1 for d in ratings if d.get("rating") in ("Supports", "Partially supports")) / len(ratings)
        except Exception:
            return float("nan")

    result["citation_precision"] = result["doc_ratings"].apply(citation_precision)

    print(f"\nFaithfulness breakdown:")
    print(result["faithfulness"].value_counts().to_string())
    print(f"\nMean citation precision: {result['citation_precision'].mean():.3f}")
    print(f"\nFaithfulness × Correctness:")
    print(pd.crosstab(result["faithfulness"], result["is_correct"],
                      colnames=["is_correct"], rownames=["faithfulness"]))


if __name__ == "__main__":
    main()
