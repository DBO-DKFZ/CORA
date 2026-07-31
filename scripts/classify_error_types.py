"""Classify GPT-5 open-ended *errors* into a small taxonomy.

For every wrong / partially-correct answer (verdict != "Correct") in the given
result CSVs, ask an LLM to label the judge's `verdict_explanation` with one of a
fixed set of error types, and write the label back into a new `error_type`
column (blank for correct rows). Designed for the paired PubMed base/RAG files so
the two can be differenced into a RAG - baseline error-taxonomy figure.

Idempotent + checkpointed: rerunning only fills rows whose error_type is still
missing, so an interrupted run resumes cleanly.
"""
import os
import time

import pandas as pd
import tqdm
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# --- config -----------------------------------------------------------------
INPUT_CSVS = [
    "results/gpt5/results_gpt5_pubmed_base.csv",
    "results/gpt5/results_gpt5_pubmed_rag.csv",
]
MODEL = "claude-sonnet-4-6"   # matches configs/answer_judge.yaml
CHECKPOINT_EVERY = 20

# Fixed taxonomy. Keys are the labels written to disk; values brief the classifier.
TAXONOMY = {
    "wrong_entity": (
        "Names a different disease/entity/answer than the reference "
        "(a substantively different answer, not merely less specific)."
    ),
    "underspecified": (
        "Correct concept but missing the key specific detail the reference "
        "requires (wrong/absent subtype, qualifier, or specificity)."
    ),
    "terminology": (
        "Correct underlying concept but stated with an imprecise, non-standard, "
        "or wrong term/wording."
    ),
    "extraneous": (
        "Correct core answer but diluted or contradicted by added unrelated, "
        "unsupported, or incorrect material."
    ),
    "reasoning_mechanism": (
        "Wrong mechanism, pathophysiology, or clinical-reasoning step underlying "
        "the answer."
    ),
    "unverifiable": (
        "The error cannot be judged from the explanation (e.g. the reference is "
        "an image, or the explanation says the answer cannot be confirmed)."
    ),
}
LABELS = list(TAXONOMY)

SYSTEM_PROMPT = (
    "You are a meticulous medical-QA error annotator. You are given a judge's "
    "explanation of why a model's answer to a dermatology question was scored "
    "wrong or only partially correct. Assign exactly ONE error-type label that "
    "best captures the PRIMARY reason the answer was not fully correct. "
    "Reply with only the label token, nothing else."
)

TAXONOMY_BLOCK = "\n".join(f"- {k}: {v}" for k, v in TAXONOMY.items())

USER_TEMPLATE = """\
Error-type labels:
{taxonomy}

Question: {question}
Reference answer: {reference}
Model's answer: {candidate}
Judge's explanation of the error: {explanation}

Return the single best-fitting label token from the list above."""


def is_error(verdict) -> bool:
    return str(verdict).strip().lower() not in ("correct", "nan", "none", "")


def classify(client, question, reference, candidate, explanation) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=16,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_TEMPLATE.format(
                taxonomy=TAXONOMY_BLOCK,
                question=str(question)[:1500],
                reference=str(reference),
                candidate=str(candidate),
                explanation=str(explanation),
            ),
        }],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip().lower()
    for lab in LABELS:                       # exact / prefix / substring, in order
        if raw == lab:
            return lab
    for lab in LABELS:
        if raw.startswith(lab) or lab in raw:
            return lab
    print(f"  [WARN] unmapped reply {raw!r} -> unverifiable")
    return "unverifiable"


def process(path, client):
    df = pd.read_csv(path)
    if "verdict_explanation" not in df.columns:
        print(f"[skip] {path}: no verdict_explanation column")
        return
    if "error_type" not in df.columns:
        df["error_type"] = ""
    df["error_type"] = df["error_type"].fillna("").astype(str)

    todo = df.index[df["verdict"].map(is_error) & (df["error_type"].str.len() == 0)]
    n_err = int(df["verdict"].map(is_error).sum())
    print(f"[{path}]  n={len(df)}  errors={n_err}  to_label={len(todo)}")

    for i, idx in enumerate(tqdm.tqdm(todo, desc=os.path.basename(path)), 1):
        row = df.loc[idx]
        expl = row.get("verdict_explanation")
        if pd.isna(expl) or not str(expl).strip():
            df.at[idx, "error_type"] = "unverifiable"
        else:
            try:
                df.at[idx, "error_type"] = classify(
                    client, row.get("question", ""), row.get("correct_choice", ""),
                    row.get("llm_response", ""), expl,
                )
            except Exception as e:
                print(f"\n  [ERROR] row {idx}: {e}")
                time.sleep(2)
                continue
        if i % CHECKPOINT_EVERY == 0:
            df.to_csv(path, index=False)
    df.to_csv(path, index=False)

    labelled = df.loc[df["verdict"].map(is_error), "error_type"]
    print("  distribution:", labelled.value_counts().to_dict())


def main():
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    for path in INPUT_CSVS:
        process(path, client)


if __name__ == "__main__":
    main()
