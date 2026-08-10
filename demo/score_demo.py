#!/usr/bin/env python3
"""
Score a demo results CSV: multiple-choice accuracy plus a per-question table.

Usage:
    python demo/score_demo.py demo/outputs/results_demo_rag.csv
    python demo/score_demo.py demo/outputs/results_demo_rag.csv demo/outputs/results_demo_base.csv
"""

import sys

import pandas as pd


def score(path: str) -> None:
    df = pd.read_csv(path)
    df["llm_response"] = df["llm_response"].astype(str).str.strip().str.upper()
    df["correct"] = df["llm_response"] == df["correct_choice"].astype(str).str.strip().str.upper()

    n = len(df)
    k = int(df["correct"].sum())
    print(f"\n=== {path} ===")
    print(f"answered: {n}   correct: {k}   accuracy: {k / n:.2%}" if n else "no rows")

    cols = ["question_id", "correct_choice", "llm_response", "correct"]
    if "final_doc_count" in df.columns:
        cols.append("final_doc_count")
    if "used_sources" in df.columns:
        cols.append("used_sources")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        score(p)
