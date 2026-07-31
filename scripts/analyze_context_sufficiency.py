"""
Analyze context sufficiency vs. LLM correctness across one or more model result CSVs.

Usage:
    python analyze_context_sufficiency.py \
        --scores results/relevance_scores.csv \
        --models results/results_gpt4o_rag.csv results/results_gpt5mini_rag.csv \
        [--categories results/question_categories.csv]

The scores CSV must have columns: question_id, context_sufficient
Each model CSV must have columns:  question_id, correct_choice, llm_response
The optional categories CSV must have columns:
    question_id, disease_prevalence, question_type, requires_visual_reasoning
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

BOOTSTRAP_N = 1000
RNG = np.random.default_rng(42)


def bootstrap_accuracy_ci(correct: np.ndarray, n_boot: int = BOOTSTRAP_N) -> tuple[float, float]:
    """Return (lower, upper) 95% CI for accuracy via percentile bootstrap."""
    if len(correct) == 0:
        return float("nan"), float("nan")
    means = np.array([
        RNG.choice(correct, size=len(correct), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ("question_id", "context_sufficient") if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")
    df["question_id"] = df["question_id"].astype(str)
    return df[["question_id", "context_sufficient"]]


def load_categories(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["question_id"] = df["question_id"].astype(str)
    keep = ["question_id", "disease_prevalence", "question_type", "requires_visual_reasoning"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")
    df["requires_visual_reasoning"] = df["requires_visual_reasoning"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    return df[keep]


def print_2x2(df: pd.DataFrame, scored: pd.Series, label_width: int = 30) -> None:
    rows = [(2, "Sufficient (2)"), (1, "Partial (1)"), (0, "Insufficient (0)")]
    contingency = []

    print(f"  {'':>{label_width}}  {'Accuracy (95% CI)':>28}  {'Correct':>8}  {'Incorrect':>9}  {'n':>6}")
    for s, label in rows:
        grp = df[scored & (df["context_sufficient"] == s)]
        n = len(grp)
        if n == 0:
            print(f"  {label:>{label_width}}  {'—':>28}  {'—':>8}  {'—':>9}  {0:>6}")
            contingency.append([0, 0])
            continue
        correct_arr = grp["is_correct"].to_numpy().astype(int)
        n_correct = int(correct_arr.sum())
        acc = n_correct / n
        lo, hi = bootstrap_accuracy_ci(correct_arr)
        ci_str = f"{acc:.1%} [{lo:.1%}, {hi:.1%}]"
        print(f"  {label:>{label_width}}  {ci_str:>28}  {n_correct:>8}  {n - n_correct:>9}  {n:>6}")
        contingency.append([n_correct, n - n_correct])

    contingency_arr = np.array(contingency)
    contingency_arr = contingency_arr[contingency_arr.sum(axis=1) > 0]
    print()
    if contingency_arr.shape[0] >= 2:
        chi2, p, dof, expected = chi2_contingency(contingency_arr)
        min_exp = expected.min()
        print(f"  χ²({dof}) = {chi2:.3f},  p = {p:.4f}", end="")
        if min_exp < 5:
            print(f"  ⚠ min expected cell = {min_exp:.1f} (<5)", end="")
        print()
    else:
        print("  Chi-square skipped: fewer than two non-empty sufficiency levels.")


def print_accuracy_by_category(df: pd.DataFrame, scored: pd.Series,
                                col: str, title: str) -> None:
    print(f"\n--- {title} ---")
    print(f"  {'':>25}  {'Accuracy (95% CI)':>28}  {'n':>6}")
    for val in sorted(df[col].dropna().unique()):
        mask = scored & (df[col] == val)
        grp = df[mask]
        if len(grp) == 0:
            continue
        correct_arr = grp["is_correct"].to_numpy().astype(int)
        acc = correct_arr.mean()
        lo, hi = bootstrap_accuracy_ci(correct_arr)
        ci_str = f"{acc:.1%} [{lo:.1%}, {hi:.1%}]"
        print(f"  {str(val):>25}  {ci_str:>28}  {len(grp):>6}")


def print_accuracy_by_sufficiency_x_category(df: pd.DataFrame, scored: pd.Series,
                                              col: str, title: str) -> None:
    print(f"\n--- Accuracy by context sufficiency × {title} ---")
    vals = sorted(df[col].dropna().unique())
    suf_labels = {2: "suf=2", 1: "suf=1", 0: "suf=0"}
    header = f"  {'':>25}" + "".join(f"  {suf_labels[s]:>20}" for s in [2, 1, 0])
    print(header)
    for val in vals:
        row_str = f"  {str(val):>25}"
        for s in [2, 1, 0]:
            mask = scored & (df[col] == val) & (df["context_sufficient"] == s)
            grp = df[mask]
            if len(grp) == 0:
                row_str += f"  {'—':>20}"
            else:
                acc = grp["is_correct"].mean()
                lo, hi = bootstrap_accuracy_ci(grp["is_correct"].to_numpy().astype(int))
                row_str += f"  {acc:.1%} [{lo:.1%},{hi:.1%}]"
        print(row_str)


def analyze(scores: pd.DataFrame, df: pd.DataFrame, model_label: str,
            categories: pd.DataFrame | None = None) -> None:
    df = df.merge(scores, on="question_id", how="inner")
    if categories is not None:
        df = df.merge(categories, on="question_id", how="left")

    scored = df["context_sufficient"].notna()

    print(f"\n{'='*60}")
    print(f"Model results: {model_label}")
    print(f"  Questions: {len(df)}  |  scored: {scored.sum()}")

    if not scored.any():
        print("  No scored rows — skipping.")
        return

    dist = df.loc[scored, "context_sufficient"].value_counts().sort_index()
    print("\nContext sufficiency distribution:")
    for s, count in dist.items():
        pct = 100 * count / scored.sum()
        print(f"  {int(s)}: {count:4d}  ({pct:.1f}%)")
    print(f"  Mean: {df.loc[scored, 'context_sufficient'].mean():.3f}")

    df["is_correct"] = (
        df["llm_response"].astype(str).str.strip().str[0].str.upper()
        == df["correct_choice"].astype(str).str.strip().str.upper()
    )
    print(f"\nOverall LLM accuracy: {df['is_correct'].mean():.1%}  (n={len(df)})")

    print(f"\n2x2 breakdown (context sufficiency vs correctness)  [95% CI, n_boot={BOOTSTRAP_N}]:")
    print_2x2(df, scored)

    if categories is None:
        return

    # ---- Stratified analysis ----
    cat_scored = scored & df["disease_prevalence"].notna()
    if not cat_scored.any():
        print("\n  (No category data matched — skipping stratified analysis.)")
        return

    print(f"\n{'='*60}")
    print("STRATIFIED ANALYSIS")

    print_accuracy_by_category(df[cat_scored], cat_scored[cat_scored], "disease_prevalence", "Accuracy by disease prevalence")
    print_accuracy_by_category(df[cat_scored], cat_scored[cat_scored], "question_type",       "Accuracy by question type")
    print_accuracy_by_category(df[cat_scored], cat_scored[cat_scored], "requires_visual_reasoning", "Accuracy by visual reasoning requirement")

    print_accuracy_by_sufficiency_x_category(df[cat_scored], cat_scored[cat_scored], "disease_prevalence", "disease prevalence")
    print_accuracy_by_sufficiency_x_category(df[cat_scored], cat_scored[cat_scored], "question_type",       "question type")
    print_accuracy_by_sufficiency_x_category(df[cat_scored], cat_scored[cat_scored], "requires_visual_reasoning", "visual reasoning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores",     required=True,  help="CSV with question_id + context_sufficient")
    parser.add_argument("--models",     nargs="+", required=True, help="One or more model result CSVs")
    parser.add_argument("--categories", default=None,   help="Optional CSV with question categories")
    args = parser.parse_args()

    scores = load_scores(args.scores)
    categories = load_categories(args.categories) if args.categories else None

    # Load all model CSVs and drop NaN llm_response rows.
    model_dfs: dict[str, pd.DataFrame] = {}
    for path in args.models:
        df = pd.read_csv(path)
        df["question_id"] = df["question_id"].astype(str)
        missing = [c for c in ("question_id", "correct_choice", "llm_response") if c not in df.columns]
        if missing:
            raise KeyError(f"{path} is missing columns: {missing}")
        n_before = len(df)
        df = df.dropna(subset=["llm_response"])
        print(f"{path}: {n_before} rows → {len(df)} after dropping {n_before - len(df)} NaN llm_response")
        model_dfs[path] = df

    # Inner join on question_id: keep only questions present in every model CSV.
    shared_ids = set.intersection(*[set(df["question_id"]) for df in model_dfs.values()])
    print(f"\nShared question IDs across all models: {len(shared_ids)}")

    for path, df in model_dfs.items():
        analyze(scores, df[df["question_id"].isin(shared_ids)].copy(), path, categories)
