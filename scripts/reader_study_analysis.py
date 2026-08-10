"""
Reader Study Analysis Script
Covers all analyses from Section 7.2 of the analysis protocol.

Inputs
------
responses.csv     : one row per physician × question pair
source_ratings.csv: one row per physician × question × citation triple

Notes on question-level vs physician-question-level citation support
--------------------------------------------------------------------
For Hypothesis A (unit = question), citation support variables are
aggregated across ALL physicians who rated sources for that question
(i.e., a question is "any_supports" = True if any physician rated any
citation as Supports). This removes within-question disagreement and
aligns the unit of analysis with LLM correctness, which is fixed per
question. The responses.csv columns `any_source_supports` and
`all_sources_support` are physician-question level and are used for
Hypotheses B and C where the unit is the physician-question pair.

LLM-as-judge validation
------------------------
The supplied source_ratings.csv contains physician-generated ratings
only (the `rating`/`supports_llm` columns are identical, the latter
being the boolean encoding of the former). If a separate LLM-judge
rating file is provided, pass its path via --llm_judge_csv. The file
must have columns: question_id, source_position, llm_judge_rating
(values: 'Supports' / 'Does not support'). Without it the kappa
analysis is skipped with an informative message.

Agentic vs. simple RAG comparison
-----------------------------------
Requires a second responses/source_ratings CSV pair for the comparison
system, supplied via --alt_responses_csv and --alt_source_ratings_csv.
Skipped with a message if not provided, or if kappa < 0.6.
"""

from __future__ import annotations

import argparse
import warnings
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Binomial

warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)

# ─────────────────────────── CLI ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Reader study statistical analysis")
    p.add_argument("--responses",        default="./results/reader_study/responses.csv")
    p.add_argument("--source_ratings",   default="./results/reader_study/source_ratings.csv")
    p.add_argument("--llm_judge_csv",    default=None,
                   help="Optional CSV with LLM-judge ratings for kappa validation")
    p.add_argument("--alt_responses_csv",     default=None,
                   help="Optional responses CSV for the comparison RAG system")
    p.add_argument("--alt_source_ratings_csv", default=None,
                   help="Optional source_ratings CSV for the comparison RAG system")
    p.add_argument("--bootstrap_n",  type=int, default=10_000)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()

# ─────────────────────────── helpers ────────────────────────────────────────

SEP = "─" * 70

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def bootstrap_mean_ci(values: np.ndarray, n: int = 10_000,
                      seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, lower_95, upper_95) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n)]
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def logistic_any_vs_none(df: pd.DataFrame, dv_col: str,
                         iv_col: str = "any_source_supports") -> dict:
    """
    Binary logistic regression:
      IV = iv_col  (True = ≥1 citation supports, False = none)
      DV = dv_col  (binary)
    Returns dict with OR, 95 % CI, z-stat, p-value.
    """
    sub = df[[dv_col, iv_col]].dropna()
    sub = sub.astype({dv_col: int, iv_col: int})
    y = sub[dv_col]
    X = sm.add_constant(sub[iv_col])
    try:
        res = sm.Logit(y, X).fit(disp=False)
        coef = res.params[iv_col]
        ci   = res.conf_int().loc[iv_col]
        return dict(
            OR=np.exp(coef),
            CI_lower=np.exp(ci[0]),
            CI_upper=np.exp(ci[1]),
            z_stat=res.tvalues[iv_col],
            p_value=res.pvalues[iv_col],
            n=len(sub),
            converged=res.mle_retvals.get("converged", True),
        )
    except Exception as exc:
        return dict(error=str(exc))


def logistic_all_vs_partial(df: pd.DataFrame, dv_col: str,
                             iv_col: str = "all_sources_support") -> dict:
    """
    Restrict to rows where ANY source supports (i.e. IV for hyp A2/B2/C2):
      IV = iv_col  (True = ALL support, False = ≥1 does not support)
      DV = dv_col  (binary)
    """
    sub = df[df["any_source_supports"]].copy() if "any_source_supports" in df.columns \
          else df.copy()
    sub = sub[[dv_col, iv_col]].dropna()
    sub = sub.astype({dv_col: int, iv_col: int})
    y = sub[dv_col]
    X = sm.add_constant(sub[iv_col])
    try:
        res = sm.Logit(y, X).fit(disp=False)
        coef = res.params[iv_col]
        ci   = res.conf_int().loc[iv_col]
        return dict(
            OR=np.exp(coef),
            CI_lower=np.exp(ci[0]),
            CI_upper=np.exp(ci[1]),
            z_stat=res.tvalues[iv_col],
            p_value=res.pvalues[iv_col],
            n=len(sub),
            converged=res.mle_retvals.get("converged", True),
        )
    except Exception as exc:
        return dict(error=str(exc))


def gee_any_vs_none(df: pd.DataFrame, dv_col: str,
                    iv_col: str = "any_source_supports",
                    group_col: str = "pid") -> dict:
    """
    GEE logistic regression with cluster-robust SEs (prereg B1/C1):
      IV = iv_col  (True = ≥1 citation supports, False = none)
      DV = dv_col  (binary), clustered by group_col (reader).
    Responses are nested within readers; clustering by question gives a
    near-identical result (verified), so reader-clustering is reported.
    """
    sub = df[[dv_col, iv_col, group_col]].dropna()
    sub = sub.astype({dv_col: int, iv_col: int})
    y = sub[dv_col]
    X = sm.add_constant(sub[iv_col])
    try:
        res = GEE(y, X, groups=sub[group_col], family=Binomial(),
                  cov_struct=Independence()).fit()
        coef = res.params[iv_col]
        ci   = res.conf_int().loc[iv_col]
        return dict(
            OR=np.exp(coef),
            CI_lower=np.exp(ci[0]),
            CI_upper=np.exp(ci[1]),
            z_stat=res.tvalues[iv_col],
            p_value=res.pvalues[iv_col],
            n=len(sub),
            n_clusters=sub[group_col].nunique(),
            converged=res.converged if hasattr(res, "converged") else True,
        )
    except Exception as exc:
        return dict(error=str(exc))


def gee_all_vs_partial(df: pd.DataFrame, dv_col: str,
                       iv_col: str = "all_sources_support",
                       group_col: str = "pid") -> dict:
    """GEE analogue of logistic_all_vs_partial (prereg B2/C2), restricted
    to rows where ANY source supports, clustered by group_col (reader)."""
    sub = df[df["any_source_supports"]].copy() if "any_source_supports" in df.columns \
          else df.copy()
    sub = sub[[dv_col, iv_col, group_col]].dropna()
    sub = sub.astype({dv_col: int, iv_col: int})
    y = sub[dv_col]
    X = sm.add_constant(sub[iv_col])
    try:
        res = GEE(y, X, groups=sub[group_col], family=Binomial(),
                  cov_struct=Independence()).fit()
        coef = res.params[iv_col]
        ci   = res.conf_int().loc[iv_col]
        return dict(
            OR=np.exp(coef),
            CI_lower=np.exp(ci[0]),
            CI_upper=np.exp(ci[1]),
            z_stat=res.tvalues[iv_col],
            p_value=res.pvalues[iv_col],
            n=len(sub),
            n_clusters=sub[group_col].nunique(),
            converged=res.converged if hasattr(res, "converged") else True,
        )
    except Exception as exc:
        return dict(error=str(exc))


def print_logistic(label: str, result: dict):
    if "error" in result:
        print(f"  {label}: ERROR — {result['error']}")
        return
    conv_note = "" if result.get("converged", True) else " [!convergence warning]"
    print(f"  {label}:")
    n_line = f"    n = {result['n']}"
    if "n_clusters" in result:
        n_line += f"  (clusters = {result['n_clusters']})"
    print(n_line)
    print(f"    OR = {result['OR']:.3f}  "
          f"95 % CI [{result['CI_lower']:.3f}, {result['CI_upper']:.3f}]")
    print(f"    z = {result['z_stat']:.3f},  p = {result['p_value']:.4f}{conv_note}")


def sig_flag(p: float, alpha: float = 0.05) -> str:
    return "  *** SIGNIFICANT — run conditional hypothesis ***" if p < alpha \
           else "  (not significant — conditional hypothesis skipped)"

# ─────────────────────────── data loading ───────────────────────────────────

def load_data(args):
    resp = pd.read_csv(args.responses)
    src  = pd.read_csv(args.source_ratings)

    # Derive question-level citation support by aggregating over ALL raters
    q_cite = (src.groupby("question_id")
                 .agg(q_any_supports=("supports_llm", "any"),
                      q_all_supports=("supports_llm", "all"))
                 .reset_index())
    resp = resp.merge(q_cite, on="question_id", how="left")

    # Physician-level delta accuracy
    resp["delta"] = resp["final_correct"].astype(int) - resp["initial_correct"].astype(int)

    # Beneficial change: was initially wrong, now correct
    resp["beneficial"] = (~resp["initial_correct"]) & resp["final_correct"]
    # Harmful change: was initially correct, now wrong
    resp["harmful"]    = resp["initial_correct"]    & (~resp["final_correct"])

    return resp, src, q_cite


# ─────────────────────────── analyses ───────────────────────────────────────

def primary_analysis(resp: pd.DataFrame, bootstrap_n: int, seed: int):
    section("PRIMARY: Delta Accuracy > 0  (Wilcoxon signed-rank, one-sided)")

    per_phys = resp.groupby("pid")["delta"].mean().reset_index()
    per_phys.columns = ["pid", "mean_delta"]

    print(f"\n  Per-physician mean delta accuracy:")
    for _, row in per_phys.iterrows():
        bar = "+" * int(abs(row.mean_delta) * 32) if row.mean_delta >= 0 \
              else "-" * int(abs(row.mean_delta) * 32)
        print(f"    {row.pid}:  {row.mean_delta:+.4f}  {bar}")

    deltas = per_phys["mean_delta"].values
    stat, p = stats.wilcoxon(deltas, alternative="greater")

    print(f"\n  n physicians = {len(deltas)}")
    print(f"  Mean delta   = {deltas.mean():+.4f}")
    print(f"  Median delta = {np.median(deltas):+.4f}")
    print(f"  W = {stat:.1f},  p (one-sided) = {p:.4f}")

    if p < 0.05:
        print("  *** SIGNIFICANT — LLM assistance improved diagnostic accuracy ***")
    else:
        print("  (not significant at α = 0.05)")

    return p


def descriptive_change_rates(resp: pd.DataFrame, bootstrap_n: int, seed: int):
    section("DESCRIPTIVE: Beneficial & Harmful Change Rates")

    # Beneficial: denominator = initially incorrect pairs
    initially_wrong = resp[~resp["initial_correct"]]
    ben_vals = initially_wrong.groupby("pid").apply(
        lambda g: g["beneficial"].mean(), include_groups=False
    ).values
    mean_b, lo_b, hi_b = bootstrap_mean_ci(ben_vals, n=bootstrap_n, seed=seed)
    print(f"\n  Beneficial change rate  (wrong → correct)")
    print(f"    n physicians = {len(ben_vals)}")
    print(f"    Mean = {mean_b:.4f}  95 % CI [{lo_b:.4f}, {hi_b:.4f}]")

    # Harmful: denominator = initially correct pairs
    initially_right = resp[resp["initial_correct"]]
    harm_vals = initially_right.groupby("pid").apply(
        lambda g: g["harmful"].mean(), include_groups=False
    ).values
    mean_h, lo_h, hi_h = bootstrap_mean_ci(harm_vals, n=bootstrap_n, seed=seed)
    print(f"\n  Harmful change rate  (correct → wrong)")
    print(f"    n physicians = {len(harm_vals)}")
    print(f"    Mean = {mean_h:.4f}  95 % CI [{lo_h:.4f}, {hi_h:.4f}]")


def descriptive_citation_metrics(src: pd.DataFrame, bootstrap_n: int, seed: int):
    section("DESCRIPTIVE: Citation Precision & Citation Support Rate")

    # Citation precision: proportion of cited docs rated 'Supports', per question
    q_prec = (src.groupby("question_id")
                 .apply(lambda g: g["supports_llm"].mean(), include_groups=False)
                 .values)
    mean_p, lo_p, hi_p = bootstrap_mean_ci(q_prec, n=bootstrap_n, seed=seed)
    print(f"\n  Citation precision  (proportion of docs rated Supports per question)")
    print(f"    n questions = {len(q_prec)}")
    print(f"    Mean = {mean_p:.4f}  95 % CI [{lo_p:.4f}, {hi_p:.4f}]")

    # Citation support rate: proportion of questions where ≥1 doc rated Supports
    q_any = src.groupby("question_id")["supports_llm"].any().astype(int).values
    mean_s, lo_s, hi_s = bootstrap_mean_ci(q_any, n=bootstrap_n, seed=seed)
    print(f"\n  Citation support rate  (≥1 citation rated Supports per question)")
    print(f"    n questions = {len(q_any)}")
    print(f"    Proportion = {mean_s:.4f}  95 % CI [{lo_s:.4f}, {hi_s:.4f}]")


def _agreement_stats(df: pd.DataFrame, unit_cols, cat_col: str) -> dict:
    """Inter-rater agreement for a binary rating across a rotating set of
    physicians. The design is not fully crossed (each unit is rated by a
    rotating ~3 physicians, not a fixed pair), so Cohen's kappa does not
    apply. We report:

      - raw pairwise agreement : mean over units of the proportion of
        agreeing rater PAIRS (uses every unit with >=2 raters)
      - Fleiss' kappa          : chance-corrected agreement, computed on the
        balanced subset of units with the modal number of raters (Fleiss
        requires an equal rater count per unit)
      - Gwet's AC1             : chance-corrected but robust to skewed
        prevalence (the 'kappa paradox'); uses every unit with >=2 raters

    Returns a dict of the summary numbers.
    """
    cats = sorted(df[cat_col].dropna().unique())
    rows = []
    for _, sub in df.groupby(list(unit_cols))[cat_col]:
        vc = sub.value_counts()
        rows.append([vc.get(c, 0) for c in cats])
    counts = np.array(rows)

    n = counts.sum(1)
    counts = counts[n >= 2]            # need >=2 raters to measure agreement
    n = counts.sum(1)
    q = counts.shape[1]

    # raw pairwise agreement
    pa_unit = (counts * (counts - 1)).sum(1) / (n * (n - 1))
    raw = pa_unit.mean()

    # Gwet's AC1
    pi = (counts / n[:, None]).mean(0)
    pe_gwet = (pi * (1 - pi)).sum() / (q - 1)
    ac1 = (raw - pe_gwet) / (1 - pe_gwet)

    # Fleiss' kappa on the balanced (modal rater count) subset
    modal = int(np.bincount(n).argmax())
    bal = counts[n == modal]
    if bal.shape[0] > 1:
        kappa = fleiss_kappa(bal, method="fleiss")
    else:
        kappa = float("nan")

    return {
        "n_units":   counts.shape[0],
        "modal":     modal,
        "n_balanced": bal.shape[0],
        "cats":      cats,
        "prev":      counts.sum(0) / counts.sum(),
        "raw":       raw,
        "kappa":     kappa,
        "ac1":       ac1,
    }


def inter_rater_agreement(resp: pd.DataFrame, src: pd.DataFrame):
    section("INTER-RATER AGREEMENT: Physician Citation-Support Ratings")

    print("\n  Design is not fully crossed (rotating raters per unit), so")
    print("  Fleiss' kappa / Gwet's AC1 are used, not Cohen's kappa.")
    print("  Fleiss' kappa is computed on units with the modal rater count;")
    print("  raw agreement and AC1 use all units rated by >=2 physicians.")

    specs = [
        ("Source-level: supports_llm  (per question x citation)",
         src,  ["question_id", "source_position"], "supports_llm"),
        ("Question-level: any_source_supports  (>=1 citation supports)",
         resp, ["question_id"], "any_source_supports"),
        ("Question-level: all_sources_support  (all citations support)",
         resp, ["question_id"], "all_sources_support"),
    ]
    for label, df, unit_cols, cat_col in specs:
        r = _agreement_stats(df, unit_cols, cat_col)
        prev = ", ".join(f"{c}={p:.2f}" for c, p in zip(r["cats"], r["prev"]))
        print(f"\n  {label}")
        print(f"    units (>=2 raters) = {r['n_units']}   "
              f"Fleiss subset = {r['n_balanced']} units x {r['modal']} raters")
        print(f"    prevalence: {prev}")
        print(f"    raw pairwise agreement = {r['raw']:.3f}")
        print(f"    Fleiss' kappa          = {r['kappa']:.3f}")
        print(f"    Gwet's AC1             = {r['ac1']:.3f}")


def hypothesis_a(resp: pd.DataFrame):
    section("HYPOTHESIS A: Citation Support → LLM Correctness  (question-level)")

    # Prereg A1: collapse per-reader support to a question-level label by
    # MAJORITY VOTE — a question is "supported" when a majority of its raters
    # rated ≥1 citation as Supports (mean(any_source_supports) >= 0.5).
    # Restricted to questions answered by exactly 3 physicians (the design's
    # per-question rater count); 2-rater questions have no majority and are
    # excluded. LLM correctness is fixed per question, so the unit is the
    # question and a plain logistic regression (Wald test) is appropriate —
    # no clustering needed.
    g = resp.groupby("question_id")
    q_df = pd.DataFrame({
        "n_raters":       g.size(),
        "support_frac":   g["any_source_supports"].mean(),
        "all_frac":       g["all_sources_support"].mean(),
        "llm_is_correct": g["llm_is_correct"].first().astype(int),
    }).reset_index()

    n_excluded = int((q_df["n_raters"] != 3).sum())
    q3 = q_df[q_df["n_raters"] == 3].copy()
    q3["maj_any_support"] = (q3["support_frac"] >= 0.5).astype(int)
    q3["maj_all_support"] = (q3["all_frac"]     >= 0.5).astype(int)
    q3["any_source_supports"] = q3["maj_any_support"].astype(bool)  # for A2 helper

    print(f"\n  Questions with exactly 3 raters: {len(q3)} "
          f"(excluded {n_excluded} with ≠3 raters)")

    print("\n  Hyp. A1: majority citation support vs. LLM correctness  (logistic, Wald)")
    r_a1 = logistic_any_vs_none(q3, dv_col="llm_is_correct",
                                iv_col="maj_any_support")
    print_logistic("A1", r_a1)
    flag = sig_flag(r_a1.get("p_value", 1.0))
    print(flag)

    if r_a1.get("p_value", 1.0) < 0.05:
        print("\n  Hyp. A2 (conditional): majority ALL-support vs. LLM correctness")
        r_a2 = logistic_all_vs_partial(q3, dv_col="llm_is_correct",
                                       iv_col="maj_all_support")
        print_logistic("A2", r_a2)


def hypothesis_b(resp: pd.DataFrame):
    section("HYPOTHESIS B: Citation Support → Physician Post-Exposure Correctness\n"
            "         (response level, GEE clustered by reader)")

    print("\n  Hyp. B1: any citation support vs. physician post-exposure correctness")
    r_b1 = gee_any_vs_none(resp, dv_col="final_correct",
                           iv_col="any_source_supports")
    print_logistic("B1", r_b1)
    flag = sig_flag(r_b1.get("p_value", 1.0))
    print(flag)

    if r_b1.get("p_value", 1.0) < 0.05:
        print("\n  Hyp. B2 (conditional): all citations support vs. physician correctness")
        r_b2 = gee_all_vs_partial(resp, dv_col="final_correct",
                                  iv_col="all_sources_support")
        print_logistic("B2", r_b2)


def hypothesis_c(resp: pd.DataFrame):
    section("HYPOTHESIS C: Citation Support → Physician Correctness\n"
            "         (pre-exposure INCORRECT responses only, GEE clustered by reader)")

    pre_wrong = resp[~resp["initial_correct"]].copy()
    print(f"  Restricted to pre-incorrect responses: n = {len(pre_wrong)}")

    print("\n  Hyp. C1: any citation support vs. physician correctness (pre-incorrect)")
    r_c1 = gee_any_vs_none(pre_wrong, dv_col="final_correct",
                           iv_col="any_source_supports")
    print_logistic("C1", r_c1)
    flag = sig_flag(r_c1.get("p_value", 1.0))
    print(flag)

    if r_c1.get("p_value", 1.0) < 0.05:
        print("\n  Hyp. C2 (conditional): all citations support vs. physician correctness"
              " (pre-incorrect)")
        r_c2 = gee_all_vs_partial(pre_wrong, dv_col="final_correct",
                                  iv_col="all_sources_support")
        print_logistic("C2", r_c2)


def llm_judge_validation(src: pd.DataFrame, llm_judge_csv: str | None):
    section("EXPLORATORY: LLM-as-Judge Validation  (Weighted Cohen's κ per physician)")

    if llm_judge_csv is None:
        print("\n  No LLM-judge CSV supplied (--llm_judge_csv).")
        print("  Skipping kappa analysis.")
        print("  Expected format: question_id, source_position, llm_judge_rating")
        print("  (values: 'Supports' / 'Does not support')")
        return None

    try:
        judge = pd.read_csv(llm_judge_csv)
    except Exception as e:
        print(f"\n  Could not read LLM-judge CSV: {e}")
        return None

    required = {"question_id", "source_position", "llm_judge_rating"}
    if not required.issubset(judge.columns):
        print(f"\n  LLM-judge CSV missing columns: {required - set(judge.columns)}")
        return None

    merged = src.merge(judge, on=["question_id", "source_position"], how="inner")
    if merged.empty:
        print("\n  No matching rows between source_ratings and LLM-judge CSV.")
        return None

    # Map to integer labels for kappa
    label_map = {"Supports": 1, "Does not support": 0}
    merged["human_bin"] = merged["rating"].map(label_map)
    merged["judge_bin"] = merged["llm_judge_rating"].map(label_map)

    kappas = {}
    print(f"\n  n physicians = {merged['pid'].nunique()}")
    print(f"  n question-citation pairs = {len(merged)}\n")

    for pid, grp in merged.groupby("pid"):
        if grp["human_bin"].nunique() < 2 or grp["judge_bin"].nunique() < 2:
            k = np.nan
            note = " (insufficient variance — kappa undefined)"
        else:
            k = cohen_kappa_score(grp["human_bin"], grp["judge_bin"],
                                  weights="linear")
            note = ""
        kappas[pid] = k
        print(f"  {pid}: κ = {k:.3f}{note}" if not np.isnan(k)
              else f"  {pid}: κ = NaN{note}")

    valid_kappas = [v for v in kappas.values() if not np.isnan(v)]
    mean_k = np.mean(valid_kappas) if valid_kappas else np.nan
    print(f"\n  Mean κ across physicians = {mean_k:.3f}")

    if mean_k >= 0.6:
        print("  κ ≥ 0.6 — proceed to agentic vs. simple RAG comparison")
    else:
        print("  κ < 0.6 — agentic vs. simple RAG comparison NOT warranted")

    return mean_k


def agentic_vs_simple_rag(src_main: pd.DataFrame,
                           alt_responses_csv: str | None,
                           alt_source_ratings_csv: str | None,
                           mean_kappa: float | None,
                           bootstrap_n: int, seed: int):
    section("EXPLORATORY: Agentic vs. Simple RAG Evidence-Grounding\n"
            "         (conditional on κ ≥ 0.6)")

    if mean_kappa is None or np.isnan(mean_kappa):
        print("\n  Kappa not available — skipping.")
        return

    if mean_kappa < 0.6:
        print(f"\n  Mean κ = {mean_kappa:.3f} < 0.6 — analysis not warranted.")
        return

    if alt_responses_csv is None or alt_source_ratings_csv is None:
        print("\n  No alternative system data supplied "
              "(--alt_responses_csv, --alt_source_ratings_csv).")
        print("  Skipping comparison.")
        return

    try:
        src_alt = pd.read_csv(alt_source_ratings_csv)
    except Exception as e:
        print(f"\n  Could not load alt source_ratings: {e}")
        return

    def citation_precision_per_q(src_df):
        return (src_df.groupby("question_id")["supports_llm"]
                      .mean()
                      .rename("citation_precision"))

    def citation_support_rate_per_q(src_df):
        return (src_df.groupby("question_id")["supports_llm"]
                      .any()
                      .astype(int)
                      .rename("citation_support_rate"))

    prec_main = citation_precision_per_q(src_main)
    prec_alt  = citation_precision_per_q(src_alt)
    supp_main = citation_support_rate_per_q(src_main)
    supp_alt  = citation_support_rate_per_q(src_alt)

    # Align on common question_ids
    common_q = list(set(prec_main.index) & set(prec_alt.index))
    if len(common_q) < 2:
        print(f"\n  Only {len(common_q)} common questions — cannot compare.")
        return

    print(f"\n  Common questions: n = {len(common_q)}")

    # Citation precision comparison
    p_main = prec_main.loc[common_q].values
    p_alt  = prec_alt.loc[common_q].values
    stat_p, pval_p = stats.wilcoxon(p_main, p_alt)
    print(f"\n  Citation Precision  (Wilcoxon signed-rank, two-sided)")
    print(f"    Main system mean:  {p_main.mean():.4f}")
    print(f"    Alt system mean:   {p_alt.mean():.4f}")
    print(f"    W = {stat_p:.1f},  p = {pval_p:.4f}")

    # Citation support rate comparison
    s_main = supp_main.loc[common_q].values
    s_alt  = supp_alt.loc[common_q].values
    if (s_main == s_alt).all():
        print(f"\n  Citation Support Rate: identical for all questions — Wilcoxon not applicable")
    else:
        stat_s, pval_s = stats.wilcoxon(s_main, s_alt)
        print(f"\n  Citation Support Rate  (Wilcoxon signed-rank, two-sided)")
        print(f"    Main system rate:  {s_main.mean():.4f}")
        print(f"    Alt system rate:   {s_alt.mean():.4f}")
        print(f"    W = {stat_s:.1f},  p = {pval_s:.4f}")


# ─────────────────────────── main ───────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("  READER STUDY ANALYSIS  —  Section 7.2")
    print("=" * 70)
    print(f"\n  responses:        {args.responses}")
    print(f"  source_ratings:   {args.source_ratings}")
    print(f"  bootstrap n:      {args.bootstrap_n:,}")
    print(f"  random seed:      {args.seed}")

    resp, src, q_cite = load_data(args)

    print(f"\n  Physicians:                 {resp['pid'].nunique()}")
    print(f"  Physician-question pairs:   {len(resp)}")
    print(f"  Unique questions:           {resp['question_id'].nunique()}")
    print(f"  Question-citation ratings:  {len(src)}")

    # Primary
    primary_analysis(resp, args.bootstrap_n, args.seed)

    # Descriptive
    descriptive_change_rates(resp, args.bootstrap_n, args.seed)
    descriptive_citation_metrics(src, args.bootstrap_n, args.seed)
    inter_rater_agreement(resp, src)

    # Secondary hypotheses
    hypothesis_a(resp)
    hypothesis_b(resp)
    hypothesis_c(resp)

    # Exploratory
    mean_k = llm_judge_validation(src, args.llm_judge_csv)
    agentic_vs_simple_rag(src, args.alt_responses_csv, args.alt_source_ratings_csv,
                          mean_k, args.bootstrap_n, args.seed)

    print(f"\n{SEP}")
    print("  Analysis complete.")
    print(SEP)


if __name__ == "__main__":
    main()