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


# ─────────────────────────── RAIR / RSR (appropriate reliance) ──────────────

def reliance_metrics(ini: np.ndarray, fin: np.ndarray, llm: np.ndarray,
                     adopt: np.ndarray) -> dict:
    """RAIR/RSR and related reliance metrics (Schemmer et al. 2023, CHI) on a
    (sub)sample. The two conflict quadrants are where the physician's unaided
    answer and the LLM disagree — RAIR is measured on qA, RSR on qB."""
    a = (~ini) & llm            # RAIR denominator: unaided wrong, LLM right
    b =  ini  & (~llm)          # RSR  denominator: unaided right, LLM wrong
    return {
        "acc_unaided":  ini.mean(),
        "acc_assisted": fin.mean(),
        "acc_llm":      llm.mean(),
        "rair":         adopt[a].mean() if a.sum() else np.nan,   # adopt correct advice
        "rsr":          fin[b].mean()   if b.sum() else np.nan,   # keep correct answer
        "adopt_wrong":  adopt[b].mean() if b.sum() else np.nan,   # adopt incorrect advice
    }


def _cluster_bootstrap(pid: np.ndarray, n: int, seed: int):
    """Yield n physician-clustered resamples of row indices."""
    pids = np.unique(pid)
    idx_by_pid = {p: np.where(pid == p)[0] for p in pids}
    rng = np.random.default_rng(seed)
    for _ in range(n):
        draw = rng.choice(pids, size=len(pids), replace=True)
        yield np.concatenate([idx_by_pid[p] for p in draw])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score 95 % CI for a binomial proportion; returns (p, lo, hi)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def newcombe_diff(k1: int, n1: int, k2: int, n2: int,
                  z: float = 1.96) -> tuple[float, float, float]:
    """Newcombe (1998) method 10 hybrid-score 95 % CI for p1 - p2 (independent props)."""
    if n1 == 0 or n2 == 0:
        return (np.nan, np.nan, np.nan)
    p1, l1, u1 = wilson(k1, n1, z)
    p2, l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return (d,
            d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def gwet_ac1_binary(subject_ratings) -> tuple[float, int]:
    """Gwet's (2008) AC1 for a binary label with multiple raters per subject and
    an unequal rater count across subjects. Returns (AC1, n_subjects_used)."""
    pas, pis, used = [], [], 0
    for r in subject_ratings:
        r = np.asarray(r, float)
        r = r[~np.isnan(r)]
        ri = r.size
        if ri < 2:                       # need >=2 raters to observe agreement
            continue
        used += 1
        n1 = r.sum()
        n0 = ri - n1
        pas.append((n1 * (n1 - 1) + n0 * (n0 - 1)) / (ri * (ri - 1)))
        pis.append(n1 / ri)
    if used == 0:
        return np.nan, 0
    pa, pi1 = np.mean(pas), np.mean(pis)
    pe = 2 * pi1 * (1 - pi1)             # K=2 categories
    return ((pa - pe) / (1 - pe) if pe < 1 else np.nan), used


def rair_rsr_overall(resp: pd.DataFrame, bootstrap_n: int, seed: int):
    """RAIR / RSR (Schemmer et al. 2023, CHI): appropriate reliance on the
    physician-LLM conflict quadrants. Returns the answered-only frame plus the
    reliance arrays, for reuse by the citation-support stratification below."""
    section("RAIR / RSR: Appropriate Reliance (Schemmer et al. 2023, CHI)")

    d = resp[resp["answered"]].copy()
    ini   = d["initial_correct"].to_numpy(bool)
    fin   = d["final_correct"].to_numpy(bool)
    llm   = d["llm_is_correct"].to_numpy(bool)
    adopt = d["final_eq_llm"].to_numpy(bool)
    pid   = d["pid"].to_numpy()

    qA = (~ini) &  llm   # unaided wrong, LLM right   -> should adopt      (RAIR)
    qB =  ini  & (~llm)  # unaided right, LLM wrong   -> should self-rely  (RSR)

    print(f"\n  N = {len(d)} answered questions, {d['pid'].nunique()} physicians")
    print(f"  RAIR quadrant (unaided wrong & LLM right): {int(qA.sum())}")
    print(f"  RSR  quadrant (unaided right & LLM wrong): {int(qB.sum())}")

    pt = reliance_metrics(ini, fin, llm, adopt)
    keys = ["acc_unaided", "acc_assisted", "acc_llm", "rair", "rsr", "adopt_wrong"]
    boot = {k: [] for k in keys}
    for ridx in _cluster_bootstrap(pid, bootstrap_n, seed):
        m = reliance_metrics(ini[ridx], fin[ridx], llm[ridx], adopt[ridx])
        for k in keys:
            boot[k].append(m[k])
    boot = {k: np.array(v, float) for k, v in boot.items()}

    def _ci(k):
        a = boot[k][~np.isnan(boot[k])]
        return np.percentile(a, 2.5), np.percentile(a, 97.5)

    def _show(label, k):
        lo, hi = _ci(k)
        print(f"    {label:<28} {pt[k]*100:5.1f}%   95 % CI [{lo*100:4.1f}, {hi*100:4.1f}]")

    print("\n  Accuracy")
    _show("Physician unaided", "acc_unaided")
    _show("Physician + RAG",   "acc_assisted")
    _show("LLM (RAG) alone",   "acc_llm")
    print("\n  Appropriate reliance")
    _show("RAIR  (adopt good advice)", "rair")
    _show("RSR   (resist bad advice)", "rsr")

    return d, ini, fin, llm, adopt, qA, qB


def rair_rsr_by_citation_support(d: pd.DataFrame, ini: np.ndarray, fin: np.ndarray,
                                 llm: np.ndarray, adopt: np.ndarray, qA: np.ndarray,
                                 qB: np.ndarray, bootstrap_n: int, seed: int):
    """EXPLORATORY (not preregistered): RAIR/RSR stratified by whether the
    physician's own source ratings show >=1 citation supporting CORA's advice.
    RAIR is recomputed on qA, RSR on qB, within each stratum; pooling the two
    strata must exactly reproduce the overall values from rair_rsr_overall()."""
    section("EXPLORATORY: RAIR / RSR by CORA Citation-Support Level")
    print("\n  Not part of the preregistered confirmatory hypothesis family (A1/B1/C1).")
    print("  No confirmatory p-values; 95 % CIs only, hypothesis-generating.")

    MIN_DENOM = 10
    precision = (d["n_support"] / d["n_support_rated"]).to_numpy(float)

    hi = d["any_source_supports"].to_numpy(bool)   # >=1 cited source rated Supports
    lo = ~hi                                        # none of the cited sources support
    STRATIFIER_LABEL = "any cited source supports (n_support >= 1)"

    print(f"\n  Stratifier: {STRATIFIER_LABEL} -- ANY vs NO citation support")
    print(f"    any-support rows: {int(hi.sum()):4d}   no-support rows: {int(lo.sum()):4d}")
    print(f"    RAIR subset qA -> any {int((qA & hi).sum())} / none {int((qA & lo).sum())}")
    print(f"    RSR  subset qB -> any {int((qB & hi).sum())} / none {int((qB & lo).sum())}"
          f"   (CORA-wrong subset is small)")

    strata = {"high": hi, "low": lo}
    rows = []
    for s, mask in strata.items():
        cnt = {
            "rair": (int(adopt[qA & mask].sum()), int((qA & mask).sum())),
            "rsr":  (int(fin[qB & mask].sum()),   int((qB & mask).sum())),
        }
        for metric, (num, den) in cnt.items():
            est, clo, chi = wilson(num, den)
            rows.append({"metric": metric.upper(), "stratum": s,
                        "numerator": num, "denominator": den,
                        "estimate": est, "wilson_lo": clo, "wilson_hi": chi,
                        "too_small": den < MIN_DENOM})
    strat_tbl = pd.DataFrame(rows)

    # Reconciliation: pooled strata must equal the overall RAIR / RSR exactly.
    for metric, onum, oden in [("RAIR", int(adopt[qA].sum()), int(qA.sum())),
                               ("RSR",  int(fin[qB].sum()),   int(qB.sum()))]:
        sub = strat_tbl[strat_tbl.metric == metric]
        pnum, pden = int(sub.numerator.sum()), int(sub.denominator.sum())
        assert pnum == onum and pden == oden, \
            f"{metric} strata don't pool to the overall value"

    # Reader-clustered bootstrap for each stratum + the high-minus-low differences.
    bkeys = ["rair_high", "rair_low", "rsr_high", "rsr_low", "rair_diff", "rsr_diff"]
    sboot = {k: [] for k in bkeys}
    for ridx in _cluster_bootstrap(d["pid"].to_numpy(), bootstrap_n, seed):
        m_hi, m_lo = hi[ridx], lo[ridx]
        qA_r, qB_r = qA[ridx], qB[ridx]
        ad, fn = adopt[ridx], fin[ridx]
        e = {}
        for name, sub in [("high", m_hi), ("low", m_lo)]:
            da, db = qA_r & sub, qB_r & sub
            e[f"rair_{name}"] = ad[da].mean() if da.sum() else np.nan
            e[f"rsr_{name}"]  = fn[db].mean() if db.sum() else np.nan
        e["rair_diff"] = e["rair_high"] - e["rair_low"]
        e["rsr_diff"]  = e["rsr_high"]  - e["rsr_low"]
        for k in bkeys:
            sboot[k].append(e[k])
    sboot = {k: np.array(v, float) for k, v in sboot.items()}

    def _boot_ci(k):
        a = sboot[k][~np.isnan(sboot[k])]
        if a.size == 0:
            return (np.nan, np.nan, np.nan)
        return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)

    bmap = {("RAIR", "high"): "rair_high", ("RAIR", "low"): "rair_low",
           ("RSR", "high"): "rsr_high",   ("RSR", "low"): "rsr_low"}
    strat_tbl["boot_lo"] = np.nan
    strat_tbl["boot_hi"] = np.nan
    for i, r in strat_tbl.iterrows():
        _, blo, bhi = _boot_ci(bmap[(r["metric"], r["stratum"])])
        strat_tbl.at[i, "boot_lo"] = blo
        strat_tbl.at[i, "boot_hi"] = bhi

    diff_rows = []
    for metric in ("RAIR", "RSR"):
        sub = strat_tbl[strat_tbl.metric == metric].set_index("stratum")
        kh, nh = int(sub.loc["high", "numerator"]), int(sub.loc["high", "denominator"])
        kl, nl = int(sub.loc["low",  "numerator"]), int(sub.loc["low",  "denominator"])
        dv, nlo, nhi = newcombe_diff(kh, nh, kl, nl)
        _, blo, bhi = _boot_ci("rair_diff" if metric == "RAIR" else "rsr_diff")
        diff_rows.append({"metric": metric, "diff_high_minus_low": dv,
                          "newcombe_lo": nlo, "newcombe_hi": nhi,
                          "boot_lo": blo, "boot_hi": bhi,
                          "high_n": nh, "low_n": nl})
    diff_tbl = pd.DataFrame(diff_rows).set_index("metric")

    # Threshold sensitivity: is the RSR effect's direction stable across cuts?
    SENS = [("any (>=1 source)", 1e-9), ("majority (>=50%)", 0.5), ("unanimous (all)", 1.0)]
    sens_rows = []
    for label, thr in SENS:
        mh = precision >= thr
        ml = ~mh
        rh = adopt[qA & mh].mean() if (qA & mh).sum() else np.nan
        rl = adopt[qA & ml].mean() if (qA & ml).sum() else np.nan
        sh = fin[qB & mh].mean()   if (qB & mh).sum() else np.nan
        sl = fin[qB & ml].mean()   if (qB & ml).sum() else np.nan
        sens_rows.append({"threshold": label, "rsr_diff": sh - sl, "rair_diff": rh - rl})
    sens_tbl = pd.DataFrame(sens_rows)
    signs = np.sign(sens_tbl["rsr_diff"].dropna())
    stable = len(signs) > 0 and signs.nunique() == 1

    # Gwet's AC1: inter-rater reliability of the stratifier itself.
    q_ratings = (d.groupby("question_id")["any_source_supports"]
                  .apply(lambda s: s.astype(float).to_numpy()))
    ac1, n_used = gwet_ac1_binary(list(q_ratings))
    print(f"\n  Support IRR: Gwet's AC1 = {ac1:.3f} on 'any_source_supports' "
          f"({n_used} of {q_ratings.size} questions, >=2 raters)")

    def _p(x):
        return "  n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x*100:5.1f}%"

    stratum_name = {"high": "any ", "low": "none"}
    for metric in ("RAIR", "RSR"):
        print(f"\n  {metric}:")
        for _, r in strat_tbl[strat_tbl.metric == metric].iterrows():
            flag = "   <-- n<10: TOO SMALL for stable estimation" if r.too_small else ""
            print(f"    {stratum_name[r.stratum]} {int(r.numerator):3d}/{int(r.denominator):3d}"
                  f" = {_p(r.estimate)}   Wilson [{_p(r.wilson_lo)},{_p(r.wilson_hi)}]"
                  f"   reader-boot [{_p(r.boot_lo)},{_p(r.boot_hi)}]{flag}")
        dr = diff_tbl.loc[metric]
        print(f"    any-none: {dr.diff_high_minus_low*100:+5.1f} pts"
              f"   Newcombe [{dr.newcombe_lo*100:+.1f},{dr.newcombe_hi*100:+.1f}]"
              f"   reader-boot [{dr.boot_lo*100:+.1f},{dr.boot_hi*100:+.1f}]")

    print(f"\n  Threshold sensitivity (RSR direction across any/majority/unanimous "
          f"support cuts): {'STABLE' if stable else 'NOT stable'}")

    rsr_dir = diff_tbl.loc["RSR", "diff_high_minus_low"]
    print("\n  Interpretation (RSR is the target of this analysis):")
    if rsr_dir < 0:
        print("    RSR is LOWER when >=1 source supports -- consistent with well-cited WRONG")
        print("    advice depressing appropriate self-reliance. Hypothesis-generating only:")
        print("    the CORA-wrong (qB) subset is small; see the sensitivity check above.")
    else:
        print("    RSR is NOT lower under any citation support in this sample -- no RSR-")
        print("    collapse signal here.")


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
    d, ini, fin, llm, adopt, qA, qB = rair_rsr_overall(resp, args.bootstrap_n, args.seed)
    inter_rater_agreement(resp, src)

    # Secondary hypotheses
    hypothesis_a(resp)
    hypothesis_b(resp)
    hypothesis_c(resp)

    # Exploratory
    rair_rsr_by_citation_support(d, ini, fin, llm, adopt, qA, qB,
                                 args.bootstrap_n, args.seed)
    mean_k = llm_judge_validation(src, args.llm_judge_csv)
    agentic_vs_simple_rag(src, args.alt_responses_csv, args.alt_source_ratings_csv,
                          mean_k, args.bootstrap_n, args.seed)

    print(f"\n{SEP}")
    print("  Analysis complete.")
    print(SEP)


if __name__ == "__main__":
    main()