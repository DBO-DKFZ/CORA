"""
Error pattern analysis using LLM-derived skin tone + gender labels
(results/skin_tone_classifications.csv) for 4 models × baseline + RAG.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

RESULTS = Path("results")
PLOTS   = Path("plots")
PLOTS.mkdir(exist_ok=True)

MODEL_PAIRS = {
    "GPT-5":     (RESULTS / "results_gpt5.csv",          RESULTS / "results_gpt5_rag.csv"),
    "GPT-5mini": (RESULTS / "results_gpt5mini.csv",      RESULTS / "results_gpt5mini_rag.csv"),
    "DeepSeek":  (RESULTS / "results_deepseekv3.1.csv",  RESULTS / "results_deepseekv3.1_rag.csv"),
    "MiniMax":   (RESULTS / "results_minimax2.7.csv",    RESULTS / "results_minimax2.7_reranked_rag.csv"),
}
COLORS = {
    "GPT-5": "#4C72B0", "GPT-5mini": "#DD8452",
    "DeepSeek": "#55A868", "MiniMax": "#C44E52",
}
SKIN_COLORS = {
    "dark": "#8B4513", "light": "#FFD700", "medium": "#CD853F", "unspecified": "#AAAAAA",
}

# ── load classifications ───────────────────────────────────────────────────────
clf = pd.read_csv(RESULTS / "skin_tone_classifications.csv")
clf["skin_tone_label"]   = clf["skin_tone_label"].str.lower().str.strip()
clf["patient_sex"]       = clf["patient_sex"].str.lower().str.strip()
shared_ids = set(clf["question_id"])

# ── helpers ───────────────────────────────────────────────────────────────────
def ci95(s: pd.Series):
    n, p = len(s), s.mean()
    if n == 0: return p, p, p
    z = 1.96
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return p, max(0, c-h), min(1, c+h)

def fisher(a: pd.Series, b: pd.Series):
    if len(a) < 3 or len(b) < 3: return None
    tbl = np.array([[a.sum(), len(a)-a.sum()], [b.sum(), len(b)-b.sum()]])
    _, p = stats.fisher_exact(tbl)
    return p

def sig(p):
    if p is None: return "—"
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))

def load_model(path: Path, variant: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["is_correct"] = (df["llm_response"].str.strip() ==
                        df["correct_choice"].str.strip()).astype(int)
    if variant == "Baseline":
        df = df[df["question_id"].isin(shared_ids)]
    df = df.merge(clf[["question_id","skin_tone_label","skin_tone_relevant",
                        "skin_tone_mentioned","patient_sex","patient_sex_relevant"]],
                  on="question_id", how="left")
    return df

def bar_with_ci(ax, groups, vals, errs, colors, title, ylabel="Accuracy",
                xlabel="", hline=True):
    x = np.arange(len(groups))
    ax.bar(x, vals, color=colors, alpha=0.85,
           yerr=errs, capsize=4, error_kw={"elinewidth":1})
    for xi, (v, g) in enumerate(zip(vals, groups)):
        if not np.isnan(v):
            ax.text(xi, v + 0.025, f"{v:.0%}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    if hline: ax.axhline(0.25, color="grey", lw=0.7, ls="--")
    ax.spines[["top","right"]].set_visible(False)

def grouped_bars(ax, data_dict, groups, title, ylabel="Accuracy", min_n=5):
    """data_dict: {series_label: {group: (acc, lo, hi, n)}}"""
    models = list(data_dict)
    x = np.arange(len(groups))
    w = 0.18
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * w
    for i, (m, gdata) in enumerate(data_dict.items()):
        vals  = [gdata[g][0] if g in gdata and gdata[g][3]>=min_n else np.nan for g in groups]
        errs  = [
            [max(0,v-gdata[g][1]) if (g in gdata and not np.isnan(v)) else 0
             for g,v in zip(groups,vals)],
            [max(0,gdata[g][2]-v) if (g in gdata and not np.isnan(v)) else 0
             for g,v in zip(groups,vals)],
        ]
        clean = [v if not np.isnan(v) else 0 for v in vals]
        ax.bar(x+offsets[i], clean, w*0.9, color=COLORS[m], label=m,
               yerr=errs, capsize=3, error_kw={"elinewidth":0.8}, alpha=0.85)
        for xi, (val, g) in enumerate(zip(vals, groups)):
            if not np.isnan(val) and g in gdata and gdata[g][3] >= min_n:
                n = gdata[g][3]
                ax.text(xi+offsets[i], val+0.015, f"n={n}",
                        ha="center", fontsize=6, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylim(0, 1.18); ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.axhline(0.25, color="grey", lw=0.7, ls="--")
    ax.legend(fontsize=7, ncol=2); ax.spines[["top","right"]].set_visible(False)

def build_group_stats(df, col):
    out = {}
    for grp, g in df.groupby(col):
        acc, lo, hi = ci95(g["is_correct"])
        out[grp] = (acc, lo, hi, len(g))
    return out

# ── assemble all dataframes ────────────────────────────────────────────────────
all_data = {}
for model, (base_path, rag_path) in MODEL_PAIRS.items():
    all_data[(model, "Baseline")] = load_model(base_path, "Baseline")
    all_data[(model, "RAG")]      = load_model(rag_path,  "RAG")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT: Overall accuracy summary
# ═══════════════════════════════════════════════════════════════════════════════
print("="*72)
print("OVERALL ACCURACY (on shared 2736-question subset)")
print("="*72)
for (model, variant), df in all_data.items():
    acc, lo, hi = ci95(df["is_correct"])
    print(f"  {model:12s} {variant:9s}  {acc:.1%}  [{lo:.1%}–{hi:.1%}]  n={len(df)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT: Skin tone label accuracy + Fisher vs unspecified
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*72)
print("ACCURACY BY SKIN TONE LABEL (LLM-derived)")
print("="*72)
SKIN_LABELS = ["dark","light","medium","unspecified"]

for (model, variant), df in all_data.items():
    print(f"\n  {model} — {variant}")
    unsp = df[df["skin_tone_label"] == "unspecified"]["is_correct"]
    for lbl in SKIN_LABELS:
        g = df[df["skin_tone_label"] == lbl]["is_correct"]
        if len(g) == 0: continue
        acc, lo, hi = ci95(g)
        p = fisher(g, unsp) if lbl != "unspecified" else None
        gap = (acc - unsp.mean()) if lbl != "unspecified" else 0
        print(f"    {lbl:12s}  n={len(g):4d}  {acc:.1%}  [{lo:.1%}–{hi:.1%}]"
              f"  gap={gap:+.1%}  p={p:.4f} {sig(p)}" if p is not None
              else f"    {lbl:12s}  n={len(g):4d}  {acc:.1%}  [{lo:.1%}–{hi:.1%}]  (reference)")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT: Skin-tone-relevant questions
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*72)
print("ACCURACY: SKIN-TONE-RELEVANT vs NOT RELEVANT QUESTIONS")
print("="*72)
for (model, variant), df in all_data.items():
    rel  = df[df["skin_tone_relevant"] == True]["is_correct"]
    nrel = df[df["skin_tone_relevant"] == False]["is_correct"]
    if len(rel) == 0: continue
    a_r,  lo_r,  hi_r  = ci95(rel)
    a_nr, lo_nr, hi_nr = ci95(nrel)
    p = fisher(rel, nrel)
    print(f"  {model:12s} {variant:9s}  "
          f"relevant={a_r:.1%}(n={len(rel)})  "
          f"not-relevant={a_nr:.1%}(n={len(nrel)})  "
          f"gap={a_r-a_nr:+.1%}  p={p:.4f} {sig(p)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT: Gender accuracy + Fisher
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*72)
print("ACCURACY BY PATIENT SEX (LLM-derived)")
print("="*72)
SEX_LABELS = ["male","female","unspecified","mixed"]

for (model, variant), df in all_data.items():
    print(f"\n  {model} — {variant}")
    unsp_sex = df[df["patient_sex"] == "unspecified"]["is_correct"]
    for lbl in SEX_LABELS:
        g = df[df["patient_sex"] == lbl]["is_correct"]
        if len(g) < 3: continue
        acc, lo, hi = ci95(g)
        p = fisher(g, unsp_sex) if lbl != "unspecified" else None
        gap = (acc - unsp_sex.mean()) if lbl != "unspecified" else 0
        print(f"    {lbl:12s}  n={len(g):4d}  {acc:.1%}  [{lo:.1%}–{hi:.1%}]"
              f"  gap={gap:+.1%}  p={p:.4f} {sig(p)}" if p is not None
              else f"    {lbl:12s}  n={len(g):4d}  {acc:.1%}  [{lo:.1%}–{hi:.1%}]  (reference)")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT: Sex-relevant questions
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*72)
print("ACCURACY: SEX-RELEVANT vs NOT RELEVANT QUESTIONS")
print("="*72)
for (model, variant), df in all_data.items():
    rel  = df[df["patient_sex_relevant"] == True]["is_correct"]
    nrel = df[df["patient_sex_relevant"] == False]["is_correct"]
    a_r,  lo_r,  hi_r  = ci95(rel)
    a_nr, lo_nr, hi_nr = ci95(nrel)
    p = fisher(rel, nrel)
    print(f"  {model:12s} {variant:9s}  "
          f"sex-relevant={a_r:.1%}(n={len(rel)})  "
          f"not-relevant={a_nr:.1%}(n={len(nrel)})  "
          f"gap={a_r-a_nr:+.1%}  p={p:.4f} {sig(p)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT: Sex × Skin tone interaction
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*72)
print("SEX × SKIN TONE INTERACTION (Baseline, n≥10)")
print("="*72)
for model in COLORS:
    df = all_data[(model, "Baseline")]
    print(f"\n  {model}")
    ct = (df.groupby(["patient_sex","skin_tone_label"])["is_correct"]
            .agg(acc="mean", n="count")
            .reset_index()
            .query("n >= 10"))
    if ct.empty:
        print("    (no groups with n≥10)")
        continue
    ct["acc_str"] = ct["acc"].map("{:.1%}".format)
    print(ct[["patient_sex","skin_tone_label","n","acc_str"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Skin tone label: all models × both variants
# ═══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
fig1.suptitle("Accuracy by LLM-Derived Skin Tone Label\n"
              "Top: Baseline  |  Bottom: RAG  (error bars = 95% CI Wilson, n shown above bars)",
              fontsize=12, fontweight="bold")

for col, model in enumerate(COLORS):
    for row, variant in enumerate(["Baseline", "RAG"]):
        ax = axes[row][col]
        df = all_data[(model, variant)]
        vals, errs_lo, errs_hi, ns, colors = [], [], [], [], []
        for lbl in SKIN_LABELS:
            g = df[df["skin_tone_label"] == lbl]["is_correct"]
            if len(g) >= 3:
                acc, lo, hi = ci95(g)
                vals.append(acc); errs_lo.append(acc-lo); errs_hi.append(hi-acc)
                ns.append(len(g)); colors.append(SKIN_COLORS.get(lbl, "#999"))
            else:
                vals.append(np.nan); errs_lo.append(0); errs_hi.append(0)
                ns.append(0); colors.append("#999")
        clean = [v if not np.isnan(v) else 0 for v in vals]
        ax.bar(range(len(SKIN_LABELS)), clean, color=colors, alpha=0.85,
               yerr=[errs_lo, errs_hi], capsize=4, error_kw={"elinewidth":1})
        for xi, (v, n) in enumerate(zip(vals, ns)):
            if not np.isnan(v) and n >= 3:
                ax.text(xi, v+0.02, f"{v:.0%}\nn={n}", ha="center", fontsize=7, va="bottom")
        ax.set_xticks(range(len(SKIN_LABELS)))
        ax.set_xticklabels(SKIN_LABELS, fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.axhline(0.25, color="grey", lw=0.7, ls="--")
        ax.set_title(f"{model}\n({variant})", fontsize=9, fontweight="bold",
                     color=COLORS[model])
        ax.spines[["top","right"]].set_visible(False)
        if col == 0: ax.set_ylabel("Accuracy", fontsize=8)

plt.tight_layout()
fig1.savefig(PLOTS / "llm_skin_tone_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"\nSaved: {PLOTS / 'llm_skin_tone_accuracy.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Patient sex: all models × both variants
# ═══════════════════════════════════════════════════════════════════════════════
SEX_PLOT_LABELS = ["male", "female", "unspecified"]
SEX_COLORS_MAP  = {"male": "#5B9BD5", "female": "#ED7D31", "unspecified": "#A5A5A5"}

fig2, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
fig2.suptitle("Accuracy by LLM-Derived Patient Sex\n"
              "Top: Baseline  |  Bottom: RAG",
              fontsize=12, fontweight="bold")

for col, model in enumerate(COLORS):
    for row, variant in enumerate(["Baseline", "RAG"]):
        ax = axes[row][col]
        df = all_data[(model, variant)]
        vals, errs_lo, errs_hi, ns, colors = [], [], [], [], []
        for lbl in SEX_PLOT_LABELS:
            g = df[df["patient_sex"] == lbl]["is_correct"]
            if len(g) >= 5:
                acc, lo, hi = ci95(g)
                vals.append(acc); errs_lo.append(acc-lo); errs_hi.append(hi-acc)
                ns.append(len(g)); colors.append(SEX_COLORS_MAP[lbl])
            else:
                vals.append(np.nan); errs_lo.append(0); errs_hi.append(0)
                ns.append(0); colors.append("#999")
        clean = [v if not np.isnan(v) else 0 for v in vals]
        ax.bar(range(len(SEX_PLOT_LABELS)), clean, color=colors, alpha=0.85,
               yerr=[errs_lo, errs_hi], capsize=4, error_kw={"elinewidth":1})
        for xi, (v, n) in enumerate(zip(vals, ns)):
            if not np.isnan(v) and n >= 5:
                ax.text(xi, v+0.02, f"{v:.0%}\nn={n}", ha="center", fontsize=7, va="bottom")
        ax.set_xticks(range(len(SEX_PLOT_LABELS)))
        ax.set_xticklabels(SEX_PLOT_LABELS, fontsize=9)
        ax.set_ylim(0, 1.2)
        ax.axhline(0.25, color="grey", lw=0.7, ls="--")
        ax.set_title(f"{model}\n({variant})", fontsize=9, fontweight="bold",
                     color=COLORS[model])
        ax.spines[["top","right"]].set_visible(False)
        if col == 0: ax.set_ylabel("Accuracy", fontsize=8)

plt.tight_layout()
fig2.savefig(PLOTS / "llm_sex_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {PLOTS / 'llm_sex_accuracy.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – RAG lift: dark skin vs all, male vs female vs unspecified
# ═══════════════════════════════════════════════════════════════════════════════
fig3, (ax_skin, ax_sex) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("RAG Accuracy Lift (RAG − Baseline) by Demographic Group",
              fontsize=12, fontweight="bold")

# skin lift
skin_lift_groups = SKIN_LABELS
x = np.arange(len(skin_lift_groups))
w = 0.18
offsets = np.linspace(-(len(COLORS)-1)/2, (len(COLORS)-1)/2, len(COLORS)) * w
for i, (model, color) in enumerate(COLORS.items()):
    lifts, ns = [], []
    for lbl in skin_lift_groups:
        b = all_data[(model,"Baseline")][all_data[(model,"Baseline")]["skin_tone_label"]==lbl]["is_correct"]
        r = all_data[(model,"RAG")][all_data[(model,"RAG")]["skin_tone_label"]==lbl]["is_correct"]
        if len(b) >= 5 and len(r) >= 5:
            lifts.append(r.mean() - b.mean()); ns.append(len(r))
        else:
            lifts.append(np.nan); ns.append(0)
    clean = [v if not np.isnan(v) else 0 for v in lifts]
    bars = ax_skin.bar(x+offsets[i], clean, w*0.9, color=color, label=model, alpha=0.85)
    for xi, (lift, n) in enumerate(zip(lifts, ns)):
        if not np.isnan(lift) and n >= 5:
            ax_skin.text(xi+offsets[i], lift+(0.005 if lift>=0 else -0.018),
                         f"{lift:+.1%}", ha="center", fontsize=6,
                         va="bottom" if lift>=0 else "top")
ax_skin.axhline(0, color="black", lw=0.8)
ax_skin.set_xticks(x); ax_skin.set_xticklabels(skin_lift_groups, fontsize=9)
ax_skin.set_ylabel("Accuracy lift (RAG − Baseline)", fontsize=9)
ax_skin.set_title("Skin Tone", fontsize=10, fontweight="bold")
ax_skin.legend(fontsize=8); ax_skin.spines[["top","right"]].set_visible(False)

# sex lift
sex_lift_groups = ["male","female","unspecified"]
x2 = np.arange(len(sex_lift_groups))
for i, (model, color) in enumerate(COLORS.items()):
    lifts = []
    for lbl in sex_lift_groups:
        b = all_data[(model,"Baseline")][all_data[(model,"Baseline")]["patient_sex"]==lbl]["is_correct"]
        r = all_data[(model,"RAG")][all_data[(model,"RAG")]["patient_sex"]==lbl]["is_correct"]
        if len(b) >= 10 and len(r) >= 10:
            lifts.append(r.mean() - b.mean())
        else:
            lifts.append(np.nan)
    clean = [v if not np.isnan(v) else 0 for v in lifts]
    ax_sex.bar(x2+offsets[i], clean, w*0.9, color=color, label=model, alpha=0.85)
    for xi, lift in enumerate(lifts):
        if not np.isnan(lift):
            ax_sex.text(xi+offsets[i], lift+(0.005 if lift>=0 else -0.018),
                        f"{lift:+.1%}", ha="center", fontsize=6,
                        va="bottom" if lift>=0 else "top")
ax_sex.axhline(0, color="black", lw=0.8)
ax_sex.set_xticks(x2); ax_sex.set_xticklabels(sex_lift_groups, fontsize=9)
ax_sex.set_ylabel("Accuracy lift (RAG − Baseline)", fontsize=9)
ax_sex.set_title("Patient Sex", fontsize=10, fontweight="bold")
ax_sex.legend(fontsize=8); ax_sex.spines[["top","right"]].set_visible(False)

plt.tight_layout()
fig3.savefig(PLOTS / "llm_rag_lift_demographics.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {PLOTS / 'llm_rag_lift_demographics.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Heatmap: model × sex × skin tone (Baseline only, n≥10)
# ═══════════════════════════════════════════════════════════════════════════════
# Build a combined label for rows
rows_heat = []
for model in COLORS:
    df = all_data[(model, "Baseline")]
    for sex in ["male","female","unspecified"]:
        for skin in SKIN_LABELS:
            sub = df[(df["patient_sex"]==sex) & (df["skin_tone_label"]==skin)]["is_correct"]
            if len(sub) >= 10:
                rows_heat.append({"model": model, "sex": sex, "skin": skin,
                                  "acc": sub.mean(), "n": len(sub)})
heat_df = pd.DataFrame(rows_heat)

if not heat_df.empty:
    heat_df["row_label"] = heat_df["model"] + " / " + heat_df["sex"]
    heat_df["col_label"] = heat_df["skin"]
    pivot_h = heat_df.pivot_table(index="row_label", columns="col_label", values="acc")

    fig4, ax4 = plt.subplots(figsize=(max(6,len(pivot_h.columns)*2),
                                      max(4,len(pivot_h)*0.5)))
    im = ax4.imshow(pivot_h.values, aspect="auto", cmap="RdYlGn", vmin=0.6, vmax=1.0)
    ax4.set_xticks(range(len(pivot_h.columns)))
    ax4.set_xticklabels(pivot_h.columns, fontsize=9)
    ax4.set_yticks(range(len(pivot_h.index)))
    ax4.set_yticklabels(pivot_h.index, fontsize=8)
    for i in range(pivot_h.shape[0]):
        for j in range(pivot_h.shape[1]):
            v = pivot_h.values[i,j]
            if not np.isnan(v):
                n = heat_df[(heat_df["row_label"]==pivot_h.index[i]) &
                            (heat_df["col_label"]==pivot_h.columns[j])]["n"].values
                n_str = f"n={n[0]}" if len(n) else ""
                ax4.text(j, i, f"{v:.0%}\n{n_str}", ha="center", va="center",
                         fontsize=7, color="black" if 0.65<v<0.9 else "white")
    plt.colorbar(im, ax=ax4, label="Accuracy", shrink=0.5)
    ax4.set_title("Accuracy: Model × Sex × Skin Tone (Baseline, n≥10)",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig4.savefig(PLOTS / "llm_sex_skin_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved: {PLOTS / 'llm_sex_skin_heatmap.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT: Chi-square tests — sex and skin tone
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*72)
print("CHI-SQUARE TESTS")
print("="*72)
for dim, col in [("Skin Tone", "skin_tone_label"), ("Patient Sex", "patient_sex")]:
    print(f"\n  {dim}")
    for (model, variant), df in all_data.items():
        sub = df.dropna(subset=[col])
        ct = pd.crosstab(sub[col], sub["is_correct"])
        if ct.shape[0] >= 2 and ct.shape[1] == 2:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            print(f"    {model:12s} {variant:9s}  χ²={chi2:6.2f}  df={dof}  "
                  f"p={p:.4f}  {sig(p)}")

print("\nAnalysis complete.")
