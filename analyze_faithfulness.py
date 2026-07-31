"""
Compare faithfulness judge results across multiple models.

Usage:
    python analyze_faithfulness.py results/faithfulness_*.csv
    python analyze_faithfulness.py results/faithfulness_llama4.csv results/faithfulness_gemma3.csv
    python analyze_faithfulness.py results/faithfulness_*.csv --out faithfulness_comparison.csv
"""

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

FAITHFUL_ORDER  = ["Yes", "No"]
SUPPORT_RATINGS = {"Supports", "Partially supports"}

console = Console(width=200)


def infer_label(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"^faithfulness_", "", stem)


def citation_precision(doc_ratings_json: str) -> float:
    try:
        ratings = json.loads(doc_ratings_json)
        if not ratings:
            return float("nan")
        return sum(1 for d in ratings if d.get("rating") in SUPPORT_RATINGS) / len(ratings)
    except Exception:
        return float("nan")


def load(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model"] = label
    df["is_correct"] = df["is_correct"].astype(bool)
    df["citation_precision"] = df["doc_ratings"].apply(citation_precision)
    df["faithfulness"] = pd.Categorical(df["faithfulness"], categories=FAITHFUL_ORDER, ordered=True)
    return df


def section(title: str) -> None:
    console.print(f"\n[bold cyan]{title}[/bold cyan]")


def fmt_val(v, float_fmt: str) -> str:
    if pd.isna(v):
        return "—"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:{float_fmt}}"
    return str(v)


def rich_table(df: pd.DataFrame, title: str, float_fmt: str = ".2f",
               index: bool = True) -> Table:
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False,
                  header_style="bold white", title_style="bold yellow",
                  show_edge=True, pad_edge=True)

    if index:
        table.add_column(df.index.name or "", style="bold", no_wrap=True)
    for col in df.columns:
        table.add_column(str(col), justify="right", no_wrap=True)

    for idx, row in df.iterrows():
        cells = ([str(idx)] if index else []) + [fmt_val(v, float_fmt) for v in row]
        table.add_row(*cells)

    return table


# ── per-model summary ─────────────────────────────────────────────────────────

def summary_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in frames:
        label = df["model"].iloc[0]
        n = len(df)
        acc = df["is_correct"].mean()
        faith_pct = (
            df["faithfulness"].value_counts(normalize=True)
            .reindex(FAITHFUL_ORDER, fill_value=0) * 100
        )
        cp = df["citation_precision"].mean()
        n_cited = df["n_cited"].mean()
        rows.append({
            "model":          label,
            "n":              n,
            "accuracy %":     acc * 100,
            "faithful yes %": faith_pct.get("Yes", 0),
            "faithful no %":  faith_pct.get("No", 0),
            "cite precision": cp,
            "mean n cited":   n_cited,
        })
    return pd.DataFrame(rows).set_index("model")


# ── faithfulness × correctness ────────────────────────────────────────────────

def print_faithfulness_x_correctness(frames: list[pd.DataFrame]) -> None:
    """
    One table per model. Rows = faithfulness. Columns = correct / wrong.
    Each cell shows  n  (% of that faithfulness group).
    Footer row shows totals.
    """
    for df in frames:
        label = df["model"].iloc[0]
        table = Table(
            title=f"Faithfulness × Correctness — {label}",
            box=box.SIMPLE_HEAVY, show_lines=False,
            header_style="bold white", title_style="bold yellow",
            show_edge=True, pad_edge=True,
        )
        table.add_column("faithfulness", style="bold", no_wrap=True)
        table.add_column("correct  (n / row%)", justify="right")
        table.add_column("wrong  (n / row%)",   justify="right")
        table.add_column("total", justify="right")
        table.add_column("% correct", justify="right")

        totals = {"correct": 0, "wrong": 0}
        for faith in FAITHFUL_ORDER:
            sub = df[df["faithfulness"] == faith]
            n_correct = int((sub["is_correct"]).sum())
            n_wrong   = int((~sub["is_correct"]).sum())
            n_total   = n_correct + n_wrong
            row_pct_c = n_correct / n_total * 100 if n_total else float("nan")
            row_pct_w = n_wrong   / n_total * 100 if n_total else float("nan")
            totals["correct"] += n_correct
            totals["wrong"]   += n_wrong
            table.add_row(
                faith,
                f"{n_correct}  ({row_pct_c:.1f}%)",
                f"{n_wrong}  ({row_pct_w:.1f}%)",
                str(n_total),
                f"[green]{row_pct_c:.1f}%[/green]" if not pd.isna(row_pct_c) else "—",
            )

        grand = totals["correct"] + totals["wrong"]
        overall_pct = totals["correct"] / grand * 100 if grand else float("nan")
        table.add_section()
        table.add_row(
            "[dim]total[/dim]",
            f"[dim]{totals['correct']}[/dim]",
            f"[dim]{totals['wrong']}[/dim]",
            f"[dim]{grand}[/dim]",
            f"[dim]{overall_pct:.1f}%[/dim]",
        )
        console.print(table)


# ── citation precision breakdown ──────────────────────────────────────────────

def citation_precision_breakdown(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in frames:
        label = df["model"].iloc[0]
        for faithful in FAITHFUL_ORDER:
            sub = df[df["faithfulness"] == faithful]
            for correct in [True, False]:
                cell = sub[sub["is_correct"] == correct]["citation_precision"]
                rows.append({
                    "model":        label,
                    "faithfulness": faithful,
                    "correct":      correct,
                    "n":            len(cell),
                    "mean_prec":    cell.mean(),
                })
    return pd.DataFrame(rows)


# ── doc rating distribution ───────────────────────────────────────────────────

def doc_rating_distribution(frames: list[pd.DataFrame]) -> pd.DataFrame:
    all_ratings = ["Supports", "Partially supports", "Irrelevant"]
    rows = []
    for df in frames:
        label = df["model"].iloc[0]
        counts: dict[str, int] = {r: 0 for r in all_ratings}
        total = 0
        for raw in df["doc_ratings"].dropna():
            try:
                ratings = json.loads(raw)
            except Exception:
                continue
            for d in ratings:
                r = d.get("rating", "")
                if r in counts:
                    counts[r] += 1
                    total += 1
        row: dict = {"model": label, "total judgements": int(total)}
        for r in all_ratings:
            row[r + " %"] = (counts[r] / total * 100) if total else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# ── rich table helpers for multi-index dataframes ─────────────────────────────

def print_citation_precision_table(cp_df: pd.DataFrame, n_df: pd.DataFrame) -> None:
    models = cp_df.columns.tolist()

    table = Table(
        title="Citation Precision  (mean % supporting docs)  by faithfulness × correctness",
        box=box.SIMPLE_HEAVY, show_lines=False,
        header_style="bold white", title_style="bold yellow",
        show_edge=True, pad_edge=True,
    )
    table.add_column("faithfulness", style="bold", no_wrap=True)
    table.add_column("correct", justify="center")
    for m in models:
        table.add_column(m, justify="right")

    for (faith, correct), row in cp_df.iterrows():
        n_row = n_df.loc[(faith, correct)] if (faith, correct) in n_df.index else {}
        prec_cells = []
        for m in models:
            v = row.get(m, float("nan"))
            n = int(n_row.get(m, 0)) if not n_df.empty else 0
            prec_cells.append(f"{v:.3f}  [dim](n={n})[/dim]" if not pd.isna(v) else "—")
        table.add_row(
            str(faith),
            "[green]✓[/green]" if correct else "[red]✗[/red]",
            *prec_cells,
        )

    console.print(table)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Faithfulness CSV files (globs accepted)")
    parser.add_argument("--out",  help="Save summary table to CSV")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.inputs:
        expanded = glob.glob(pattern)
        paths.extend(expanded if expanded else [pattern])
    paths = sorted(set(paths))

    if not paths:
        sys.exit("No input files found.")

    frames: list[pd.DataFrame] = []
    for p in paths:
        label = infer_label(p)
        try:
            df = load(p, label)
            frames.append(df)
            console.print(f"  Loaded [bold]{label}[/bold]  ({len(df)} rows)")
        except Exception as e:
            console.print(f"  [yellow][WARN][/yellow] Could not load {p}: {e}")

    if not frames:
        sys.exit("No files loaded.")

    # ── 1. Summary ────────────────────────────────────────────────────────────
    section("SUMMARY")
    summary = summary_table(frames)
    console.print(rich_table(summary, "Overview", float_fmt=".2f"))
    if args.out:
        summary.to_csv(args.out)
        console.print(f"  Summary saved to [bold]{args.out}[/bold]")

    # ── 2. Faithfulness × Correctness ─────────────────────────────────────────
    section("FAITHFULNESS × CORRECTNESS")
    print_faithfulness_x_correctness(frames)

    # ── 3. Citation precision breakdown ───────────────────────────────────────
    section("CITATION PRECISION")
    cp_df = citation_precision_breakdown(frames)
    pivot = cp_df.pivot_table(
        index=["faithfulness", "correct"], columns="model",
        values="mean_prec", aggfunc="mean",
    )
    n_pivot = cp_df.pivot_table(
        index=["faithfulness", "correct"], columns="model",
        values="n", aggfunc="sum",
    )
    pivot.index = pivot.index.set_levels(
        pd.CategoricalIndex(
            pivot.index.get_level_values(0).unique(),
            categories=FAITHFUL_ORDER, ordered=True,
        ),
        level=0,
    )
    pivot  = pivot.sort_index()
    n_pivot = n_pivot.reindex(pivot.index)
    print_citation_precision_table(pivot, n_pivot)

    # ── 4. Doc rating distribution ─────────────────────────────────────────────
    section("DOC-LEVEL RATING DISTRIBUTION  (% of all doc judgements)")
    drd = doc_rating_distribution(frames)
    console.print(rich_table(drd, "Doc Rating Distribution", float_fmt=".1f"))


if __name__ == "__main__":
    main()
