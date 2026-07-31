#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple


# ----------------------------
# Section removal helpers
# ----------------------------

HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")

def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()

def remove_named_sections(md: str, titles: Iterable[str]) -> Tuple[str, bool]:
    wanted = {_norm_title(t) for t in titles}
    lines = md.splitlines(keepends=True)

    changed = False
    i = 0
    out: List[str] = []

    while i < len(lines):
        m = HEADING_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        level = len(m.group("hashes"))
        title = _norm_title(m.group("title"))

        if title not in wanted:
            out.append(lines[i])
            i += 1
            continue

        # Skip this heading and everything until next heading of level <= this level
        changed = True
        i += 1
        while i < len(lines):
            m2 = HEADING_RE.match(lines[i])
            if m2 and len(m2.group("hashes")) <= level:
                break
            i += 1

    return "".join(out), changed


# ----------------------------
# Affiliations block removal
# ----------------------------

AFFIL_LINE_RE = re.compile(
    r"""
    ^\s*(
        (\$\^\{\d+(,\s*\d+)*\}\$)          # $^{1}$ or $^{1,2}$
        |(<sup>\d+</sup>)                 # <sup>3</sup>
        |(\d{1,3})(?!\d)                  # 4  (but not 1234...)
    )
    \s*[\)\.\:]?\s*
    .+
    """,
    re.VERBOSE,
)

CORRESP_RE = re.compile(r"^\s*\*?\s*correspondence\s*:", re.IGNORECASE)
EMAIL_RE = re.compile(r"\bE-?mail\s*:", re.IGNORECASE)

def remove_affiliations_block(md: str) -> Tuple[str, bool]:
    lines = md.splitlines(keepends=True)

    # find first "Abstract" heading
    abstract_idx = None
    for idx, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if m and _norm_title(m.group("title")) == "abstract":
            abstract_idx = idx
            break
    if abstract_idx is None:
        return md, False

    # walk backwards from the line before Abstract
    j = abstract_idx - 1
    while j >= 0 and lines[j].strip() == "":
        j -= 1
    end = j
    if end < 0:
        return md, False

    start = end
    affil_hits = 0
    while start >= 0:
        s = lines[start].rstrip("\n")
        if s.strip() == "":
            break

        if AFFIL_LINE_RE.match(s) or CORRESP_RE.match(s) or EMAIL_RE.search(s):
            if AFFIL_LINE_RE.match(s):
                affil_hits += 1
            start -= 1
            continue

        break

    start += 1

    # only remove if it looks like real affiliations
    if affil_hits >= 3 and start <= end:
        new_lines = lines[:start] + ["\n"] + lines[abstract_idx:]
        return "".join(new_lines), True

    return md, False


# ----------------------------
# Processing
# ----------------------------

def transform_text(original: str) -> Tuple[str, bool]:
    text, c1 = remove_named_sections(
        original,
        titles=["references", "reference", "bibliography", "works cited", "literature cited"],
    )
    text, c2 = remove_affiliations_block(text)
    return text, (c1 or c2)

def output_path_for(input_file: Path, root: Path, out_dir: Path) -> Path:
    rel = input_file.relative_to(root)
    return out_dir / rel

def main():
    ap = argparse.ArgumentParser(
        description="Remove References sections and affiliation blocks from markdown files under a directory."
    )
    ap.add_argument("root", type=Path, help="Root directory to scan (recursively).")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to write cleaned files into.")
    ap.add_argument("--pattern", default="*.md", help="Glob pattern to match (default: *.md).")
    ap.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    ap.add_argument(
        "--print-unchanged",
        action="store_true",
        help="In --dry-run, print files that would NOT change.",
    )
    ap.add_argument(
        "--copy-unchanged",
        action="store_true",
        help="Also copy files that would not change into --out-dir (default: skip them).",
    )
    args = ap.parse_args()

    root: Path = args.root.resolve()
    out_dir: Path = args.out_dir.resolve()

    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    total = 0
    changed = 0
    skipped_unchanged = 0
    unchanged_list: List[Path] = []

    for in_path in root.rglob(args.pattern):
        if not in_path.is_file():
            continue

        total += 1
        original = in_path.read_text(encoding="utf-8", errors="replace")
        cleaned, would_change = transform_text(original)

        if not would_change and not args.copy_unchanged:
            skipped_unchanged += 1
            if args.dry_run and args.print_unchanged:
                unchanged_list.append(in_path)
            continue

        out_path = output_path_for(in_path, root, out_dir)
        if args.dry_run:
            action = "WRITE" if would_change else "COPY"
            print(f"{action}: {in_path} -> {out_path}")
            if would_change:
                changed += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(cleaned if would_change else original, encoding="utf-8")
        if would_change:
            changed += 1

    if args.dry_run:
        print(f"\nDRY RUN: {changed} of {total} files would be changed.")
        if args.copy_unchanged:
            print("DRY RUN: unchanged files would also be copied.")
        else:
            print(f"DRY RUN: {skipped_unchanged} unchanged files would be skipped (use --copy-unchanged to copy them).")

        if args.print_unchanged and unchanged_list:
            print("\nFiles that would NOT be changed (and would be skipped):")
            for p in unchanged_list:
                print(str(p))
    else:
        print(f"UPDATED: wrote {changed} changed files into {out_dir}.")
        if args.copy_unchanged:
            print("UPDATED: unchanged files were also copied.")
        else:
            print(f"UPDATED: {skipped_unchanged} unchanged files were skipped (use --copy-unchanged to copy them).")


if __name__ == "__main__":
    main()
