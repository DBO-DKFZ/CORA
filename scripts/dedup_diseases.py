"""
dedup_diseases.py (Scipt 2/3)

Pass 1: Deduplicate disease names within small batches (BATCH_SIZE=10).
Pass 2: Re-deduplicate the reduced canonical list in larger batches to catch
        cross-batch synonyms that didn't appear together in pass 1.

Input:  INPUT_CSV  — raw question_categories.csv
Output: DEDUP_CSV  — dedup_map.csv with columns:
          original_disease, canonical_disease, disease_category
"""

import argparse
import os
import json
import time
import pandas as pd
import anthropic
from dotenv import load_dotenv

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL      = "claude-sonnet-4-6"
BATCH_SIZE_P1 = 10   # small batches for pass 1
BATCH_SIZE_P2 = 50   # larger batches for pass 2 cross-batch merging

TEST_MODE        = False
TEST_SAMPLE_SIZE = 100

# Input/output paths default to the medqa run; override with --input/--output
# to apply the same pipeline to another dataset (e.g. pubmed).
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input",  default="./results/question_categories.csv",
                    help="raw question_categories.csv (output of categorize_questions.py)")
parser.add_argument("--output", default="./results/dedup_map.csv",
                    help="dedup_map.csv to write")
args = parser.parse_args()
INPUT_CSV = args.input
DEDUP_CSV = args.output

load_dotenv()
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY not found. Add it to a .env file: ANTHROPIC_API_KEY=sk-ant-...")
client = anthropic.Anthropic(api_key=api_key)

# ── 1. Load & basic cleaning ───────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
if TEST_MODE:
    df = df.sample(n=min(TEST_SAMPLE_SIZE, len(df)), random_state=42)
    print(f"⚠️  TEST MODE: using {len(df)} rows (set TEST_MODE=False for full run)")

df = df.dropna(subset=["disease", "disease_category"])
df["disease"]          = df["disease"].str.strip().str.title()
df["disease_category"] = df["disease_category"].str.strip().str.replace("_", " ").str.title()

# Remove rows where disease name is actually a category name or "General"
category_names = set(df["disease_category"].unique())
df = df[~df["disease"].isin(category_names)]
df = df[df["disease"].str.lower() != "general"]

# Assign each disease to its most frequent category
freq = (df.groupby(["disease", "disease_category"])
          .size().reset_index(name="count"))
primary = (freq.sort_values("count", ascending=False)
               .drop_duplicates(subset="disease")
               [["disease", "disease_category"]])

all_diseases = sorted(primary["disease"].tolist())
print(f"\nUnique disease names before dedup: {len(all_diseases)}")
print("\nCategory distribution:")
print(primary["disease_category"].value_counts().to_string())


# ── 2. Dedup helper ────────────────────────────────────────────────────────────
def deduplicate_batch(disease_list: list[str]) -> dict:
    """
    Send a batch to Claude and get back {original_name: canonical_name}.
    """
    prompt = f"""You are a medical terminology expert. Below is a list of skin disease names.
Many may be duplicates: same disease with different spellings, accents, British/American
variants, synonym names, or with/without parenthetical suffixes.

For each disease name, return the single best canonical medical name:
- Merge true synonyms (e.g. "Tinea Versicolor" / "Pityriasis Versicolor" → "Tinea Versicolor")
- Merge spelling variants (e.g. "Behcet Disease" / "Behçet's Disease" → "Behçet's Disease")
- Merge specificity variants of the same base disease (e.g. "Varicella", "Varicella (Chickenpox)" → "Varicella")
- Keep genuinely distinct diseases separate (e.g. "Tinea Capitis" ≠ "Tinea Corporis")
- Use the most widely accepted modern medical name

Return ONLY a JSON object: {{"Original Name": "Canonical Name", ...}}
No explanation, no markdown, no extra text.

Diseases:
{json.dumps(disease_list, indent=2)}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── 3. Pass 1 — deduplicate within small batches ──────────────────────────────
print(f"\n── Pass 1: dedup within batches of {BATCH_SIZE_P1} ──────────────────────────────")
pass1_map = {}  # original -> canonical after pass 1

batches = [all_diseases[i:i+BATCH_SIZE_P1]
           for i in range(0, len(all_diseases), BATCH_SIZE_P1)]

for idx, batch in enumerate(batches):
    print(f"  Batch {idx+1}/{len(batches)} ({len(batch)} diseases)...", end=" ", flush=True)
    result = deduplicate_batch(batch)
    pass1_map.update(result)
    print(f"done → {len(set(result.values()))} canonical")
    if idx < len(batches) - 1:
        time.sleep(0.3)

# Reduce to unique canonical names after pass 1
canonical_after_p1 = sorted(set(pass1_map.values()))
print(f"\nUnique diseases after pass 1: {len(canonical_after_p1)}")


# ── 4. Pass 2 — cross-batch merging on reduced list ───────────────────────────
print(f"\n── Pass 2: cross-batch merging in batches of {BATCH_SIZE_P2} ─────────────────────")
pass2_map = {}  # canonical_p1 -> canonical_p2

batches2 = [canonical_after_p1[i:i+BATCH_SIZE_P2]
            for i in range(0, len(canonical_after_p1), BATCH_SIZE_P2)]

for idx, batch in enumerate(batches2):
    print(f"  Batch {idx+1}/{len(batches2)} ({len(batch)} diseases)...", end=" ", flush=True)
    result = deduplicate_batch(batch)
    pass2_map.update(result)
    print(f"done → {len(set(result.values()))} canonical")
    if idx < len(batches2) - 1:
        time.sleep(0.3)

canonical_after_p2 = sorted(set(pass2_map.values()))
print(f"\nUnique diseases after pass 2: {len(canonical_after_p2)}")


# ── 5. Compose final mapping & save ───────────────────────────────────────────
# Chain: original -> pass1_canonical -> pass2_canonical
final_map = {
    orig: pass2_map.get(p1, p1)
    for orig, p1 in pass1_map.items()
}

result_df = primary.copy()
result_df["canonical_disease"] = result_df["disease"].map(final_map).fillna(result_df["disease"])
result_df = result_df.rename(columns={"disease": "original_disease"})

# One row per canonical disease (assign to most frequent category)
deduped = (result_df.groupby("canonical_disease")
           .agg(disease_category=("disease_category",
                                  lambda x: x.value_counts().index[0]))
           .reset_index())

# ── 5b. Manual patches — missed merges not caught by LLM passes ───────────────
# Same disease, different names that ended up as separate canonicals after pass 2.
MANUAL_PATCHES = {
    "Genital Warts":   "Condylomata Acuminata",       # common name vs medical term
    "Cutaneous Warts": "Verruca Vulgaris",             # generic vs specific
    "Herpes Simplex":  "Herpes Simplex Virus Infection",
    "Herpes Labialis": "Herpes Simplex Labialis",
}
result_df["canonical_disease"] = result_df["canonical_disease"].replace(MANUAL_PATCHES)
n_after_patch = result_df["canonical_disease"].nunique()
print(f"  After manual patches     : {n_after_patch}")

# Save full mapping (original → canonical → category)
result_df.to_csv(DEDUP_CSV, index=False)

print(f"\n── Done ──────────────────────────────────────────────────────────────────")
print(f"Saved dedup map → {DEDUP_CSV}")
print(f"  Original unique diseases : {len(all_diseases)}")
print(f"  After pass 1             : {len(canonical_after_p1)}")
print(f"  After pass 2             : {len(canonical_after_p2)}")
print(f"\nCategory distribution after dedup:")
print(deduped["disease_category"].value_counts().to_string())