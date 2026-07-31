"""
assign_subcategories.py

Takes the dedup map produced by dedup_diseases.py and assigns a medical
subcategory to each canonical disease, then merges everything back onto
the original rows preserving all metadata.

Input:  INPUT_CSV  — raw question_categories.csv
        DEDUP_CSV  — dedup_map.csv (output of dedup_diseases.py)
Output: OUTPUT_CSV — cleaned_diseases.csv with all original columns plus:
          original_disease, canonical_disease, disease_subcategory
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
BATCH_SIZE = 10   # diseases per API call

TEST_MODE        = False
TEST_SAMPLE_SIZE = 100

# Paths default to the medqa run; override with --input/--dedup/--output to apply
# the same pipeline to another dataset (e.g. pubmed).
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input",  default="./results/question_categories.csv",
                    help="raw question_categories.csv (output of categorize_questions.py)")
parser.add_argument("--dedup",  default="./results/dedup_map.csv",
                    help="dedup_map.csv (output of dedup_diseases.py)")
parser.add_argument("--output", default="./results/question_categories_cleaned.csv",
                    help="cleaned CSV to write")
args = parser.parse_args()
INPUT_CSV  = args.input
DEDUP_CSV  = args.dedup
OUTPUT_CSV = args.output

load_dotenv()
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY not found. Add it to a .env file: ANTHROPIC_API_KEY=sk-ant-...")
client = anthropic.Anthropic(api_key=api_key)

# ── Valid subcategories per category ──────────────────────────────────────────
VALID_SUBCATS = {
    "Inflammatory":        ["Autoimmune", "Eczematous", "Neutrophilic", "Papulosquamous", "Vascular & Urticarial"],
    "Infectious":          ["Bacterial", "Viral", "Fungal", "Parasitic & Other"],
    "Neoplastic":          ["Malignant", "Premalignant", "Benign", "Vascular Tumours"],
    "Pigmentation":        ["Hyperpigmentation", "Hypopigmentation", "Mixed / Vascular"],
    "Genetic Rare":        ["Genodermatoses", "Metabolic-Genetic", "Syndromic"],
    "Hair Nail":           ["Hair Loss", "Hair Excess", "Nail Disorders"],
    "Metabolic Systemic":  ["Endocrine-Associated", "Nutritional", "Systemic Disease Manifestations"],
    "Other":               ["Traumatic & Environmental", "Psychocutaneous", "Drug-Induced", "Unclassified"],
}


# ── 1. Load dedup map ──────────────────────────────────────────────────────────
dedup_df = pd.read_csv(DEDUP_CSV)
# dedup_df has: original_disease, canonical_disease, disease_category
canonical_to_category = dedup_df.drop_duplicates("canonical_disease") \
                                 .set_index("canonical_disease")["disease_category"].to_dict()
original_to_canonical = dedup_df.set_index("original_disease")["canonical_disease"].to_dict()

canonical_diseases = sorted(canonical_to_category.keys())
print(f"Canonical diseases loaded from dedup map: {len(canonical_diseases)}")

if TEST_MODE:
    canonical_diseases = canonical_diseases[:TEST_SAMPLE_SIZE]
    print(f"⚠️  TEST MODE: classifying first {len(canonical_diseases)} canonical diseases")


# ── 2. Subcategory assignment ──────────────────────────────────────────────────
def assign_subcategory_batch(category: str, diseases: list[str]) -> dict:
    """
    Classify a batch of diseases into subcategories for one specific category.
    Returns {disease: subcategory}
    """
    valid = VALID_SUBCATS.get(category, ["Unclassified"])
    valid_str = "\n".join(f"  - {s}" for s in valid)

    prompt = f"""You are a dermatology expert. Assign each disease to the most appropriate subcategory.

Primary category: {category}
Valid subcategories (use ONLY these exact names, nothing else):
{valid_str}

Rules:
- Pick the single most clinically appropriate subcategory
- Do NOT invent or use any subcategory not listed above
- If genuinely unclear, use "Unclassified"

Return ONLY a JSON object: {{"disease name": "subcategory name", ...}}
No explanation, no markdown, no extra text.

Diseases to classify:
{json.dumps(diseases, indent=2)}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


print("\n── Assigning subcategories ───────────────────────────────────────────────")

# Group canonical diseases by category
by_cat: dict[str, list[str]] = {}
for d in canonical_diseases:
    cat = canonical_to_category.get(d, "Other")
    by_cat.setdefault(cat, []).append(d)

subcat_map: dict[str, str] = {}
total_cats = len(by_cat)

for cat_idx, (cat, diseases) in enumerate(by_cat.items()):
    batches = [diseases[i:i+BATCH_SIZE] for i in range(0, len(diseases), BATCH_SIZE)]
    n = len(batches)
    for bidx, batch in enumerate(batches):
        print(f"  [{cat_idx+1}/{total_cats}] {cat} — batch {bidx+1}/{n} "
              f"({len(batch)} diseases)...", end=" ", flush=True)
        result = assign_subcategory_batch(cat, batch)
        subcat_map.update(result)
        print("done")
        time.sleep(0.3)

print(f"\nSubcategories assigned: {len(subcat_map)}")


# ── 3. Load raw CSV and merge everything back ──────────────────────────────────
# IMPORTANT: every row is preserved — no rows are dropped.
# Cleaning (title-case, category fix) is applied but rows with "General"
# or category-name leakage just get canonical_disease = original value
# and disease_subcategory = "Unclassified".
raw = pd.read_csv(INPUT_CSV)
if TEST_MODE:
    raw = raw.head(TEST_SAMPLE_SIZE)

raw["disease"]          = raw["disease"].fillna("").str.strip().str.title()
raw["disease_category"] = raw["disease_category"].fillna("").str.strip().str.replace("_", " ").str.title()

# Add canonical disease + subcategory columns — no rows dropped
raw["original_disease"]    = raw["disease"]
raw["canonical_disease"]   = raw["disease"].map(original_to_canonical).fillna(raw["disease"])
raw["disease_subcategory"] = raw["canonical_disease"].map(subcat_map).fillna("Unclassified")

# Update disease_category from dedup map where available
raw["disease_category"] = raw["canonical_disease"].map(canonical_to_category).fillna(raw["disease_category"])

# Final column selection — drop raw disease and noisy raw_llm_response
keep = [c for c in raw.columns if c not in ["disease", "raw_llm_response"]]
final = raw[keep]

final.to_csv(OUTPUT_CSV, index=False)

print(f"\n── Done ──────────────────────────────────────────────────────────────────")
print(f"Saved → {OUTPUT_CSV}")
print(f"Rows: {len(final)}")
print(f"\nSubcategory distribution:")
deduped_view = final.drop_duplicates("canonical_disease")
print(deduped_view.groupby(["disease_category", "disease_subcategory"])
      .size().reset_index(name="count").to_string(index=False))