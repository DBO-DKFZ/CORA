"""
label_question_structure.py

Classify each question's structure as one_liner, short_stem, or vignette
and write results to a new CSV (question_id, question_structure).

Resumable: already-labelled question IDs in the output CSV are skipped.
Each result is written immediately so progress is not lost.

Usage:
    python scripts/label_question_structure.py
"""

import argparse
import csv
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from together import Together
from tqdm import tqdm

SYSTEM_PROMPT = """You are a medical education expert. Classify a multiple-choice question's structure into one of three categories.

Return ONLY a JSON object with a single key — no markdown, no explanation:
{"question_structure": "<one_liner|short_stem|vignette>"}

Definitions:

  "one_liner"   — Pure factual recall with no scenario of any kind. The question is a
                  direct definition, association, or classification with no patient, no
                  research study, and no clinical context.
                  e.g. "X is caused by ___", "X sign is seen in ___", "What is erythema?"
                  A question with a patient, a researcher, a study, or any described
                  scenario is NOT a one_liner even if it is short.

  "short_stem"  — A patient is mentioned but the clinical picture is sparse: typically
                  just age + 1 symptom or a single finding, with no physical exam,
                  no lab/biopsy results, and no meaningful history. The answer follows
                  directly without differential reasoning.
                  Rule: if you can remove the patient framing and it becomes a one_liner,
                  it is a short_stem.

  "vignette"    — A rich clinical scenario with at least THREE of the following present:
                  age, chief complaint/duration, physical exam findings, lab/biopsy/imaging
                  results, relevant past history or medications, vital signs.
                  Requires working through a clinical scenario or differential to answer.
"""

VALID_STRUCTURES = {"one_liner", "short_stem", "vignette"}


DEFAULTS = {
    "input_json":      "/home/t252a/data/derma/medqa_derma_final.json",
    "output_csv":      "results/question_structure.csv",
    "categories_csv":  "results/question_categories_cleaned.csv",
    "model":           "gpt-5",
    "provider":        "openai",
    "max_workers":     8,
    "verbose":         True,
    "local_base_url":  "",
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


def load_questions(path: str) -> dict[str, str]:
    """Return {question_id: formatted_question_text} from the input JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", data)
    result = {}
    for qid, q in questions.items():
        text = q["question"] + "\n\n"
        for letter, option in q.get("answer_options", {}).items():
            text += f"{letter}. {option}\n"
        result[qid] = text.strip()
    return result


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "question_id" not in (reader.fieldnames or []):
            return set()
        return {row["question_id"] for row in reader}


def append_row(path: Path, question_id: str, structure: str) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question_structure"])
        if write_header:
            writer.writeheader()
        writer.writerow({"question_id": question_id, "question_structure": structure})


def make_client(provider: str, local_base_url: str = ""):
    load_dotenv()
    provider = provider.lower()
    if provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "together":
        return Together(api_key=os.getenv("TOGETHER_API_KEY"))
    elif provider == "anthropic":
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "local":
        return OpenAI(
            base_url=local_base_url or os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("LOCAL_API_KEY", "local-dev-key"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


def call_llm(client, model: str, provider: str, question_text: str) -> str:
    if provider.lower() == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question_text}],
        )
        return response.content[0].text

    kwargs = {"response_format": {"type": "json_object"}}
    if "gpt-5" in model:
        kwargs["reasoning_effort"] = "minimal"
    else:
        kwargs["temperature"] = 0
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
        ],
        **kwargs,
    )
    return response.choices[0].message.content


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "rate limit" in msg or "rate_limit" in msg or "429" in msg or "too many requests" in msg


def classify_one(qid: str, question_text: str, client, model: str, provider: str, verbose: bool) -> str | None:
    """Return the question_structure label, or None on unrecoverable failure."""
    raw = None
    for attempt in range(3):
        try:
            raw = call_llm(client, model, provider, question_text)
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r'^```(?:json)?\s*', '', text)
                text = re.sub(r'\s*```$', '', text)
            start, end = text.find('{'), text.rfind('}')
            if start != -1 and end != -1:
                text = text[start:end + 1]
            parsed = json.loads(text)
            structure = parsed.get("question_structure", "").strip()
            if structure not in VALID_STRUCTURES:
                raise ValueError(f"Invalid question_structure: {structure!r}")
            return structure
        except Exception as e:
            rate_limited = _is_rate_limit(e)
            wait = 30 * (2 ** attempt) if rate_limited else 5 * (2 ** attempt)
            if verbose:
                kind = "rate limit" if rate_limited else "parse error" if raw is not None else "error"
                tqdm.write(f"  {kind} on {qid} (attempt {attempt + 1}): {e} — retrying in {wait}s")
                if raw is not None and not rate_limited:
                    tqdm.write(f"    LLM output: {raw[:200]!r}")
                tqdm.write(traceback.format_exc())
            if attempt < 2:
                time.sleep(wait)

    if verbose:
        tqdm.write(f"  Skipping {qid} after 3 failed attempts")
    return None


def merge_into_categories(structure_csv: Path, categories_csv: Path, verbose: bool) -> None:
    """Add question_structure column to categories_csv in place."""
    if not structure_csv.exists():
        print(f"  Structure CSV not found: {structure_csv} — skipping merge.")
        return
    if not categories_csv.exists():
        print(f"  Categories CSV not found: {categories_csv} — skipping merge.")
        return

    # Load structure labels
    structure_map: dict[str, str] = {}
    with open(structure_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            structure_map[row["question_id"]] = row["question_structure"]

    # Read existing categories CSV
    with open(categories_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "question_structure" not in fieldnames:
        fieldnames = fieldnames + ["question_structure"]

    matched = 0
    for row in rows:
        qid = row["question_id"]
        if qid in structure_map:
            row["question_structure"] = structure_map[qid]
            matched += 1
        elif "question_structure" not in row:
            row["question_structure"] = ""

    with open(categories_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"  Merged {matched}/{len(rows)} rows into {categories_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_json = cfg["input_json"]
    output_csv = Path(cfg["output_csv"])
    model = cfg["model"]
    provider = cfg["provider"]
    max_workers = cfg.get("max_workers", 4)
    verbose = cfg.get("verbose", True)
    local_base_url = cfg.get("local_base_url", "")

    questions = load_questions(input_json)
    done_ids = load_done_ids(output_csv)
    remaining = [(qid, text) for qid, text in questions.items() if qid not in done_ids]

    print(f"Total: {len(questions)}  |  done: {len(done_ids)}  |  remaining: {len(remaining)}")
    if not remaining:
        print("Nothing to do.")
        return

    client = make_client(provider, local_base_url)
    errors = 0
    bar = tqdm(total=len(remaining), desc="Labelling", unit="q")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(classify_one, qid, text, client, model, provider, verbose): qid
            for qid, text in remaining
        }
        for future in as_completed(futures):
            qid = futures[future]
            try:
                structure = future.result()
            except Exception as e:
                tqdm.write(f"  Unexpected error on {qid}: {e}")
                structure = None

            if structure is None:
                errors += 1
            else:
                append_row(output_csv, qid, structure)
                if verbose:
                    tqdm.write(f"  [{qid}] → {structure}")

            bar.update(1)
            bar.set_postfix(errors=errors)

    bar.close()
    print(f"\nDone. Output: {output_csv}  |  errors: {errors}")
    if errors:
        print(f"  Re-run to retry {errors} failed questions.")

    categories_csv = Path(cfg["categories_csv"])
    print(f"\nMerging into {categories_csv} ...")
    merge_into_categories(output_csv, categories_csv, verbose)


if __name__ == "__main__":
    main()
