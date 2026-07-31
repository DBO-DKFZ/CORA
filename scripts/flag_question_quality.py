"""
flag_question_quality.py

Runs one of two annotation passes over the question dataset, selected by MODE:

  MODE = "quality"
    Detects:
      - language:      the language the question is written in
      - is_valid:      false if the question is garbled, truncated, or nonsensical
      - quality_notes: brief explanation when is_valid is false

  MODE = "difficulty"
    Estimates question difficulty for an average dermatologist:
      - difficulty:           "easy" | "medium" | "hard"
      - difficulty_rationale: one sentence explaining the rating

Results are written to a sidecar CSV after each LLM call and merged into
CATEGORIES_CSV once all questions are processed without errors.

Resumable: question IDs already present in the sidecar are skipped.

Usage:
    python scripts/flag_question_quality.py
"""

import csv
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm
import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from together import Together

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_JSON      = "/home/t252a/data/derma/medqa_derma_final.json"
CATEGORIES_CSV  = "results/question_categories_cleaned.csv"

MODEL           = "gpt-5"#"Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
PROVIDER        = "openai"   # openai | anthropic | together | local
MAX_WORKERS     = 8
VERBOSE         = True

# "quality" → flag language / validity
# "difficulty" → estimate difficulty for an average dermatologist
MODE            = "difficulty"
# ──────────────────────────────────────────────────────────────────────────────

# ── Mode-specific settings ─────────────────────────────────────────────────────

_QUALITY_PROMPT = """You are a quality auditor for a medical multiple-choice question dataset.

For each question, return a JSON object — no markdown, no commentary, ONLY the JSON:

{
  "language": "<lowercase English word for the language, e.g. 'english', 'german', 'french', 'portuguese', 'spanish', 'arabic', 'chinese', 'other'>",
  "is_valid": <true|false>,
  "quality_notes": "<one short sentence describing the specific structural problem if is_valid is false, otherwise empty string>"
}

Set is_valid to false ONLY for unambiguous structural defects:
- Two or more answer options have identical text
- The question text or answer options contain random characters or text that is not a word in any language and cannot plausibly be a misspelling or abbreviation of a medical term
- The question explicitly signals broken content (e.g. "(wrong options)", "(incomplete)", "(corrupted)" in the title)
- Answer options are missing entirely or not labeled with letters

When in doubt, set is_valid to true. Do NOT flag:
- Misspellings of medical terms (e.g. "pemphigiod", "saborrhea" — these are typos, not gibberish)
- Non-English questions
- Difficult, ambiguous, or poorly worded questions that are still interpretable
- Unusual abbreviations, eponyms, or drug names you don't recognise
- Combination options like "a and b", "both 1 and 2", "all of the above" """

_DIFFICULTY_PROMPT = """You are an expert dermatologist assessing the difficulty of dermatology multiple-choice questions.

Rate each question from the perspective of a competent, board-certified general dermatologist (not a subspecialty expert).

Return a JSON object — no markdown, no commentary, ONLY the JSON:

{
  "difficulty": "<easy|medium|hard>",
  "difficulty_rationale": "<one sentence explaining the rating>"
}

Guidelines:
- easy:   Tests core knowledge any practicing dermatologist would recall without hesitation
          (common conditions, first-line treatments, classic presentations, basic pathophysiology).
- medium: Requires solid specialist training; involves less-common conditions, second-line or
          combination therapies, nuanced clinical distinctions, or recall of specific criteria/stages.
- hard:   Demands subspecialty depth, rare disease knowledge, current evidence-based guidelines,
          complex pathophysiology, or multi-step reasoning where plausible distractors are difficult
          to rule out even with strong general dermatology training.

Focus on the knowledge required to choose the correct answer, not on the length or wording of the question."""

_MODE_CONFIG = {
    "quality": {
        "system_prompt":  _QUALITY_PROMPT,
        "sidecar_csv":    "results/question_quality.csv",
        "fieldnames":     ["question_id", "language", "is_valid", "quality_notes", "raw_llm_response"],
        "new_columns":    ["language", "is_valid", "quality_notes"],
        "include_answer": False,
        "normalise":      lambda p: {
            "language":      str(p.get("language", "unknown")).lower().strip(),
            "is_valid":      bool(p.get("is_valid", True)),
            "quality_notes": str(p.get("quality_notes", "")).strip(),
        },
        "verbose_flag":   lambda p: not p["is_valid"],
        "verbose_msg":    lambda q, p: (
            f"  [{q['question_id']}] INVALID ({p['language']}): {p['quality_notes']}"
        ),
    },
    "difficulty": {
        "system_prompt":  _DIFFICULTY_PROMPT,
        "sidecar_csv":    "results/question_difficulty.csv",
        "fieldnames":     ["question_id", "difficulty", "difficulty_rationale", "raw_llm_response"],
        "new_columns":    ["difficulty", "difficulty_rationale"],
        "include_answer": True,
        "normalise":      lambda p: {
            "difficulty":           str(p.get("difficulty", "medium")).lower().strip(),
            "difficulty_rationale": str(p.get("difficulty_rationale", "")).strip(),
        },
        "verbose_flag":   lambda p: True,
        "verbose_msg":    lambda q, p: (
            f"  [{q['question_id']}] {p['difficulty'].upper()}: {p['difficulty_rationale']}"
        ),
    },
}

# Resolved at runtime in main()
SYSTEM_PROMPT      = None
SIDECAR_CSV        = None
SIDECAR_FIELDNAMES = None
NEW_COLUMNS        = None
_INCLUDE_ANSWER    = None
_NORMALISE         = None
_VERBOSE_FLAG      = None
_VERBOSE_MSG       = None


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", data)
    return [{"question_id": k, **v} for k, v in questions.items()]


def format_question(q: dict, include_answer: bool = False) -> str:
    text = q["question"] + "\n\n"
    for letter, option in q["answer_options"].items():
        text += f"{letter}. {option}\n"
    if include_answer:
        letter = q.get("correct_choice", ["?"])[0]
        text += f"\nCorrect answer: {letter}. {q.get('correct_answer', '')}"
    return text.strip()


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "question_id" not in (reader.fieldnames or []):
            return set()
        return {row["question_id"] for row in reader}


def append_row(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SIDECAR_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_client(provider: str):
    provider = provider.lower()
    if provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "together":
        return Together(api_key=os.getenv("TOGETHER_API_KEY"))
    elif provider == "anthropic":
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "local":
        return OpenAI(
            base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("LOCAL_API_KEY", "local-dev-key"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


def call_llm(client, model: str, provider: str, question_text: str, system_prompt: str) -> str:
    if provider.lower() == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": question_text}],
        )
        if not response.content:
            raise ValueError(f"Empty Anthropic response: stop_reason={response.stop_reason}")
        return response.content[0].text

    kwargs = {"response_format": {"type": "json_object"}}
    if "gpt-5" in model:
        kwargs["reasoning_effort"] = "minimal"
    else:
        kwargs["temperature"] = 0
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text},
        ],
        **kwargs,
    )
    if not response.choices:
        raise ValueError(f"Empty choices in response")
    return response.choices[0].message.content


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "rate limit" in msg or "rate_limit" in msg or "429" in msg or "too many requests" in msg


def process_question(q: dict, client, model: str, provider: str, verbose: bool) -> tuple[dict, str] | None:
    question_text = format_question(q, include_answer=_INCLUDE_ANSWER)
    raw = None

    for attempt in range(3):
        try:
            raw = call_llm(client, model, provider, question_text, SYSTEM_PROMPT)
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r'^```(?:json)?\s*', '', text)
                text = re.sub(r'\s*```$', '', text)
            start, end = text.find('{'), text.rfind('}')
            if start != -1 and end != -1:
                text = text[start:end + 1]
            parsed = json.loads(text)
            parsed = _NORMALISE(parsed)
            return parsed, raw
        except Exception as e:
            rate_limited = _is_rate_limit(e)
            wait = 30 * (2 ** attempt) if rate_limited else 5 * (2 ** attempt)
            if verbose:
                kind = "rate limit" if rate_limited else "parse error" if raw is not None else "error"
                tqdm.tqdm.write(f"  {kind} on {q['question_id']} (attempt {attempt + 1}): {e} — retrying in {wait}s")
                if raw is not None and not rate_limited:
                    tqdm.tqdm.write(f"    LLM output: {raw[:200]!r}")
                tqdm.tqdm.write(traceback.format_exc())
            if attempt < 2:
                time.sleep(wait)

    if verbose:
        tqdm.tqdm.write(f"  Skipping {q['question_id']} after 3 failed attempts")
    return None


def merge_into_categories(categories_csv: Path, sidecar_csv: Path, new_columns: list[str], verbose: bool) -> None:
    """Merge new_columns from sidecar_csv into categories_csv in-place."""
    if not sidecar_csv.exists():
        print("No sidecar found — nothing to merge.")
        return

    sidecar: dict[str, dict] = {}
    with open(sidecar_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sidecar[row["question_id"]] = {col: row.get(col, "") for col in new_columns}

    with open(categories_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        rows = list(reader)

    added_fields = [f for f in new_columns if f not in existing_fields]
    all_fields = existing_fields + added_fields

    n_updated = 0
    for row in rows:
        qid = row["question_id"]
        if qid in sidecar:
            for col in new_columns:
                row[col] = sidecar[qid].get(col, row.get(col, ""))
            n_updated += 1
        else:
            for col in added_fields:
                row.setdefault(col, "")

    with open(categories_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"  Merged {n_updated}/{len(rows)} rows into {categories_csv}")
        if "is_valid" in new_columns:
            n_invalid = sum(1 for r in rows if str(r.get("is_valid", "")).lower() == "false")
            n_nonenglish = sum(1 for r in rows if r.get("language", "") not in ("", "english"))
            print(f"  Non-English: {n_nonenglish}  |  Invalid: {n_invalid}")
        if "difficulty" in new_columns:
            for level in ("easy", "medium", "hard"):
                n = sum(1 for r in rows if r.get("difficulty", "") == level)
                print(f"  {level.capitalize()}: {n}")


def main() -> None:
    global SYSTEM_PROMPT, SIDECAR_CSV, SIDECAR_FIELDNAMES, NEW_COLUMNS, _INCLUDE_ANSWER, _NORMALISE, _VERBOSE_FLAG, _VERBOSE_MSG

    if MODE not in _MODE_CONFIG:
        raise ValueError(f"Unknown MODE {MODE!r}. Choose 'quality' or 'difficulty'.")

    cfg = _MODE_CONFIG[MODE]
    SYSTEM_PROMPT      = cfg["system_prompt"]
    SIDECAR_CSV        = cfg["sidecar_csv"]
    SIDECAR_FIELDNAMES = cfg["fieldnames"]
    NEW_COLUMNS        = cfg["new_columns"]
    _INCLUDE_ANSWER    = cfg["include_answer"]
    _NORMALISE         = cfg["normalise"]
    _VERBOSE_FLAG      = cfg["verbose_flag"]
    _VERBOSE_MSG       = cfg["verbose_msg"]

    categories_csv = Path(CATEGORIES_CSV)
    sidecar_csv    = Path(SIDECAR_CSV)

    questions = load_questions(INPUT_JSON)
    done_ids  = load_done_ids(sidecar_csv)
    remaining = [q for q in questions if q["question_id"] not in done_ids]

    if VERBOSE:
        print(f"Mode: {MODE}  |  Total: {len(questions)}  |  done: {len(done_ids)}  |  remaining: {len(remaining)}")

    if not remaining:
        print("All questions processed. Merging into categories CSV...")
        merge_into_categories(categories_csv, sidecar_csv, NEW_COLUMNS, VERBOSE)
        return

    client = make_client(PROVIDER)
    errors = 0
    bar = tqdm.tqdm(total=len(remaining), desc=f"Running {MODE}", unit="q")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_question, q, client, MODEL, PROVIDER, VERBOSE): q
            for q in remaining
        }
        for future in as_completed(futures):
            q = futures[future]
            try:
                result = future.result()
            except Exception as e:
                tqdm.tqdm.write(f"  Unexpected error on {q['question_id']}: {e}")
                result = None

            if result is None:
                errors += 1
            else:
                parsed, raw = result
                row = {"question_id": q["question_id"], "raw_llm_response": raw}
                row.update(parsed)
                append_row(sidecar_csv, row)
                if VERBOSE and _VERBOSE_FLAG(parsed):
                    tqdm.tqdm.write(_VERBOSE_MSG(q, parsed))

            bar.update(1)
            bar.set_postfix(errors=errors)

    bar.close()
    print(f"\nDone. Sidecar: {sidecar_csv}  |  errors: {errors}")

    if errors == 0:
        print("\nMerging into categories CSV...")
        merge_into_categories(categories_csv, sidecar_csv, NEW_COLUMNS, VERBOSE)
    else:
        print(f"\n{errors} questions failed. Re-run to retry; merge runs automatically once errors=0.")


if __name__ == "__main__":
    main()
