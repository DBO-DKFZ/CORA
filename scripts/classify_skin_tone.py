"""
Classify skin tone information in each of the 2736 RAG questions using
Qwen/Qwen3-235B-A22B-Instruct-2507-tput via the Together API.

Output CSV columns:
    question_id, skin_tone_mentioned, skin_tone_label, skin_tone_relevant,
    fitzpatrick_mentioned, fitzpatrick_type, skin_tone_reasoning, raw_llm_response

The script is resumable: already-processed IDs in the output CSV are skipped.
Each row is written immediately so no progress is lost on interruption.

Usage:
    python classify_skin_tone.py [--workers N] [--output PATH]
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import tqdm
from dotenv import load_dotenv
from together import Together

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
INPUT_CSV   = Path("results/results_gpt5_rag.csv")   # any RAG file — same questions
OUTPUT_CSV  = Path("results/skin_tone_classifications.csv")
MODEL       = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
MAX_RETRIES = 4
RETRY_DELAY = 5   # seconds; doubles on each retry

FIELDNAMES = [
    "question_id",
    "skin_tone_mentioned",   # bool — any explicit skin/ethnicity mention?
    "skin_tone_label",       # dark | light | medium | unspecified | mixed
    "skin_tone_relevant",    # bool — is skin tone clinically relevant to this Q?
    "fitzpatrick_mentioned", # bool — Fitzpatrick type or phototype mentioned?
    "fitzpatrick_type",      # e.g. "I-II", "IV", "V-VI", or ""
    "skin_tone_reasoning",   # 1-2 sentence rationale
    "patient_sex",           # male | female | unspecified | mixed
    "patient_sex_relevant",  # bool — is sex clinically relevant to this Q?
    "raw_llm_response",
]

SYSTEM_PROMPT = """You are a dermatology expert. Analyse the clinical question and return a JSON object describing skin-tone and patient sex information.

Return ONLY this JSON — no other text, no markdown fences:
{
  "skin_tone_mentioned": <true|false>,
  "skin_tone_label": "<dark|light|medium|unspecified|mixed>",
  "skin_tone_relevant": <true|false>,
  "fitzpatrick_mentioned": <true|false>,
  "fitzpatrick_type": "<e.g. I-II, IV, V-VI, or empty string>",
  "skin_tone_reasoning": "<1-2 sentence explanation>",
  "patient_sex": "<male|female|unspecified|mixed>",
  "patient_sex_relevant": <true|false>
}

Definitions:

skin_tone_mentioned: true if the question explicitly mentions ANY of the following —
  ethnicity (African-American, Black, Caucasian, Asian, Hispanic, Latino, Middle-Eastern,
  Indian, Mediterranean), skin descriptors (dark skin, fair skin, light skin, olive skin,
  darker complexion, hyperpigmented baseline), or Fitzpatrick/phototype scale.

skin_tone_label:
  "dark"        — patient described as African-American, Black, dark-skinned, or South Asian
                  with dark skin, or Fitzpatrick IV-VI
  "light"       — Caucasian, white, fair-skinned, Northern European, or Fitzpatrick I-II
  "medium"      — Hispanic, Latino, Mediterranean, Middle-Eastern, East/Southeast Asian,
                  olive-skinned, or Fitzpatrick III
  "mixed"       — multiple skin tones mentioned or unclear from mixed ethnic background
  "unspecified" — no skin tone or ethnicity information given

skin_tone_relevant: true if the skin tone or ethnicity of the patient is CLINICALLY
  relevant to the correct answer — e.g. the disease has different presentations, prevalence,
  or treatment outcomes by skin type; or skin tone affects diagnosis (e.g. visualising
  erythema, detecting cyanosis, post-inflammatory hyperpigmentation, keloid risk, etc.).

fitzpatrick_mentioned: true only if the Fitzpatrick scale, phototype, or skin type number
  (I through VI) is explicitly used in the question text.

fitzpatrick_type: the type mentioned (e.g. "IV", "I-II"), or empty string if not mentioned.

patient_sex:
  "male"        — patient is identified as male (man, boy, he/his, male)
  "female"      — patient is identified as female (woman, girl, she/her, female)
  "mixed"       — question involves both sexes or sex is ambiguous from conflicting cues
  "unspecified" — no sex or gender information given

patient_sex_relevant: true if the patient's sex is CLINICALLY relevant to the correct
  answer — e.g. the condition has different prevalence, presentation, or treatment by sex;
  sex-specific anatomy is involved; hormonal factors, pregnancy, or sex-linked genetics
  affect the diagnosis or management."""

USER_TEMPLATE = "Question:\n{question}"


# ── helpers ───────────────────────────────────────────────────────────────────
def parse_response(raw: str) -> dict:
    """Extract JSON from the LLM response, stripping any stray markdown."""
    text = raw.strip()
    # strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # find first { ... } block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in: {raw[:200]}")
    return json.loads(text[start:end])


def call_llm(client: Together, question: str) -> tuple[dict, str]:
    """Call the model; return (parsed_dict, raw_text). Raises on failure."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(question=question)},
        ],
    )
    raw = response.choices[0].message.content
    parsed = parse_response(raw)
    return parsed, raw


def classify_with_retry(client: Together, qid: int, question: str) -> dict:
    """Classify one question, retrying on transient errors."""
    delay = RETRY_DELAY
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            parsed, raw = call_llm(client, question)
            return {
                "question_id":          qid,
                "skin_tone_mentioned":  parsed.get("skin_tone_mentioned", False),
                "skin_tone_label":      parsed.get("skin_tone_label", "unspecified"),
                "skin_tone_relevant":   parsed.get("skin_tone_relevant", False),
                "fitzpatrick_mentioned":parsed.get("fitzpatrick_mentioned", False),
                "fitzpatrick_type":     parsed.get("fitzpatrick_type", ""),
                "skin_tone_reasoning":  parsed.get("skin_tone_reasoning", ""),
                "patient_sex":          parsed.get("patient_sex", "unspecified"),
                "patient_sex_relevant": parsed.get("patient_sex_relevant", False),
                "raw_llm_response":     raw,
            }
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
    # all retries exhausted — write a sentinel row so we know it failed
    return {
        "question_id":          qid,
        "skin_tone_mentioned":  None,
        "skin_tone_label":      "ERROR",
        "skin_tone_relevant":   None,
        "fitzpatrick_mentioned":None,
        "fitzpatrick_type":     "",
        "skin_tone_reasoning":  f"ERROR: {last_err}",
        "patient_sex":          "ERROR",
        "patient_sex_relevant": None,
        "raw_llm_response":     "",
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main(workers: int):
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        sys.exit("TOGETHER_API_KEY not set in environment / .env")

    # load questions
    questions_df = pd.read_csv(INPUT_CSV)[["question_id", "question"]]
    total = len(questions_df)
    print(f"Loaded {total} questions from {INPUT_CSV}")

    # resume: read already-processed IDs
    done_ids: set[int] = set()
    if OUTPUT_CSV.exists():
        done_df = pd.read_csv(OUTPUT_CSV)
        done_ids = set(done_df["question_id"].tolist())
        print(f"Resuming — {len(done_ids)} already classified, "
              f"{total - len(done_ids)} remaining")

    todo = questions_df[~questions_df["question_id"].isin(done_ids)]
    if todo.empty:
        print("All questions already classified.")
        return

    # open output CSV in append mode
    write_header = not OUTPUT_CSV.exists() or len(done_ids) == 0
    out_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        out_file.flush()

    # one Together client per thread
    def make_client():
        return Together(api_key=api_key)

    errors = 0
    with tqdm.tqdm(total=len(todo), desc="Classifying", unit="q") as pbar:
        # Use a thread pool — Together API is I/O bound
        with ThreadPoolExecutor(max_workers=workers,
                                initializer=lambda: None) as pool:
            # submit all tasks; each thread gets its own client
            futures = {}
            _tls_clients: dict[int, Together] = {}

            def _task(row):
                import threading
                tid = threading.get_ident()
                if tid not in _tls_clients:
                    _tls_clients[tid] = make_client()
                return classify_with_retry(_tls_clients[tid],
                                           int(row["question_id"]),
                                           row["question"])

            fut_to_qid = {
                pool.submit(_task, row): int(row["question_id"])
                for _, row in todo.iterrows()
            }

            for fut in as_completed(fut_to_qid):
                result = fut.result()
                writer.writerow(result)
                out_file.flush()
                if result["skin_tone_label"] == "ERROR":
                    errors += 1
                pbar.update(1)

    out_file.close()
    print(f"\nDone. Results saved to {OUTPUT_CSV}")
    print(f"Errors: {errors} / {len(todo)}")

    # quick summary
    df = pd.read_csv(OUTPUT_CSV)
    df = df[df["skin_tone_label"] != "ERROR"]
    print("\n--- Classification summary ---")
    print(f"skin_tone_label:\n{df['skin_tone_label'].value_counts().to_string()}")
    print(f"\nskin_tone_mentioned (True):  {df['skin_tone_mentioned'].sum()} / {len(df)}")
    print(f"skin_tone_relevant  (True):  {df['skin_tone_relevant'].sum()} / {len(df)}")
    print(f"fitzpatrick_mentioned (True):{df['fitzpatrick_mentioned'].sum()} / {len(df)}")
    print(f"\npatient_sex:\n{df['patient_sex'].value_counts().to_string()}")
    print(f"\npatient_sex_relevant (True): {df['patient_sex_relevant'].sum()} / {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify skin tone in RAG questions")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel API threads (default: 16)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output CSV path")
    args = parser.parse_args()

    if args.output:
        OUTPUT_CSV = Path(args.output)

    main(args.workers)
