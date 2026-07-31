"""
categorize_questions.py (Scipt 1/3)

Categorize dermatology benchmark questions along five axes using an LLM.

Output CSV columns:
    question_id, disease, disease_category, disease_prevalence, question_type,
    requires_visual_reasoning, raw_llm_response

The script is resumable: already-processed question IDs in the output CSV
are skipped. Each result is written immediately so progress is not lost.

Usage:
    python categorize_questions.py --config configs/categorize_questions.yaml
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

import tqdm

import yaml
import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from together import Together

REQUIRED_KEYS = ("model", "provider", "input_json", "output_csv")

SYSTEM_PROMPT = """You are a dermatology expert. For each multiple-choice question, return a JSON object categorizing it along five axes.

Do NOT include any analysis, explanation, or commentary. Output ONLY the JSON object — no markdown, no text before or after:

{
  "disease": "<specific disease or condition name, or 'general' if not disease-specific>",
  "disease_category": "<infectious|inflammatory|neoplastic|genetic_rare|hair_nail|pigmentation|metabolic_systemic|other>",
  "disease_prevalence": "<common|moderate|rare>",
  "question_type": "<diagnosis|treatment|mechanism|epidemiology|prognosis|other>",
  "requires_visual_reasoning": <true|false>
}

Definitions:

disease: the specific named condition (e.g. "acne vulgaris", "pemphigus vulgaris", "xeroderma pigmentosum").
  Use "general" only if the question is not about any specific disease.

disease_category (examples are illustrative, not exhaustive — use the best-matching category):
  "infectious"          — bacterial, viral, fungal, or parasitic skin infections (e.g. tinea,
                          impetigo, herpes zoster, scabies, candidiasis, onychomycosis,
                          leishmaniasis, etc.)
  "inflammatory"        — immune-mediated and inflammatory dermatoses: eczema, psoriasis,
                          autoimmune blistering diseases, vasculitis, drug reactions, acne,
                          rosacea, lupus, dermatomyositis, urticaria, contact dermatitis, etc.
  "neoplastic"          — benign and malignant skin tumours and lymphomas: melanoma, BCC, SCC,
                          mycosis fungoides, Kaposi sarcoma, actinic keratosis, nevi, DFSP, etc.
  "genetic_rare"        — inherited genodermatoses and rare syndromes with cutaneous
                          manifestations: epidermolysis bullosa, ichthyosis, xeroderma
                          pigmentosum, neurofibromatosis, Darier disease, Ehlers-Danlos, etc.
  "hair_nail"           — disorders primarily of hair or nail: alopecia areata, telogen
                          effluvium, androgenetic alopecia, nail dystrophies, trichorrhexis, etc.
  "pigmentation"        — disorders of melanin pigmentation: vitiligo, melasma, acanthosis
                          nigricans, post-inflammatory hyper/hypopigmentation, albinism, etc.
  "metabolic_systemic"  — cutaneous manifestations of systemic or metabolic disease: diabetic
                          dermopathy, pretibial myxedema, xanthomas, sarcoidosis, amyloidosis,
                          nutritional deficiency dermatoses (pellagra, scurvy), etc.
  "other"               — does not fit any of the above categories

disease_prevalence:
  "common"   — seen routinely in general dermatology practice: acne, psoriasis, eczema,
               atopic dermatitis, melanoma, BCC, SCC, rosacea, seborrheic dermatitis, tinea,
               urticaria, contact dermatitis, warts, molluscum, onychomycosis, scabies
  "moderate" — seen occasionally: pemphigus vulgaris, bullous pemphigoid, dermatomyositis,
               cutaneous lupus, pityriasis rosea, lichen planus, mycosis fungoides,
               Kaposi sarcoma, vitiligo, alopecia areata, hidradenitis suppurativa
  "rare"     — uncommon or orphan conditions: genodermatoses (epidermolysis bullosa,
               ichthyosis, xeroderma pigmentosum, Darier disease), rare vasculitides,
               rare tumors (Merkel cell carcinoma, DFSP), rare drug reactions (TEN, SJS)

question_type:
  "diagnosis"    — identifying a condition from clinical findings, histopathology, or workup
  "treatment"    — selecting therapy, drug dosage, management algorithm, or monitoring
  "mechanism"    — pathophysiology, drug mechanism, immune response, or disease biology
  "epidemiology" — incidence, prevalence, risk factors, demographics, genetics, associations
  "prognosis"    — outcomes, staging, survival, Breslow thickness, recurrence
  "other"        — does not fit the above

requires_visual_reasoning: true if (a) the question mentions an image, photograph, figure,
dermoscopy image, or histopathology slide being shown, OR (b) the question describes specific
lesion morphology (color, borders, surface texture, distribution) that must be interpreted to
answer correctly. False if the question is answerable from factual recall alone."""

FIELDNAMES = [
    "question_id",
    "disease",
    "disease_category",
    "disease_prevalence",
    "question_type",
    "requires_visual_reasoning",
    "raw_llm_response",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    return cfg


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", data)
    return [{"question_id": k, **v} for k, v in questions.items()]


def format_question(q: dict) -> str:
    text = q["question"] + "\n\n"
    for letter, option in q["answer_options"].items():
        text += f"{letter}. {option}\n"
    correct_letter = q.get("correct_choice", [])
    if isinstance(correct_letter, list):
        correct_letter = ", ".join(correct_letter)
    correct_text = q.get("correct_answer", "")
    if correct_letter or correct_text:
        label = f"{correct_letter}. {correct_text}" if correct_letter else correct_text
        text += f"\nCorrect answer: {label}"
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
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


VALID_DISEASE_CATEGORY  = {"infectious", "inflammatory", "neoplastic", "genetic_rare", "hair_nail", "pigmentation", "metabolic_systemic", "other"}
VALID_DISEASE_PREVALENCE = {"common", "moderate", "rare"}
VALID_QUESTION_TYPE      = {"diagnosis", "treatment", "mechanism", "epidemiology", "prognosis", "other"}


def make_client(provider: str):
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
            base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("LOCAL_API_KEY", "local-dev-key"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


def call_llm(client, model: str, provider: str, question_text: str) -> str:
    if provider.lower() == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question_text}],
        )
        if not response.content:
            raise ValueError(f"Empty content in Anthropic response: stop_reason={response.stop_reason}")
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
    if not response.choices:
        raise ValueError(f"Empty choices in response: finish_reason={getattr(response, 'finish_reason', None)}, model={response.model}")
    return response.choices[0].message.content


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "rate limit" in msg or "rate_limit" in msg or "429" in msg or "too many requests" in msg


def process_question(q: dict, client, model: str, provider: str, verbose: bool) -> tuple[dict, str] | None:
    """Call the LLM for one question and return (parsed_dict, raw_text), or None on unrecoverable failure."""
    question_text = format_question(q)
    raw = None

    for attempt in range(3):
        try:
            raw = call_llm(client, model, provider, question_text)
            text = raw.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = re.sub(r'^```(?:json)?\s*', '', text)
                text = re.sub(r'\s*```$', '', text)
            # Extract JSON object from anywhere in the response (handles leading analysis text)
            start, end = text.find('{'), text.rfind('}')
            if start != -1 and end != -1:
                text = text[start:end + 1]
            parsed = json.loads(text)
            # Validate and fall back to "other" for unrecognised enum values
            if parsed.get("disease_category") not in VALID_DISEASE_CATEGORY:
                parsed["disease_category"] = "other"
            if parsed.get("disease_prevalence") not in VALID_DISEASE_PREVALENCE:
                parsed["disease_prevalence"] = ""
            if parsed.get("question_type") not in VALID_QUESTION_TYPE:
                parsed["question_type"] = "other"
            return parsed, raw
        except Exception as e:
            rate_limited = _is_rate_limit(e)
            wait = 30 * (2 ** attempt) if rate_limited else 5 * (2 ** attempt)
            if verbose:
                tqdm.tqdm.write(f"  TRACEBACK:\n{traceback.format_exc()}")
                kind = "rate limit" if rate_limited else "parse error" if raw is not None else "error"
                tqdm.tqdm.write(f"  {kind} on {q['question_id']} (attempt {attempt + 1}): {e} — retrying in {wait}s")
                if raw is not None and not rate_limited:
                    tqdm.tqdm.write(f"    LLM output: {raw[:300]!r}")
            if attempt < 2:
                time.sleep(wait)

    # All retries exhausted — return None so the row is not written and will be retried next run
    if verbose:
        tqdm.tqdm.write(f"  Skipping {q['question_id']} after 3 failed attempts")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = cfg["model"]
    provider = cfg["provider"]
    output_path = Path(cfg["output_csv"])
    verbose = cfg.get("verbose", True)
    max_workers = cfg.get("max_workers", 2)

    questions = load_questions(cfg["input_json"])
    done_ids = load_done_ids(output_path)
    remaining = [q for q in questions if q["question_id"] not in done_ids]

    if verbose:
        print(f"Total: {len(questions)}  |  done: {len(done_ids)}  |  remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    client = make_client(provider)
    errors = 0
    bar = tqdm.tqdm(total=len(remaining), desc="Categorizing", unit="q")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(process_question, q, client, model, provider, verbose): q
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
                bar.update(1)
                bar.set_postfix(errors=errors)
                continue

            parsed, raw = result
            append_row(output_path, {
                "question_id": q["question_id"],
                "disease": parsed.get("disease", ""),
                "disease_category": parsed.get("disease_category", ""),
                "disease_prevalence": parsed.get("disease_prevalence", ""),
                "question_type": parsed.get("question_type", ""),
                "requires_visual_reasoning": parsed.get("requires_visual_reasoning", ""),
                "raw_llm_response": raw,
            })

            bar.update(1)
            bar.set_postfix(errors=errors)
            if verbose and parsed:
                tqdm.tqdm.write(
                    f"  [{q['question_id']}] {parsed.get('disease', '')} | "
                    f"{parsed.get('disease_category', '')} | "
                    f"{parsed.get('disease_prevalence', '')} | "
                    f"{parsed.get('question_type', '')} | "
                    f"visual={parsed.get('requires_visual_reasoning', '')}"
                )

    bar.close()
    print(f"\nDone. Output: {output_path}  |  errors: {errors}")


if __name__ == "__main__":
    main()
