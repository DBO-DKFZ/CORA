"""
LLM judge for open-ended answer correctness.

The answerer was run in open-ended mode (no options shown), so `llm_response` is
free text (e.g. "Melanoma", "Churg-Strauss Syndrome"). This judge decides whether
that free-text answer matches the ground-truth answer (the text of the correct
option), allowing for synonyms, abbreviations, and paraphrase.

For each case the judge returns one of:
    Correct | Partially correct | Incorrect

Usage:
    python run_answer_judge.py --config configs/answer_judge.yaml
    python run_answer_judge.py --input_csv results/results_llama4_case_reports_reranked_rag.csv \
                               --model openai/gpt-oss-120b --provider together

By default the verdict is written back into the input results CSV as new columns
(no separate file), resuming by skipping rows already judged:
    verdict, verdict_explanation, verdict_raw
Pass --output_csv (or set it in the config) to write the augmented table to a
different path instead of editing the input in place.
"""

import json
import os
import re
import argparse
import yaml
import pandas as pd
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from together import Together
from anthropic import Anthropic
from dotenv import load_dotenv


_VERDICT_RULES = """\
Decide how well the candidate answer matches the reference answer in clinical meaning,
not wording. Accept synonyms, abbreviations, eponyms, drug class vs. specific drug, and
paraphrase as long as the clinical intent is the same.

Use exactly one verdict:
- "Correct": the candidate conveys the same clinical answer as the reference (a synonym
  or equivalent phrasing counts as correct).
- "Partially correct": the candidate is in the right direction but less specific, names a
  broader category, omits a key qualifier, or hedges among several answers including the
  right one.
- "Incorrect": the candidate names a different diagnosis / step / treatment, or is wrong.

Respond with ONLY a JSON object matching this schema (no other text):
{
  "verdict": "<Correct | Partially correct | Incorrect>",
  "explanation": "<one sentence>"
}"""


def build_system_prompt(include_question: bool) -> str:
    """System prompt for the judge, worded for whichever inputs are actually shown."""
    if include_question:
        inputs = ("- A clinical question (diagnosis, next diagnostic step, or treatment)\n"
                  "- The REFERENCE ANSWER (the single correct answer)\n"
                  "- A CANDIDATE ANSWER written in free text by a model")
    else:
        inputs = ("- The REFERENCE ANSWER (the single correct answer to a clinical question)\n"
                  "- A CANDIDATE ANSWER written in free text by a model")
    return (
        "You are a board-certified dermatologist grading a free-text answer to a clinical question.\n\n"
        "You will be given:\n"
        f"{inputs}\n\n"
        + _VERDICT_RULES
    )


VERDICTS = ["Correct", "Partially correct", "Incorrect"]


def correct_answer_text(row: pd.Series) -> str:
    try:
        options = json.loads(row["answer_options"])
    except Exception:
        options = {}
    letter = str(row.get("correct_choice", "")).strip().upper()
    letter = letter[0] if letter else ""
    return options.get(letter, "")


def build_prompt(row: pd.Series, gold_text: str, include_question: bool = False) -> str:
    candidate = str(row.get("llm_response", "")).strip()
    q_block = f"QUESTION:\n{row['question']}\n\n" if include_question else ""
    return (
        f"{q_block}"
        f"REFERENCE ANSWER:\n{gold_text}\n\n"
        f"CANDIDATE ANSWER:\n{candidate}"
    )


def call_llm(client, model: str, provider: str, prompt: str, system_prompt: str) -> str:
    if provider == "anthropic":
        # Anthropic SDK: system prompt is a separate param, response is a list of
        # content blocks. effort "low" is the lowest reasoning-effort setting.
        # Passed via extra_body so it works regardless of SDK typing.
        response = client.messages.create(
            model=model,
            temperature=0,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"output_config": {"effort": "low"}},
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    extra = {}
    if provider == "openai" and "gpt-5" in model:
        extra["reasoning_effort"] = "minimal"
    else:
        extra["temperature"] = 0

    response = client.chat.completions.create(
        model=model,
        **extra,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def parse_response(raw: str) -> tuple[str, str]:
    """Returns (verdict, explanation)."""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(clean)
        verdict = parsed.get("verdict", "")
        explanation = parsed.get("explanation", "")
        if verdict not in VERDICTS:
            verdict = next((v for v in VERDICTS if v.lower() in verdict.lower()), verdict)
        return verdict, explanation
    except json.JSONDecodeError:
        return "", raw


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     help="YAML config file")
    parser.add_argument("--input_csv",  help="Results CSV to evaluate")
    parser.add_argument("--output_csv", help="Output CSV path")
    parser.add_argument("--model",      help="Judge model identifier")
    parser.add_argument("--provider",   choices=["openai", "together", "anthropic", "local"])
    parser.add_argument("--verbose",    action="store_true", default=None)
    parser.add_argument("--workers",    type=int, help="Parallel judge calls (default: 8)")
    parser.add_argument("--include_question", action=argparse.BooleanOptionalAction, default=None,
                        help="Show the clinical question to the judge (default: off — reference vs candidate only)")
    args = parser.parse_args()

    cfg: dict = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    def resolve(key, default=None):
        val = getattr(args, key, None)
        if val is not None and val is not False:
            return val
        return cfg.get(key, default)

    input_csv  = resolve("input_csv")
    output_csv = resolve("output_csv")   # optional; default = write verdict columns back into input_csv
    model      = resolve("model", "openai/gpt-oss-120b")
    provider   = resolve("provider", "together")
    base_url   = resolve("base_url")
    verbose    = bool(resolve("verbose", False))
    workers    = int(resolve("workers", 8))

    # Boolean flag resolved explicitly (resolve() can't distinguish an explicit False).
    if args.include_question is not None:
        include_question = args.include_question
    else:
        include_question = bool(cfg.get("include_question", False))

    if not input_csv:
        parser.error("Provide --input_csv (or --config with input_csv).")
    target_csv = output_csv or input_csv   # in-place augmentation unless an explicit output is given
    system_prompt = build_system_prompt(include_question)

    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "together":
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    elif provider == "anthropic":
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        client = OpenAI(
            base_url=base_url or os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1"),
            api_key=os.getenv("LOCAL_API_KEY", "local-dev-key"),
        )

    # Load the results CSV to augment in place (or the prior target if resuming there).
    src = target_csv if (target_csv != input_csv and os.path.exists(target_csv)) else input_csv
    df = pd.read_csv(src)
    df["question_id"] = df["question_id"].astype(str)

    # Verdict lives alongside the answerer's own columns — no separate file.
    for col in ("verdict", "verdict_explanation", "verdict_raw"):
        if col not in df.columns:
            df[col] = pd.NA

    unjudged = df["verdict"].isna() | df["verdict"].astype(str).str.strip().isin(("", "nan"))
    todo_idx = df.index[unjudged].tolist()
    print(f"Rows: {len(df)}  |  already judged: {len(df) - len(todo_idx)}  |  to judge: {len(todo_idx)}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"INPUT:  {input_csv}  ({len(df)} rows)")
        print(f"JUDGE:  {model}  [{provider}]")
        print(f"PROMPT: {'question + reference + candidate' if include_question else 'reference + candidate only'}")
        print(f"TARGET: {target_csv}  (columns: verdict, verdict_explanation, verdict_raw)")
        print(f"{'='*60}\n")

    def judge_one(idx):
        """Runs on a worker thread; does the (slow) LLM call only, no shared-state writes."""
        row = df.loc[idx]
        qid = row["question_id"]
        gold_text = correct_answer_text(row)

        if not gold_text:
            return idx, qid, None, None, None, gold_text, ("WARN", "no reference answer text; skipping")

        candidate = str(row.get("llm_response", "")).strip()
        if not candidate or candidate.lower() == "nan":
            # Model produced no answer — count as Incorrect without an LLM call.
            return idx, qid, "Incorrect", "Model returned an empty answer.", "", gold_text, None

        prompt = build_prompt(row, gold_text, include_question)
        try:
            raw = call_llm(client, model, provider, prompt, system_prompt)
        except Exception as e:
            return idx, qid, None, None, None, gold_text, ("ERROR", f"LLM call failed: {e}")
        verdict, explanation = parse_response(raw)
        return idx, qid, verdict, explanation, raw, gold_text, None

    judged = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(judge_one, idx) for idx in todo_idx]
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Judging answers"):
            idx, qid, verdict, explanation, raw, gold_text, err = fut.result()
            row = df.loc[idx]

            if err is not None:
                level, msg = err
                print(f"\n  [{level}] [{qid}] {msg}")
                continue

            df.at[idx, "verdict"] = verdict
            df.at[idx, "verdict_explanation"] = explanation
            df.at[idx, "verdict_raw"] = raw

            judged += 1
            if verbose:
                print(f"  [{qid}] {verdict:18s} | gold={gold_text[:45]!r} | resp={str(row.get('llm_response',''))[:45]!r}")
            if judged % 20 == 0:
                df.to_csv(target_csv, index=False)
                print(f"  [checkpoint] {judged} newly judged; {target_csv} updated")

    df.to_csv(target_csv, index=False)
    print(f"\nDone. Verdicts written into {target_csv} ({len(df)} rows).")

    # Summary stats over all judged rows
    scored = df[df["verdict"].notna() & ~df["verdict"].astype(str).str.strip().isin(("", "nan"))]
    print(f"\nVerdict breakdown:")
    print(scored["verdict"].value_counts().to_string())

    n = len(scored)
    if n:
        n_correct = (scored["verdict"] == "Correct").sum()
        n_partial = (scored["verdict"] == "Partially correct").sum()
        print(f"\nStrict accuracy  (Correct only):               {n_correct / n:.3f}  ({n_correct}/{n})")
        print(f"Lenient accuracy (Correct + Partially correct): {(n_correct + n_partial) / n:.3f}  ({n_correct + n_partial}/{n})")


if __name__ == "__main__":
    main()
