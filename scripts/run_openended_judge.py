import json
import os
import pandas as pd
import tqdm
from openai import OpenAI
from together import Together
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a medical expert judge evaluating whether a model's answer is correct. "
    "You will be given a medical question, the correct answer, and the model's answer. "
    "Respond with only 1 if the model's answer conveys the same medical meaning as the correct answer, "
    "or 0 if it does not. No explanation, no other text."
)

USER_TEMPLATE = """\
Question: {question}

Correct answer: {correct_answer}

Model's answer: {llm_response}

Does the model's answer convey the same medical meaning as the correct answer? Answer 1 or 0."""


def build_client(provider: str, local_base_url: str):
    provider = provider.lower()
    if provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), provider
    elif provider == "together":
        return Together(api_key=os.getenv("TOGETHER_API_KEY")), provider
    elif provider == "local":
        return OpenAI(base_url=local_base_url, api_key="none"), provider
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_judge(client, provider: str, model: str, question: str, correct_answer: str, llm_response: str) -> int:
    prompt = USER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        llm_response=llm_response,
    )
    extra = {}
    if provider == "openai" and model in ("gpt-5", "gpt-5-mini"):
        extra["reasoning_effort"] = "minimal"
    else:
        extra["temperature"] = 0

    response = client.chat.completions.create(
        model=model,
        **extra,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("1"):
        return 1
    elif raw.startswith("0"):
        return 0
    else:
        print(f"  [WARN] Unexpected judge response: {raw!r} — recording as NaN")
        return None


def get_correct_answer_text(row: pd.Series) -> str:
    try:
        options = json.loads(row["answer_options"])
        letter = str(row["correct_choice"]).strip().upper()
        return options.get(letter, letter)
    except Exception:
        return str(row["correct_choice"])


INPUT_CSV = "results/results_llama4_openended_reranked_rag.csv"
OUTPUT_CSV = None  # set to a path to write separately, or None to overwrite INPUT_CSV
MODEL = "llama4"
PROVIDER = "local"  # openai | together | local
LOCAL_BASE_URL = "http://localhost:8080/v1"


CHECKPOINT_EVERY = 20  # save to disk every N rows


def main():
    output_path = OUTPUT_CSV or INPUT_CSV
    client, provider = build_client(PROVIDER, LOCAL_BASE_URL)

    df = pd.read_csv(INPUT_CSV)

    if "llm_correct" not in df.columns:
        df["llm_correct"] = None

    todo_mask = df["llm_correct"].isna()
    n_done = (~todo_mask).sum()
    n_todo = todo_mask.sum()
    print(f"Loaded {len(df)} rows — {n_done} already judged, {n_todo} remaining.")

    n_since_checkpoint = 0
    for idx in tqdm.tqdm(df[todo_mask].index, desc="Judging answers"):
        row = df.loc[idx]
        llm_response = str(row.get("llm_response", "")).strip()
        if not llm_response or llm_response.lower() in ("nan", "none", ""):
            df.at[idx, "llm_correct"] = 0
        else:
            correct_answer = get_correct_answer_text(row)
            try:
                verdict = call_judge(client, provider, MODEL, row["question"], correct_answer, llm_response)
                df.at[idx, "llm_correct"] = verdict
            except Exception as e:
                print(f"\n  [ERROR] row {idx}: {e}")
                continue

        n_since_checkpoint += 1
        if n_since_checkpoint % CHECKPOINT_EVERY == 0:
            df.to_csv(output_path, index=False)
            print(f"  [checkpoint] {n_done + n_since_checkpoint} rows saved to {output_path}")

    df.to_csv(output_path, index=False)

    judged = df["llm_correct"].notna()
    accuracy = df.loc[judged, "llm_correct"].mean()
    print(f"\nSaved to {output_path}")
    print(f"Judged: {judged.sum()} / {len(df)}  |  Accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
