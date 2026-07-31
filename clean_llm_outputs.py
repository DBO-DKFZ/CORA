import os
import pandas as pd
from tqdm import tqdm
from together import Together
from dotenv import load_dotenv

load_dotenv()


class LLMExtractor:
    ANSWERING_SYSTEM_PROMPT = """You are extracting the selected multiple-choice answer from an explanation.

Task:
Return only the chosen answer letter from the text without any additional text or explanation.

Rules:
- Output exactly one uppercase letter: A, B, C, D, etc.
- Do not output any other words, punctuation, or explanation.
- Prefer the explicitly selected option if stated like "C", "Answer: C", "(C)", or "C)".
- If multiple letters appear, return the one identified as the final chosen/correct answer.
- If the answer is not clear or cannot be determined, return "None".
"""

    def __init__(self, model: str, provider: str = "together"):
        self.model = model
        self.provider = provider
        self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    def _call_llm(self, prompt: str) -> str:
        extra = {}
        if self.provider == "openai" and "gpt-5" in self.model:
            extra["reasoning_effort"] = "minimal"
        else:
            extra["temperature"] = 0

        response = self.client.chat.completions.create(
            model=self.model,
            **extra,
            messages=[
                {"role": "system", "content": self.ANSWERING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()


CHECKPOINT_EVERY = 20


def process_csv(input_path: str, output_path: str, model: str):
    df = pd.read_csv(input_path)

    if "llm_response" not in df.columns:
        raise ValueError("Column 'llm_response' not found in CSV")

    # Resume from existing output if present
    if os.path.exists(output_path):
        done_df = pd.read_csv(output_path)
        results = done_df["llm_response"].tolist()
        n_prior = sum(1 for r in results if str(r).upper().strip().rstrip(".") in {"A","B","C","D","E","F","G","H","I","J"})
        n_retry = sum(1 for r in results if pd.isna(r) or str(r).strip().lower() == "none")
        print(f"Resuming: {n_prior} already clean, {n_retry} failed rows will be retried.")
    else:
        results = [None] * len(df)

    extractor = LLMExtractor(model=model)

    n_already_clean = 0
    n_cleaned = 0
    n_failed = 0
    valid_letters = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}

    def normalize(val) -> str | None:
        s = str(val).upper().strip().rstrip(".")
        return s if s in valid_letters else None

    for idx, text in tqdm(enumerate(df["llm_response"]), total=len(df), desc="Cleaning responses"):
        # Skip rows that already have a valid letter (with or without trailing period)
        if normalize(results[idx]):
            results[idx] = normalize(results[idx])
            n_already_clean += 1
            continue

        text_str = str(text).strip()
        if normalize(text_str):
            results[idx] = normalize(text_str)
            n_already_clean += 1
        else:
            try:
                result = extractor._call_llm(text_str)
                if result not in valid_letters:
                    print(f"  [row {idx}] LLM returned invalid output '{result}' for: {text_str[:80]!r} → None")
                    result = None
                    n_failed += 1
                else:
                    print(f"  [row {idx}] Cleaned: {text_str[:80]!r} → {result}")
                    n_cleaned += 1
            except Exception as e:
                print(f"  [row {idx}] Error: {e}")
                result = None
                n_failed += 1
            results[idx] = result

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            df["llm_response"] = results
            df.to_csv(output_path, index=False)

    df["llm_response"] = results
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    print(f"  Already clean (single letter): {n_already_clean} / {len(df)}")
    print(f"  Cleaned by LLM:                {n_cleaned} / {len(df)}")
    print(f"  Failed / set to None:          {n_failed} / {len(df)}")


if __name__ == "__main__":

    MODEL = "gemma3_qwenagent_reranked_rag"

    INPUT_CSV = f"./results/results_{MODEL}.csv"
    OUTPUT_CSV = f"./results/results_{MODEL}_cleaned.csv"
    MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"

    process_csv(INPUT_CSV, OUTPUT_CSV, MODEL_NAME)
    