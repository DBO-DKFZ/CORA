"""
Counterfactual analysis of citation faithfulness.

For each RAG question we know which retrieved documents the LLM *cited*
(`used_sources`). We probe how the answer depends on the retrieved evidence by
replacing docs' text with off-topic real docs from other questions (verified
off-topic by the same cross-encoder that ranked them), keeping doc
count/positions/format identical and re-calling the LLM. One off-topic doc is
assigned per position, so every condition reuses the same content per slot and
the only thing differing across conditions is *which* positions get clobbered:

  - control:     re-call on the unmodified prompt (determinism sanity check).
  - treatment:   replace the CITED docs       -> are cited docs NECESSARY?
  - sufficiency: replace ALL NON-cited docs   -> are cited docs SUFFICIENT?
                  (i.e. keep only the cited docs intact)
  - allreplaced: replace ALL docs             -> does the answer depend on
                  retrieval at all, or is it parametric? (ceiling)

Usage:
    poetry run python analyze_counterfactual.py \
        --config configs/answerer_llama4_qwenagent_reranked_rag.yaml \
        --sample 200 --seed 0 \
        --out results/counterfactual_llama4_qwenagent.csv
"""

import argparse
import json
import os
import random
import re

import pandas as pd
import tqdm
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

# Must match run_answerer.AnswerGenerator.RAG_SYSTEM_PROMPT exactly.
RAG_SYSTEM_PROMPT = (
    "You are a medical expert assistant. "
    "You must answer based on the context from the guidelines provided by a database.\n"
    "You are provided with the following:\n"
    "- Multiple choice question\n"
    "- Database context knowledge\n\n"
    "Do NOT use information that is not relevant for answering the question.\n"
    "Ensure your answer is annotated with the Document IDs of the context that were used to answer the question.\n\n"
    "STRICT OUTPUT FORMAT — no other text allowed:\n"
    "Answer: 'X'\n"
    "Used sources: [Document ID N], [Document ID M], [Document ID O], ... [Document ID Z]\n\n"
    "If no sources are relevant:\n"
    "Answer: 'X'\n"
    "Used sources: None\n\n"
    "RULES:\n"
    "- Do NOT explain your reasoning\n"
    "- Do NOT add any text before or after the format"
    "- The answer MUST be only one of the provided answer choices."
)


def format_rag_prompt(question_prompt: str, documents: list) -> str:
    context = "\n\n".join(f"Document ID {i+1}:\n{doc}" for i, doc in enumerate(documents))
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question_prompt}"


def parse_rag_response(response: str) -> tuple[str, str]:
    answer, used_sources = "", ""
    for line in response.strip().splitlines():
        line = line.strip()
        if line.startswith("Answer:"):
            m = re.search(r"['\"]([A-Ja-j])['\"]", line)
            if m:
                answer = m.group(1).upper()
            else:
                rest = line[len("Answer:"):].strip()
                if rest:
                    answer = rest[0].upper()
        elif line.startswith("Used sources:"):
            used_sources = line[len("Used sources:"):].strip()
    return answer, used_sources


def cited_positions(used_sources: str, n_docs: int) -> list[int]:
    """Return 0-based positions of cited docs (Document ID is 1-based)."""
    if not isinstance(used_sources, str):
        return []
    ids = [int(x) for x in re.findall(r"Document ID\s*(\d+)", used_sources)]
    return sorted({i - 1 for i in ids if 1 <= i <= n_docs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results_csv", default=None,
                    help="defaults to the config's output_file")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reranker_model", default="mixedbread-ai/mxbai-rerank-large-v1",
                    help="cross-encoder used to verify replacements are off-topic; "
                         "must match the model that produced reranker_scores")
    ap.add_argument("--candidate_pool", type=int, default=64,
                    help="how many foreign docs to score per question when picking "
                         "off-topic replacements")
    ap.add_argument("--reranker_device", default="cuda:3",
                    help="device for the verification cross-encoder; pick a GPU the "
                         "LLM is not using (or 'cpu')")
    ap.add_argument("--baseline_csv", default="results/results_llama4_cleaned.csv",
                    help="no-retrieval (parametric) answers, joined on question_id")
    ap.add_argument("--stratum", default="all", choices=["all", "disagree", "agree"],
                    help="disagree = baseline answer != RAG answer (retrieval changed "
                         "the outcome); agree = same; all = no filter")
    args = ap.parse_args()

    load_dotenv()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_csv = args.results_csv or cfg["output_file"]
    docs_dir = cfg["retrieved_docs_dir"]
    model = cfg["model"]
    provider = cfg["provider"].lower()
    if provider == "local":
        client = OpenAI(base_url=cfg.get("local_base_url", "http://localhost:8000/v1"),
                        api_key="none")
    elif provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError(f"provider {provider} not wired up here")

    print(f"Loading reranker for off-topic verification: {args.reranker_model} "
          f"(device={args.reranker_device})")
    reranker = CrossEncoder(args.reranker_model, device=args.reranker_device)

    def call_llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content

    # --- load results, restrict to answerable counterfactual cases ---
    df = pd.read_csv(results_csv)
    df["orig_answer"] = df["llm_response"].astype(str).str.upper().str.strip()
    df["n_docs"] = df["retrieved_documents"].apply(
        lambda s: len(json.loads(s)) if isinstance(s, str) else 0)
    df["cited_pos"] = df.apply(
        lambda r: cited_positions(r["used_sources"], r["n_docs"]), axis=1)
    df["n_cited"] = df["cited_pos"].apply(len)

    # baseline (no-retrieval) answer per question, for stratification / reversion
    base = pd.read_csv(args.baseline_csv)
    base["baseline_answer"] = base["llm_response"].astype(str).str.upper().str.strip()
    base = base[base["baseline_answer"].str.match(r"^[A-E]$")]
    base_map = dict(zip(base["question_id"], base["baseline_answer"]))
    df["baseline_answer"] = df["question_id"].map(base_map)

    eligible = df[(df["n_cited"] >= 1)
                  & (df["n_docs"] >= 2)
                  & (df["orig_answer"].str.match(r"^[A-E]$"))].copy()
    print(f"{len(eligible)}/{len(df)} rows eligible (>=1 cited doc, valid original answer)")

    if args.stratum != "all":
        have_base = eligible["baseline_answer"].notna()
        differs = eligible["baseline_answer"] != eligible["orig_answer"]
        eligible = eligible[have_base & (differs if args.stratum == "disagree" else ~differs)].copy()
        print(f"stratum='{args.stratum}': {len(eligible)} questions "
              f"(baseline {'!=' if args.stratum == 'disagree' else '=='} RAG answer)")

    rng = random.Random(args.seed)
    n = min(args.sample, len(eligible))
    sample_ids = rng.sample(list(eligible["question_id"]), n)
    sample = eligible[eligible["question_id"].isin(sample_ids)].copy()
    print(f"Sampled {len(sample)} questions (seed={args.seed})")

    # --- global off-topic replacement pool: (qid, doc_text) over ALL rows ---
    pool = []
    for _, r in df.iterrows():
        try:
            for doc in json.loads(r["retrieved_documents"]):
                pool.append((r["question_id"], doc))
        except Exception:
            continue
    print(f"Replacement pool: {len(pool)} docs")

    # --- resume: keep rows already in the output file, skip those question_ids ---
    rows = []
    done_ids = set()
    if os.path.exists(args.out):
        try:
            prev = pd.read_csv(args.out)
            rows = prev.to_dict("records")
            done_ids = set(prev["question_id"].astype(str))
            print(f"Resuming: {len(done_ids)} already done, "
                  f"{len(sample) - len(sample[sample['question_id'].astype(str).isin(done_ids)])} remaining.")
        except (pd.errors.EmptyDataError, KeyError):
            pass

    for _, r in tqdm.tqdm(sample.iterrows(), total=len(sample), desc="Counterfactual"):
        qid = r["question_id"]
        if str(qid) in done_ids:
            continue
        with open(os.path.join(docs_dir, f"{qid}.json")) as f:
            rec = json.load(f)
        question_prompt = rec["question_prompt"]
        question = rec["question"]  # reranker scores against the raw question
        docs = list(rec["retrieval_result"]["documents"])
        kept_scores = rec["retrieval_result"].get("reranker_scores") or []
        n_docs = len(docs)

        cited = [p for p in r["cited_pos"] if p < n_docs]
        if not cited:
            continue
        k = len(cited)
        noncited = [p for p in range(n_docs) if p not in cited]

        # per-question deterministic RNG
        qrng = random.Random(f"{args.seed}-{qid}")

        # Off-topic threshold: a replacement must be LESS relevant to the
        # question than anything that was actually retrieved (the 10 kept docs
        # are the top-ranked, so their min sets the floor of "relevant").
        threshold = min(kept_scores) if kept_scores else None

        # Draw a candidate pool of foreign docs, score each against the raw
        # question with the same cross-encoder, and keep those below the
        # threshold (verified off-topic). We need ENOUGH to fill every position
        # (replace-all uses all n_docs), and we assign one off-topic doc PER
        # position so every condition reuses the same content per slot -- the
        # only thing that differs across conditions is which positions we clobber.
        foreign = [d for (pq, d) in pool if pq != qid]
        cand = qrng.sample(foreign, min(len(foreign), args.candidate_pool))
        cand_scores = reranker.predict([(question, d) for d in cand]).tolist()
        scored = sorted(zip(cand_scores, range(len(cand))), key=lambda x: x[0])  # most off-topic first

        if threshold is not None:
            passing = [(s, i) for s, i in scored if s < threshold]
        else:
            passing = scored
        if len(passing) >= n_docs:
            chosen = qrng.sample(passing, n_docs)  # representative off-topic docs
        else:
            # not enough below threshold: take all passing, fill from the most off-topic remainder
            remainder = [t for t in scored if t not in passing]
            chosen = passing + remainder[: n_docs - len(passing)]
        repl_by_pos = [cand[i] for _, i in chosen]          # one off-topic doc per position
        repl_scores = [s for s, _ in chosen]
        n_repl_above_threshold = sum(
            1 for s in repl_scores if threshold is not None and s >= threshold)

        # relevance of the cited docs we are clobbering (proof they were relevant)
        cited_doc_scores = [kept_scores[p] for p in cited if p < len(kept_scores)]

        def replace_positions(positions):
            """Copy of docs with the given positions overwritten by off-topic docs."""
            d = list(docs)
            for p in positions:
                d[p] = repl_by_pos[p]
            return d

        # treatment    : replace CITED docs        -> are cited docs NECESSARY?
        # sufficiency  : replace ALL non-cited docs -> are cited docs SUFFICIENT? (keep only cited)
        # allreplaced  : replace EVERYTHING         -> retrieval dependence vs parametric (ceiling)
        treat_docs = replace_positions(cited)
        suff_docs = replace_positions(noncited)
        all_docs = replace_positions(range(n_docs))

        ctrl_ans, ctrl_src = parse_rag_response(call_llm(format_rag_prompt(question_prompt, docs)))
        treat_ans, treat_src = parse_rag_response(call_llm(format_rag_prompt(question_prompt, treat_docs)))
        suff_ans, suff_src = parse_rag_response(call_llm(format_rag_prompt(question_prompt, suff_docs)))
        all_ans, all_src = parse_rag_response(call_llm(format_rag_prompt(question_prompt, all_docs)))

        rows.append({
            "question_id": qid,
            "correct_choice": str(r["correct_choice"]).upper(),
            "orig_answer": r["orig_answer"],
            "baseline_answer": r["baseline_answer"],
            "control_answer": ctrl_ans,
            "treatment_answer": treat_ans,
            "sufficiency_answer": suff_ans,
            "allreplaced_answer": all_ans,
            "n_cited": k,
            "n_noncited": len(noncited),
            "min_kept_score": threshold,
            "cited_doc_score_mean": (sum(cited_doc_scores) / len(cited_doc_scores)
                                     if cited_doc_scores else None),
            "repl_score_mean": sum(repl_scores) / len(repl_scores) if repl_scores else None,
            "repl_score_max": max(repl_scores) if repl_scores else None,
            "n_repl_above_threshold": n_repl_above_threshold,
            "control_changed": ctrl_ans != r["orig_answer"],
            "treatment_changed": treat_ans != r["orig_answer"],
            "sufficiency_changed": suff_ans != r["orig_answer"],
            "allreplaced_changed": all_ans != r["orig_answer"],
            # did removing docs revert the answer to the parametric (baseline) prior?
            "treatment_reverts_to_baseline": pd.notna(r["baseline_answer"]) and treat_ans == r["baseline_answer"],
            "sufficiency_reverts_to_baseline": pd.notna(r["baseline_answer"]) and suff_ans == r["baseline_answer"],
            "allreplaced_reverts_to_baseline": pd.notna(r["baseline_answer"]) and all_ans == r["baseline_answer"],
            "control_used_sources": ctrl_src,
            "treatment_used_sources": treat_src,
            "sufficiency_used_sources": suff_src,
            "allreplaced_used_sources": all_src,
        })

        if len(rows) % 20 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    print(f"\nSaved {len(out)} rows to {args.out}")
    if len(out):
        n = len(out)
        print(f"\nControl     (unmodified, determinism): "
              f"{out['control_changed'].sum()}/{n}  ({out['control_changed'].mean():.1%})  -- want ~0")
        print(f"Treatment   (replace CITED, necessity): "
              f"{out['treatment_changed'].sum()}/{n}  ({out['treatment_changed'].mean():.1%})")
        print(f"Sufficiency (replace all NON-cited):    "
              f"{out['sufficiency_changed'].sum()}/{n}  ({out['sufficiency_changed'].mean():.1%})  "
              f"-- low => cited docs alone are sufficient")
        print(f"Replace-all (replace ALL docs, ceiling):"
              f"{out['allreplaced_changed'].sum()}/{n}  ({out['allreplaced_changed'].mean():.1%})  "
              f"-- low => answer is parametric, not retrieval-driven")
        print("\n  GROUNDING SIGNATURE: treatment HIGH (cited necessary) "
              "+ sufficiency LOW (cited sufficient)")
        if out["baseline_answer"].notna().any():
            print("\n--- reversion to baseline (parametric) answer ---")
            print(f"Treatment   reverts to baseline: "
                  f"{out['treatment_reverts_to_baseline'].sum()}/{n}  "
                  f"({out['treatment_reverts_to_baseline'].mean():.1%})  "
                  f"-- on disagree stratum, HIGH => cited docs drove the RAG answer")
            print(f"Sufficiency reverts to baseline: "
                  f"{out['sufficiency_reverts_to_baseline'].sum()}/{n}  "
                  f"({out['sufficiency_reverts_to_baseline'].mean():.1%})  "
                  f"-- LOW => cited docs hold the RAG answer (don't fall back to prior)")
        print("\n--- off-topic verification ---")
        print(f"Cited docs replaced (relevance):  {out['cited_doc_score_mean'].mean():.3f} "
              f"± {out['cited_doc_score_mean'].std():.3f}")
        print(f"Replacement docs (relevance):     {out['repl_score_mean'].mean():.3f} "
              f"± {out['repl_score_mean'].std():.3f}")
        print(f"Replacements above their question's threshold: "
              f"{int(out['n_repl_above_threshold'].sum())} "
              f"(in {(out['n_repl_above_threshold'] > 0).sum()}/{len(out)} questions)")


if __name__ == "__main__":
    main()
