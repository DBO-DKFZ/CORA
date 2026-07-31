import json
import os
import argparse
import yaml
import tqdm
from sentence_transformers import CrossEncoder


REQUIRED_KEYS = ("input_dir", "output_dir", "reranker_model")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    return cfg


class Reranker:
    """
    Reranks retrieved documents in agentic retrieval records using a cross-encoder.

    Reads per-question JSON files from input_dir (same format produced by
    run_agentic_retrieval.py), reranks each record's documents against the
    question, and writes updated records to output_dir.

    The output JSON is identical to the input except:
      - retrieval_result["documents"] is sorted by descending reranker score
      - retrieval_result["reranker_scores"] is added (parallel list of floats)
      - retrieval_result["final_doc_count"] is updated if top_k truncates
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        reranker_model: str,
        top_k: int = None,
        batch_size: int = 32,
        verbose: bool = False,
        sufficient_only: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.top_k = top_k
        self.batch_size = batch_size
        self.verbose = verbose
        self.sufficient_only = sufficient_only

        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print(f"Loading reranker: {reranker_model}")
        self.model = CrossEncoder(reranker_model)
        if verbose:
            print("Reranker loaded.\n")

    def _vprint(self, msg: str = "") -> None:
        if self.verbose:
            print(msg)

    def _rerank(self, question: str, documents: list[str]) -> tuple[list[str], list[float]]:
        """Return (reranked_docs, scores) sorted by descending score."""
        if not documents:
            return [], []

        pairs = [(question, doc) for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size).tolist()

        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        if self.top_k is not None:
            ranked = ranked[: self.top_k]

        reranked_scores, reranked_docs = zip(*ranked) if ranked else ([], [])
        return list(reranked_docs), list(reranked_scores)

    def process(self):
        filenames = sorted(f for f in os.listdir(self.input_dir) if f.endswith(".json"))
        already_done = {f for f in os.listdir(self.output_dir) if f.endswith(".json")}
        todo = [f for f in filenames if f not in already_done]

        if already_done:
            print(f"Resuming: {len(already_done)} already reranked, {len(todo)} remaining.")

        self._vprint(f"Input:  {self.input_dir}")
        self._vprint(f"Output: {self.output_dir}")
        self._vprint(f"top_k:  {self.top_k if self.top_k else 'all'}\n")

        skipped = 0
        if self.sufficient_only:
            filtered = []
            for filename in todo:
                with open(os.path.join(self.input_dir, filename), "r", encoding="utf-8") as f:
                    rec = json.load(f)
                if (rec.get("retrieval_result") or {}).get("final_sufficient"):
                    filtered.append(filename)
                else:
                    skipped += 1
            todo = filtered
            print(f"sufficient_only: {len(todo)} to rerank, {skipped} skipped (final_sufficient=False).")

        for filename in tqdm.tqdm(todo, desc="Reranking"):
            in_path = os.path.join(self.input_dir, filename)
            out_path = os.path.join(self.output_dir, filename)

            with open(in_path, "r", encoding="utf-8") as f:
                record = json.load(f)

            question = record["question"]
            retrieval_result = record.get("retrieval_result") or {}
            documents = retrieval_result.get("documents") or []

            self._vprint(f"[{record['question_id']}]  {len(documents)} docs → reranking...")

            reranked_docs, scores = self._rerank(question, documents)

            retrieval_result["documents"] = reranked_docs
            retrieval_result["reranker_scores"] = scores
            retrieval_result["final_doc_count"] = len(reranked_docs)
            record["retrieval_result"] = retrieval_result

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

        reranked = len(todo)
        print(f"\nReranked records saved to {self.output_dir}/")
        print(f"Reranked: {reranked}  |  Skipped (final_sufficient=False): {skipped}  |  Already done: {len(already_done)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)

    reranker = Reranker(
        input_dir=cfg["input_dir"],
        output_dir=cfg["output_dir"],
        reranker_model=cfg["reranker_model"],
        top_k=cfg.get("top_k"),
        batch_size=cfg.get("batch_size", 32),
        verbose=cfg.get("verbose", False),
        sufficient_only=cfg.get("sufficient_only", False),
    )
    reranker.process()
