"""
Interactive terminal query tool for the dermrag system.

Runs a single free-text (open-ended) question through the full pipeline:
    agentic retrieval  →  (optional) cross-encoder rerank  →  LLM answer

Reuses the same classes as the batch pipeline:
    AgenticRetriever  (run_agentic_retrieval.py)
    CrossEncoder      (same model as run_reranker.py)
    AnswerGenerator   (run_answerer.py)  -- prompts + LLM call, open-ended mode

Usage:
    # one-off question
    python query.py "What is the first-line treatment for bullous pemphigoid?"

    # interactive REPL (no question given)
    python query.py

    # tweak knobs
    python query.py --top-k 5 --no-rerank "your question"
    python query.py --answer-model gpt-5 --answer-provider openai "your question"

All defaults mirror the production configs; override with flags or --config.
"""

import os
import re
import json
import argparse
import yaml
import tempfile
from datetime import datetime

from dotenv import load_dotenv

from run_agentic_retrieval import AgenticRetriever, set_seed
from run_answerer import AnswerGenerator


# Defaults mirror configs/agentic_retrieval.yaml, configs/reranker_mixedbread.yaml
# and configs/answerer_gpt5_qwenagent_reranked_rag.yaml
DEFAULTS = {
    # retrieval
    "chroma_path": "chromadb_snowflakev2",
    "collection_name": "eadv_guidelines",
    "books_collection_name": "books",
    "embed_model": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "agent_model": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "agent_provider": "together",
    "retrieval_top_k": 10,
    "max_iterations": 3,
    "confidence_threshold": 0.5,
    "enable_query_decomposition": True,
    "enable_self_critique": True,
    # rerank
    "rerank": True,
    "reranker_model": "mixedbread-ai/mxbai-rerank-large-v1",
    "rerank_top_k": 10,
    "rerank_batch_size": 16,
    # answer
    "answer_model": "gpt-5",
    "answer_provider": "openai",
    "local_base_url": "http://localhost:8000/v1",
}


def load_overrides(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class QueryEngine:
    """Lazy-loaded pipeline: heavy models load once, then answer many questions."""

    def __init__(self, cfg: dict, verbose: bool = True):
        load_dotenv()
        self.cfg = cfg
        self.verbose = verbose

        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self._log(f"Loading embedding model: {cfg['embed_model']}")
        embed_model = HuggingFaceEmbedding(model_name=cfg["embed_model"])

        # AgenticRetriever needs an output_dir (used only by its batch entrypoint).
        scratch_dir = tempfile.mkdtemp(prefix="dermrag_query_")
        self._log(f"Connecting to ChromaDB: {cfg['chroma_path']}")
        self.retriever = AgenticRetriever(
            input_file="",
            output_dir=scratch_dir,
            chroma_path=cfg["chroma_path"],
            collection_name=cfg["collection_name"],
            embed_model=embed_model,
            books_collection_name=cfg["books_collection_name"],
            top_k=cfg["retrieval_top_k"],
            verbose=self.verbose,
            max_iterations=cfg["max_iterations"],
            enable_query_decomposition=cfg["enable_query_decomposition"],
            enable_self_critique=cfg["enable_self_critique"],
            agent_model=cfg["agent_model"],
            provider=cfg["agent_provider"],
            local_base_url=cfg["local_base_url"],
            confidence_threshold=cfg["confidence_threshold"],
        )

        self.reranker = None
        if cfg["rerank"]:
            from sentence_transformers import CrossEncoder

            self._log(f"Loading reranker: {cfg['reranker_model']}")
            self.reranker = CrossEncoder(cfg["reranker_model"])

        # AnswerGenerator: reuse its open-ended RAG prompt + LLM call only.
        self.answerer = AnswerGenerator(
            output_csv="",
            model=cfg["answer_model"],
            provider=cfg["answer_provider"],
            rag_mode=True,
            retrieved_docs_dir=scratch_dir,
            local_base_url=cfg["local_base_url"],
            verbose=False,
            open_ended=True,
        )
        self._log("Ready.\n")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def _parse_open_ended(response: str) -> tuple[str, str]:
        """Capture the full (possibly multi-line) answer block and the sources line.

        Unlike AnswerGenerator._parse_open_ended_rag_response (line-based, made for
        single-letter MCQ answers), this keeps everything between the 'Answer:' and
        'Used sources:' markers so multi-line open-ended answers aren't truncated.
        """
        text = response.strip()
        lines = text.splitlines()

        answer_lines: list[str] = []
        used_sources = ""
        in_answer = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Used sources:"):
                used_sources = stripped[len("Used sources:"):].strip()
                in_answer = False
            elif stripped.startswith("Answer:"):
                answer_lines = [stripped[len("Answer:"):].strip()]
                in_answer = True
            elif in_answer:
                answer_lines.append(line)

        answer = "\n".join(answer_lines).strip()
        # Fall back to the raw response if the model didn't follow the format.
        if not answer:
            answer = text
        return answer, used_sources

    def _rerank(self, question: str, documents: list[str]) -> tuple[list[str], list[float]]:
        if not documents:
            return [], []
        pairs = [(question, doc) for doc in documents]
        scores = self.reranker.predict(
            pairs, batch_size=self.cfg["rerank_batch_size"]
        ).tolist()
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        top_k = self.cfg["rerank_top_k"]
        if top_k is not None:
            ranked = ranked[:top_k]
        scores_out, docs_out = zip(*ranked) if ranked else ([], [])
        return list(docs_out), list(scores_out)

    def answer(self, question: str) -> dict:
        # 1. Retrieve
        retrieval = self.retriever.agentic_retrieval(question)
        documents = retrieval["documents"]

        # 2. Rerank (optional)
        scores = None
        if self.reranker is not None and documents:
            documents, scores = self._rerank(question, documents)

        # 3. Answer
        if documents:
            prompt = self.answerer._format_rag_prompt(question, documents)
        else:
            prompt = question
        raw = self.answerer._call_llm(prompt)
        answer, used_sources = self._parse_open_ended(raw)

        return {
            "question": question,
            "answer": answer or raw.strip(),
            "used_sources": used_sources,
            "documents": documents,
            "reranker_scores": scores,
            "final_sufficient": retrieval.get("final_sufficient"),
            "raw": raw,
        }


def print_result(result: dict, show_docs: bool) -> None:
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])
    if result["used_sources"]:
        print(f"\nUsed sources: {result['used_sources']}")
    print(
        f"\nRetrieved {len(result['documents'])} document(s)  |  "
        f"retrieval sufficient: {result['final_sufficient']}"
    )

    if show_docs and result["documents"]:
        print("\n" + "-" * 70)
        print("RETRIEVED DOCUMENTS")
        print("-" * 70)
        scores = result["reranker_scores"]
        for i, doc in enumerate(result["documents"]):
            tag = f"[Document ID {i + 1}]"
            if scores is not None:
                tag += f"  (rerank score: {scores[i]:.3f})"
            print(f"\n{tag}\n{doc}")
    print()


def _used_documents(result: dict) -> list[dict]:
    """Return only the documents the model cited in 'Used sources', with their IDs/scores.

    Document IDs in the response are 1-based (see AnswerGenerator._format_rag_prompt).
    """
    documents = result["documents"]
    scores = result["reranker_scores"]
    cited_ids = sorted(set(int(n) for n in re.findall(r"Document ID\s*(\d+)", result["used_sources"] or "")))
    used = []
    for doc_id in cited_ids:
        idx = doc_id - 1
        if 0 <= idx < len(documents):
            used.append({
                "doc_id": doc_id,
                "score": scores[idx] if scores is not None else None,
                "text": documents[idx],
            })
    return used


def save_result(path: str, result: dict, cfg: dict) -> None:
    """Append one result to a file as a JSON line (one record per query)."""
    used_documents = _used_documents(result)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": result["question"],
        "answer": result["answer"],
        "used_sources": result["used_sources"],
        "final_sufficient": result["final_sufficient"],
        "used_doc_count": len(used_documents),
        "retrieved_doc_count": len(result["documents"]),
        "used_documents": used_documents,
        "raw": result["raw"],
        "models": {
            "agent_model": cfg["agent_model"],
            "answer_model": cfg["answer_model"],
            "reranked": cfg["rerank"],
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_cfg(args) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(load_overrides(args.config))
    # CLI flags take precedence over config file and defaults
    if args.retrieval_top_k is not None:
        cfg["retrieval_top_k"] = args.retrieval_top_k
    if args.top_k is not None:
        cfg["rerank_top_k"] = args.top_k
    if args.no_rerank:
        cfg["rerank"] = False
    if args.agent_model:
        cfg["agent_model"] = args.agent_model
    if args.agent_provider:
        cfg["agent_provider"] = args.agent_provider
    if args.answer_model:
        cfg["answer_model"] = args.answer_model
    if args.answer_provider:
        cfg["answer_provider"] = args.answer_provider
    return cfg


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Query the dermrag system from the terminal.")
    parser.add_argument("question", nargs="*", help="Question to ask. Omit for interactive mode.")
    parser.add_argument("--config", default=None, help="Optional YAML to override defaults.")
    parser.add_argument("--top-k", type=int, default=None, help="Docs to keep after reranking.")
    parser.add_argument("--retrieval-top-k", type=int, default=None, help="Docs per retrieval query.")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the reranking stage.")
    parser.add_argument("--agent-model", default=None, help="Retrieval/agent LLM.")
    parser.add_argument("--agent-provider", default=None, help="together | openai | local.")
    parser.add_argument("--answer-model", default=None, help="Answering LLM.")
    parser.add_argument("--answer-provider", default=None, help="together | openai | local.")
    parser.add_argument("--show-docs", action="store_true", help="Print full retrieved documents.")
    parser.add_argument("--quiet", action="store_true", help="Suppress pipeline progress logs.")
    parser.add_argument(
        "--save", "-o", metavar="PATH", default=None,
        help="Append each result to PATH as JSON lines (one record per query).",
    )
    args = parser.parse_args()

    cfg = build_cfg(args)
    engine = QueryEngine(cfg, verbose=not args.quiet)

    if args.question:
        question = " ".join(args.question)
        result = engine.answer(question)
        print_result(result, show_docs=args.show_docs)
        if args.save:
            save_result(args.save, result, cfg)
            print(f"Saved to {args.save}")
        return

    # Interactive REPL
    print("Interactive query mode. Type a question, or 'quit' / Ctrl-D to exit.")
    if args.save:
        print(f"Saving results to {args.save}")
    print()
    while True:
        try:
            question = input("query> ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        result = engine.answer(question)
        print_result(result, show_docs=args.show_docs)
        if args.save:
            save_result(args.save, result, cfg)


if __name__ == "__main__":
    main()
