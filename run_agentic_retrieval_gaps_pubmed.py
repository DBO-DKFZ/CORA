import json
import os
import re
import time
import argparse
import yaml
import random
import tqdm
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from together import Together
from dotenv import load_dotenv
from typing import List, Dict
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


REQUIRED_KEYS = ("input_json", "output_dir", "verbose", "embed_model", "chroma_path", "case_reports_collection_name")


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    return cfg


PLANNING_PROMPT = (
    "You are an intelligent retrieval planning agent for medical questions. "
    "Your job is to analyze the question and decide the best retrieval strategy. "
    "Think step-by-step about what information is needed."
)
REWRITE_PROMPT = (
    "You are a query reformulation expert. "
    "Rewrite medical questions to optimize retrieval from knowledge bases. "
    "Make queries more specific, add medical terminology, and focus on key concepts."
)
CRITIQUE_PROMPT = (
    "You are a critical evaluator of retrieved information. "
    "Assess whether the retrieved documents contain sufficient information "
    "to answer the question accurately. Be strict in your evaluation."
)


class AgenticRetriever:
    def __init__(
        self,
        input_file: str,
        output_dir: str,
        chroma_path: str,
        case_reports_collection_name: str,
        embed_model,
        top_k: int = 3,
        verbose: bool = False,
        max_iterations: int = 3,
        enable_query_decomposition: bool = True,
        enable_self_critique: bool = True,
        agent_model: str = "openai/gpt-oss-120b",
        confidence_threshold: float = 0.7,
        provider: str = "together",
        local_base_url: str = "http://localhost:8000/v1",
        rerun_fallbacks: bool = False,
        keep_question_types: list = None,
    ):
        load_dotenv()
        self.input_file = input_file
        self.output_dir = output_dir
        self.embed_model = embed_model
        self.top_k = top_k
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.enable_query_decomposition = enable_query_decomposition
        self.enable_self_critique = enable_self_critique
        self.agent_model = agent_model
        self.confidence_threshold = confidence_threshold
        self.rerun_fallbacks = rerun_fallbacks
        self.keep_question_types = set(keep_question_types) if keep_question_types else None

        os.makedirs(output_dir, exist_ok=True)
        provider = provider.lower()
        if provider == "openai":
            self.agent_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "together":
            self.agent_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        elif provider == "local":
            self.agent_client = OpenAI(base_url=local_base_url, api_key="none")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = chroma_client.get_collection(name=case_reports_collection_name)
        if self.verbose:
            print(f"Collection: case_reports='{case_reports_collection_name}' "
                  f"({self.collection.count()} chunks)")

    # ------------------------------------------------------------------ #
    #  LLM calls                                                           #
    # ------------------------------------------------------------------ #

    def _call_agent(self, system_prompt: str, user_prompt: str) -> str:
        while True:
            try:
                response = self.agent_client.chat.completions.create(
                    model=self.agent_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content
            except Exception as e:
                if self.verbose:
                    print(f"[RETRY] 503 from LLM — retrying in 5s...")
                time.sleep(5)

    def _parse_json(self, response: str, fallback: dict) -> dict:
        text = re.sub(r'^```(?:json)?\s*', '', response.strip())
        text = re.sub(r'\s*```$', '', text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return fallback

    # ------------------------------------------------------------------ #
    #  Agentic pipeline steps                                              #
    # ------------------------------------------------------------------ #

    def plan_retrieval_strategy(self, question: str) -> dict:
        """Ask the LLM whether to decompose the question and what concepts to focus on."""
        prompt = f"""
        Analyze this medical question and plan the retrieval strategy.

        QUESTION: {question}

        Determine:
        1. Does this question require multiple sub-queries (e.g., definition + mechanism + treatment)?
        2. What are the key medical concepts/terms to focus on?

        Return a JSON object with:
        {{
            "requires_decomposition": true/false,
            "key_concepts": ["concept1", "concept2"],
            "reasoning": "1-2 line explanation"
        }}
        """
        return self._parse_json(self._call_agent(PLANNING_PROMPT, prompt), fallback={
            "requires_decomposition": False,
            "key_concepts": [],
            "reasoning": "Default fallback strategy",
        })

    def decompose_query(self, question: str) -> List[str]:
        """Break a complex question into 2-3 focused sub-queries for better retrieval."""
        prompt = f"""
        Break this complex medical question into 2-3 focused sub-queries for better retrieval.

        QUESTION: {question}

        Return a JSON array of sub-queries:
        ["sub-query 1", "sub-query 2", "sub-query 3"]

        If the question is already simple, return just the original question in an array.
        """
        response = self._call_agent(REWRITE_PROMPT, prompt)
        try:
            result = json.loads(response)
            return result if isinstance(result, list) else [question]
        except json.JSONDecodeError:
            return [question]

    def reformulate_query(self, query: str, iteration: int, gaps: List[str] = None) -> str:
        """Rewrite a query for a specific iteration, progressively broadening the search."""
        gaps_section = ""
        if gaps:
            gaps_list = "\n".join(f"  - {g}" for g in gaps)
            gaps_section = f"\nInformation gaps identified from previous retrieval:\n{gaps_list}\nPrioritize addressing these gaps in the reformulated query.\n"
        prompt = f"""
        This is retrieval iteration {iteration}. Reformulate this medical question for better retrieval.

        ORIGINAL QUERY: {query}
        {gaps_section}
        Iteration {iteration} strategy:
        - Iteration 1: Focus on core medical terminology and conditions
        - Iteration 2: Broaden to include related symptoms, mechanisms, or treatments
        - Iteration 3: Try alternative phrasings or related concepts

        Return ONLY the reformulated query, no explanation.
        """
        return self._call_agent(REWRITE_PROMPT, prompt)

    def critique_retrieval(self, question: str, documents: List[str], iteration: int) -> dict:
        """Evaluate whether the retrieved documents are sufficient to answer the question."""
        docs_text = "\n\n".join(f"Doc {i+1}: {doc[:500]}..." for i, doc in enumerate(documents))

        prompt = f"""
        Evaluate if the retrieved documents contain sufficient information to answer this question.

        QUESTION: {question}

        RETRIEVED DOCUMENTS (Iteration {iteration}):
        {docs_text}

        Provide a critical assessment:
        1. Are the documents relevant?
        2. Do they contain enough detail to answer accurately?
        3. Are there gaps in the information?
        4. Should we retrieve more documents or proceed to answer?

        Return JSON:
        {{
            "sufficient": true/false,
            "confidence": 0.0-1.0,
            "gaps": ["gap1", "gap2"],
            "recommendation": "answer" | "retrieve_more",
            "reasoning": "2-3 line explanation"
        }}
        """
        return self._parse_json(self._call_agent(CRITIQUE_PROMPT, prompt), fallback={
            "sufficient": False,
            "confidence": 0.5,
            "gaps": [],
            "recommendation": "retrieve_more",
            "reasoning": "Default assessment",
        })

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def retrieve_for_queries(self, queries: List[str]) -> List[str]:
        """Embed each query, pull top-k from the case-report collection, dedupe (order-preserving)."""
        docs = []
        for query in queries:
            embedding = self.embed_model.get_text_embedding(query)
            results = self.collection.query(query_embeddings=[embedding], n_results=self.top_k)
            if results.get("documents"):
                docs.extend(results["documents"][0])

        seen = set()
        return [d for d in docs if not (d in seen or seen.add(d))]

    # ------------------------------------------------------------------ #
    #  Main agentic loop                                                   #
    # ------------------------------------------------------------------ #

    def agentic_retrieval(self, question: str, question_id: str = "") -> dict:
        """
        Full agentic retrieval pipeline:
          1. Plan whether to decompose the question.
          2. Optionally decompose the question into sub-queries.
          3. Retrieve documents, then iteratively reformulate and re-retrieve
             until the critique passes or max iterations is reached.
        Returns a dict with all retrieved documents and metadata.
        """
        prefix = f"[{question_id}] " if question_id else ""
        if self.verbose:
            print(f"\n{'='*60}\n{prefix}QUESTION: {question[:100]}...\n{'='*60}")

        strategy = self.plan_retrieval_strategy(question)
        if self.verbose:
            print(f"{prefix}[PLAN] decompose={strategy['requires_decomposition']}")
            print(f"{prefix}[PLAN] reasoning: {strategy['reasoning']}")

        should_decompose = strategy.get("requires_decomposition") and self.enable_query_decomposition
        queries = self.decompose_query(question) if should_decompose else [question]
        if self.verbose:
            print(f"{prefix}[DECOMPOSE] {len(queries)} sub-queries: {queries}" if should_decompose
                  else f"{prefix}[DECOMPOSE] skipped — using original question")

        plan_metadata = {
            "decomposed": should_decompose,
            "sub_queries": queries if should_decompose else [],
            "key_concepts": strategy.get("key_concepts", []),
            "reasoning": strategy.get("reasoning", ""),
        }

        all_documents = []
        retrieval_history = []

        def run_retrieval_iteration(queries: List[str], iteration: int) -> bool:
            """
            Retrieve docs, optionally critique them, and log the result.
            Returns True if retrieval should stop (sufficient docs or nothing retrieved).
            """
            nonlocal all_documents

            new_docs = self.retrieve_for_queries(queries)
            all_documents = list(dict.fromkeys(all_documents + new_docs))  # merge + deduplicate

            if self.verbose:
                print(f"{prefix}  → new: {len(new_docs)} docs  |  total: {len(all_documents)} docs")

            critique = None
            should_stop = not new_docs  # stop if nothing was retrieved

            if self.enable_self_critique and new_docs:
                critique = self.critique_retrieval(question, all_documents, iteration)
                if self.verbose:
                    print(f"{prefix}[CRITIQUE] sufficient={critique['sufficient']}  "
                          f"confidence={critique['confidence']:.2f}  "
                          f"recommendation={critique['recommendation']}")
                    if critique.get("gaps"):
                        print(f"{prefix}[CRITIQUE] gaps: {critique['gaps']}")

                confident_enough = (critique["sufficient"]
                                    and critique["confidence"] > self.confidence_threshold)
                if confident_enough:
                    if self.verbose:
                        print(f"{prefix}[STOP] confidence {critique['confidence']:.2f} above threshold — stopping")
                    should_stop = True

            retrieval_history.append({
                "iteration": iteration,
                "query": queries if len(queries) > 1 else queries[0],
                "docs_retrieved": len(new_docs),
                "total_docs": len(all_documents),
                "critique": critique,
            })
            return should_stop

        # Initial retrieval pass
        if self.verbose:
            print(f"\n{prefix}--- Initial retrieval ---")
        sufficient = run_retrieval_iteration(queries, iteration=1)

        # Reformulation iterations (only if initial pass wasn't sufficient)
        if not sufficient:
            for i in range(1, self.max_iterations):
                if self.verbose:
                    print(f"\n{prefix}--- Reformulation iteration {i}/{self.max_iterations - 1} ---")

                last_gaps = (retrieval_history[-1].get("critique") or {}).get("gaps", [])
                reformulated = [self.reformulate_query(q, i, gaps=last_gaps) for q in queries]
                if self.verbose:
                    for orig, ref in zip(queries, reformulated):
                        if orig != ref:
                            print(f"{prefix}[REFORMULATE] {ref}")

                if run_retrieval_iteration(reformulated, iteration=i + 1):
                    break

        if self.verbose:
            print(f"\n{prefix}[DONE] {len(retrieval_history)} iteration(s) — {len(all_documents)} documents total")

        last_critique = (retrieval_history or [{}])[-1].get("critique") or {}
        final_sufficient = (
            bool(all_documents)
            and last_critique.get("sufficient", False)
            and last_critique.get("confidence", 0) > self.confidence_threshold
        )

        return {
            "plan":              plan_metadata,
            "conditions":        strategy.get("key_concepts", []),
            "retrieval_history": retrieval_history,
            "documents":         all_documents,
            "final_sufficient":  final_sufficient,
        }

    # ------------------------------------------------------------------ #
    #  Output helpers                                                      #
    # ------------------------------------------------------------------ #

    def format_question(self, q_data: dict) -> str:
        prompt = f"{q_data['question']}\n\n"
        for k, v in q_data["answer_options"].items():
            prompt += f"{k}. {v}\n"
        return prompt

    def _fallback_qids(self) -> set:
        """Return question IDs in output_dir that have at least one fallback critique."""
        qids = set()
        for fname in os.listdir(self.output_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(self.output_dir, fname), "r", encoding="utf-8") as f:
                record = json.load(f)
            history = record.get("retrieval_result", {}).get("retrieval_history", [])
            if any((s.get("critique") or {}).get("reasoning") == "Default assessment" for s in history):
                qids.add(os.path.splitext(fname)[0])
        return qids

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def _process_one(self, q_id: str, q_data: dict) -> dict:
        t0 = time.time()
        retrieval_result = self.agentic_retrieval(q_data["question"], question_id=q_id)
        elapsed = time.time() - t0

        if self.verbose:
            print(f"[RECORD] [{q_id}] sufficient={retrieval_result['final_sufficient']}  "
                  f"docs={len(retrieval_result['documents'])}  "
                  f"iterations={len(retrieval_result['retrieval_history'])}  "
                  f"time={elapsed:.1f}s")

        record = {
            "question_id":     q_id,
            "question":        q_data["question"],
            "question_prompt": self.format_question(q_data),
            "answer_options":  q_data["answer_options"],
            "correct_choice":  q_data["correct_choice"][0],
            "answer_source":   "agentic_rag" if retrieval_result["final_sufficient"] else "agentic_rag_insufficient",
            "retrieval_result": retrieval_result,
        }

        with open(os.path.join(self.output_dir, f"{q_id}.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        return record

    def process(self, max_workers: int = 4):
        with open(self.input_file, "r") as f:
            questions = json.load(f)["questions"]

        if self.keep_question_types is not None:
            n_before = len(questions)
            questions = {
                qid: q for qid, q in questions.items()
                if (q.get("metadata") or {}).get("question_type") in self.keep_question_types
            }
            if self.verbose:
                print(f"Question-type filter {sorted(self.keep_question_types)}: "
                      f"kept {len(questions)}/{n_before} questions.")

        already_done = {os.path.splitext(f)[0] for f in os.listdir(self.output_dir) if f.endswith(".json")}

        if self.rerun_fallbacks:
            fallback_qids = self._fallback_qids()
            already_done -= fallback_qids
            if self.verbose:
                print(f"rerun_fallbacks: {len(fallback_qids)} questions will be re-processed.")

        if self.verbose:
            print(f"Loaded {len(questions)} questions from {self.input_file}")
            print(f"Output directory: {self.output_dir}")
            if already_done:
                print(f"Skipping {len(already_done)} already-processed questions.\n")

        todo = {q_id: q_data for q_id, q_data in questions.items() if q_id not in already_done}

        new_records = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._process_one, q_id, q_data): q_id
                       for q_id, q_data in todo.items()}
            for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Agentic retrieval"):
                new_records.append(fut.result())

        print(f"\nSaved {len(new_records)} new records → {self.output_dir}/")
        if self.verbose and new_records:
            avg_iters = sum(len(r["retrieval_result"]["retrieval_history"]) for r in new_records) / len(new_records)
            avg_docs = sum(len(r["retrieval_result"]["documents"]) for r in new_records) / len(new_records)
            print(f"Average iterations:        {avg_iters:.2f}")
            print(f"Average documents retrieved: {avg_docs:.2f}")


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)

    retriever = AgenticRetriever(
        input_file=cfg["input_json"],
        output_dir=cfg["output_dir"],
        chroma_path=cfg["chroma_path"],
        case_reports_collection_name=cfg["case_reports_collection_name"],
        embed_model=HuggingFaceEmbedding(model_name=cfg["embed_model"]),
        top_k=cfg.get("topk", 3),
        verbose=cfg["verbose"],
        max_iterations=cfg.get("max_iterations", 3),
        enable_query_decomposition=cfg.get("enable_query_decomposition", True),
        enable_self_critique=cfg.get("enable_self_critique", True),
        agent_model=cfg.get("agent_model", "openai/gpt-oss-120b"),
        provider=cfg.get("provider", "together"),
        local_base_url=cfg.get("local_base_url", "http://localhost:8000/v1"),
        confidence_threshold=cfg.get("confidence_threshold", 0.7),
        rerun_fallbacks=cfg.get("rerun_fallbacks", False),
        keep_question_types=cfg.get("keep_question_types"),
    )
    retriever.process()