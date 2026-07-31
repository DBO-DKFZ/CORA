import json
import os
import argparse
import yaml
import tqdm
import chromadb
from openai import OpenAI
from together import Together
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


REQUIRED_KEYS = ("input_json", "output_dir", "verbose", "embed_model", "chroma_path", "collection_name")

REWRITE_PROMPT = (
    "You are a query reformulation expert. "
    "Rewrite medical questions to optimize retrieval from knowledge bases. "
    "Make queries more specific, add medical terminology, and focus on key concepts."
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    return cfg


class NaiveRetriever:
    """
    Single-pass retrieval pipeline: reformulate the query once, then retrieve
    top_k documents from each enabled collection.  Outputs the same JSON format
    as run_agentic_retrieval.py so run_reranker.py and run_answerer.py work
    without modification.
    """

    def __init__(
        self,
        input_file: str,
        output_dir: str,
        chroma_path: str,
        collection_name: str,
        embed_model,
        books_collection_name: str = "books",
        top_k: int = 15,
        verbose: bool = False,
        reformulation_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        provider: str = "together",
        local_base_url: str = "http://localhost:8000/v1",
    ):
        load_dotenv()
        self.input_file = input_file
        self.output_dir = output_dir
        self.embed_model = embed_model
        self.top_k = top_k
        self.verbose = verbose
        self.reformulation_model = reformulation_model

        os.makedirs(output_dir, exist_ok=True)

        provider = provider.lower()
        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "together":
            self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        elif provider == "local":
            self.client = OpenAI(base_url=local_base_url, api_key="none")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = chroma_client.get_collection(name=collection_name)
        self.books_collection = chroma_client.get_collection(name=books_collection_name)

    def _reformulate_query(self, question: str) -> str:
        prompt = (
            f"Reformulate this medical question to optimize retrieval from a dermatology knowledge base.\n\n"
            f"QUESTION: {question}\n\n"
            f"Return ONLY the reformulated query, no explanation."
        )
        response = self.client.chat.completions.create(
            model=self.reformulation_model,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _retrieve_from_collection(self, query: str, collection) -> list[str]:
        embedding = self.embed_model.get_text_embedding(query)
        results = collection.query(query_embeddings=[embedding], n_results=self.top_k)
        return results["documents"][0] if results.get("documents") else []

    def _retrieve_all(self, query: str) -> list[str]:
        docs = (
            self._retrieve_from_collection(query, self.collection)
            + self._retrieve_from_collection(query, self.books_collection)
        )
        seen = set()
        return [d for d in docs if not (d in seen or seen.add(d))]

    def format_question(self, q_data: dict) -> str:
        prompt = f"{q_data['question']}\n\n"
        for k, v in q_data["answer_options"].items():
            prompt += f"{k}. {v}\n"
        return prompt

    def naive_retrieval(self, question: str, question_id: str = "") -> dict:
        prefix = f"[{question_id}] " if question_id else ""
        if self.verbose:
            print(f"\n{'='*60}\n{prefix}QUESTION: {question[:100]}...")

        reformulated = self._reformulate_query(question)
        if self.verbose:
            print(f"{prefix}[REFORMULATE] {reformulated}")

        documents = self._retrieve_all(reformulated)
        if self.verbose:
            print(f"{prefix}[RETRIEVE] {len(documents)} docs retrieved")

        return {
            "plan": {
                "collections": ["primary", "books"],
                "decomposed": False,
                "sub_queries": [],
                "key_concepts": [],
                "reasoning": "Naive RAG: single reformulation pass",
            },
            "conditions": [],
            "retrieval_history": [{
                "iteration": 1,
                "query": reformulated,
                "docs_retrieved": len(documents),
                "total_docs": len(documents),
                "critique": None,
            }],
            "documents": documents,
            "final_sufficient": True,
        }

    def process(self):
        with open(self.input_file, "r") as f:
            questions = json.load(f)["questions"]

        already_done = {os.path.splitext(f)[0] for f in os.listdir(self.output_dir) if f.endswith(".json")}

        if self.verbose:
            print(f"Loaded {len(questions)} questions from {self.input_file}")
            if already_done:
                print(f"Skipping {len(already_done)} already-processed questions.\n")

        todo = {q_id: q_data for q_id, q_data in questions.items() if q_id not in already_done}

        for q_id, q_data in tqdm.tqdm(todo.items(), desc="Naive retrieval"):
            retrieval_result = self.naive_retrieval(q_data["question"], question_id=q_id)

            record = {
                "question_id":     q_id,
                "question":        q_data["question"],
                "question_prompt": self.format_question(q_data),
                "answer_options":  q_data["answer_options"],
                "correct_choice":  q_data["correct_choice"][0],
                "answer_source":   "naive_rag",
                "retrieval_result": retrieval_result,
            }

            with open(os.path.join(self.output_dir, f"{q_id}.json"), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(todo)} records → {self.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)

    retriever = NaiveRetriever(
        input_file=cfg["input_json"],
        output_dir=cfg["output_dir"],
        chroma_path=cfg["chroma_path"],
        collection_name=cfg["collection_name"],
        embed_model=HuggingFaceEmbedding(model_name=cfg["embed_model"]),
        books_collection_name=cfg.get("books_collection_name", "books"),
        top_k=cfg.get("topk", 15),
        verbose=cfg["verbose"],
        reformulation_model=cfg.get("reformulation_model", "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"),
        provider=cfg.get("provider", "together"),
        local_base_url=cfg.get("local_base_url", "http://localhost:8000/v1"),
    )
    retriever.process()
