#!/usr/bin/env python3
"""
Build ChromaDB index from EADV guidelines and dermatology books.

Two collections are created in a single persistent ChromaDB:
  - eadv_guidelines : Data/EADV/{condition}/{document_title}/vlm/*.md
  - dermatology_books: Data/books/{book_title}/vlm/*.md

Run with REBUILD=True (default) to drop and recreate collections on each run.
"""

from pathlib import Path
from typing import Dict, List

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR       = Path("./Data")
CHROMA_DB_PATH = "./chromadb"
REBUILD        = True   # drop existing collections before indexing

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
EMBED_BATCH_SIZE = 2    # lower if GPU OOM; 4B model is memory-heavy

# Prepended to queries at retrieval time; documents are encoded without it.
# Adjust the task description to match your retrieval use-case.
QUERY_INSTRUCTION = (
    "Instruct: Retrieve relevant dermatological guidelines or textbook passages "
    "that answer the clinical question\nQuery: "
)

# ── Embedding model ─────────────────────────────────────────────────────────────

def get_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        query_instruction=QUERY_INSTRUCTION,
        trust_remote_code=True,
        device="cuda",
        embed_batch_size=EMBED_BATCH_SIZE,
    )

# ── Document loaders ────────────────────────────────────────────────────────────

def load_eadv_documents() -> List[Document]:
    """
    Load .md files from Data/EADV.
    Path pattern: EADV/{condition}/{document_title}/vlm/<file>.md
    Metadata attached: source, condition, document_title, file_path.
    """
    docs: List[Document] = []
    root = DATA_DIR / "EADV"

    for md_file in sorted(root.rglob("*.md")):
        parts = md_file.relative_to(root).parts
        # Expect exactly: (condition, document_title, "vlm", filename)
        if len(parts) != 4 or parts[2] != "vlm":
            continue

        condition, document_title = parts[0], parts[1]
        docs.append(Document(
            text=md_file.read_text(encoding="utf-8", errors="replace"),
            metadata={
                "source":         "eadv",
                "condition":      condition,
                "document_title": document_title,
                "file_path":      str(md_file),
            },
            # file_path is noisy in the embedded text; keep it for filtering only
            excluded_embed_metadata_keys=["file_path"],
            excluded_llm_metadata_keys=["file_path"],
        ))

    print(f"EADV   : {len(docs)} documents loaded")
    return docs


def load_books_documents() -> List[Document]:
    """
    Load .md files from Data/books.
    Path pattern: books/{book_title}/vlm/<file>.md
    Metadata attached: source, book_title, file_path.
    """
    docs: List[Document] = []
    root = DATA_DIR / "books"

    for md_file in sorted(root.rglob("*.md")):
        parts = md_file.relative_to(root).parts
        # Expect exactly: (book_title, "vlm", filename)
        if len(parts) != 3 or parts[1] != "vlm":
            continue

        book_title = parts[0]
        docs.append(Document(
            text=md_file.read_text(encoding="utf-8", errors="replace"),
            metadata={
                "source":     "book",
                "book_title": book_title,
                "file_path":  str(md_file),
            },
            excluded_embed_metadata_keys=["file_path"],
            excluded_llm_metadata_keys=["file_path"],
        ))

    print(f"Books  : {len(docs)} documents loaded")
    return docs

# ── Collection builder ──────────────────────────────────────────────────────────

def build_collection(
    documents: List[Document],
    collection_name: str,
    chroma_client: chromadb.PersistentClient,
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    if REBUILD:
        try:
            chroma_client.delete_collection(collection_name)
            print(f"  Dropped existing collection '{collection_name}'")
        except Exception:
            pass

    # Split each document on markdown headers; inherits document metadata
    nodes = MarkdownNodeParser().get_nodes_from_documents(documents)
    print(f"  {len(nodes)} nodes after markdown splitting")

    # Tag each node with its position within its source document
    doc_counters: Dict[str, int] = {}
    for node in nodes:
        fp = node.metadata.get("file_path", "")
        node.metadata["chunk_index"] = doc_counters.get(fp, 0)
        doc_counters[fp] = doc_counters.get(fp, 0) + 1

    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print(f"  Collection '{collection_name}' ready.\n")
    return index

# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> None:
    embed_model  = get_embed_model()
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    print("=== Dermatology books ===")
    build_collection(
        load_books_documents(),
        "dermatology_books",
        chroma_client,
        embed_model,
    )

    print("=== EADV guidelines ===")
    build_collection(
        load_eadv_documents(),
        "eadv_guidelines",
        chroma_client,
        embed_model,
    )

    print(f"Done. ChromaDB written to: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
