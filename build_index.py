#!/usr/bin/env python3
"""
Build the ChromaDB retrieval index from guideline and textbook markdown.

Two collections are created in a single persistent ChromaDB:
  - eadv_guidelines : {data_dir}/EADV/{condition}/{document_title}/vlm/*.md
  - books           : {data_dir}/books/{book_title}/vlm/*.md

Case reports are a third collection, added separately by ingest_case_reports.py.

The defaults reproduce the index used in the paper (Snowflake arctic-embed-l-v2.0
into ./chromadb_snowflakev2), which is also what the retrieval configs in configs/
point at. Every setting is overridable so the same script can build the small demo
index described in the README:

  python build_index.py --data-dir demo/corpus --chroma-path demo/chromadb_demo \\
                        --embed-model sentence-transformers/all-MiniLM-L6-v2 \\
                        --device cpu

IMPORTANT: --embed-model must match the `embed_model` in the retrieval config that
queries this index. Querying an index with a different embedding model silently
returns meaningless neighbours.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# ── Defaults (the paper's configuration) ───────────────────────────────────────

DEFAULT_DATA_DIR    = "./Data"
DEFAULT_CHROMA_PATH = "./chromadb_snowflakev2"
DEFAULT_EMBED_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
DEFAULT_BATCH_SIZE  = 2      # lower if the GPU OOMs; large embedders are memory-heavy

# ── Embedding model ─────────────────────────────────────────────────────────────

def get_embed_model(
    model_name: str,
    device: str,
    batch_size: int,
    query_instruction: str | None = None,
) -> HuggingFaceEmbedding:
    """
    Documents are always encoded without a prefix; `query_instruction` only affects
    query-time encoding, so leave it unset unless the retrieval side sets the same one.
    """
    return HuggingFaceEmbedding(
        model_name=model_name,
        query_instruction=query_instruction,
        trust_remote_code=True,
        device=device,
        embed_batch_size=batch_size,
    )

# ── Document loaders ────────────────────────────────────────────────────────────

def load_eadv_documents(data_dir: Path) -> List[Document]:
    """
    Load .md files from {data_dir}/EADV.
    Path pattern: EADV/{condition}/{document_title}/vlm/<file>.md
    Metadata attached: source, condition, document_title, file_path.
    """
    docs: List[Document] = []
    root = data_dir / "EADV"

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


def load_books_documents(data_dir: Path) -> List[Document]:
    """
    Load .md files from {data_dir}/books.
    Path pattern: books/{book_title}/vlm/<file>.md
    Metadata attached: source, book_title, file_path.
    """
    docs: List[Document] = []
    root = data_dir / "books"

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
    rebuild: bool = True,
) -> VectorStoreIndex | None:
    if rebuild:
        try:
            chroma_client.delete_collection(collection_name)
            print(f"  Dropped existing collection '{collection_name}'")
        except Exception:
            pass

    if not documents:
        # Retrieval scripts call get_collection() unconditionally, so the collection
        # must exist even when this source contributes nothing (e.g. a demo corpus
        # with no textbook content).
        chroma_client.get_or_create_collection(collection_name)
        print(f"  No documents found — created empty collection '{collection_name}'.\n")
        return None

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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help=f"Corpus root containing EADV/ and books/ (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH,
                   help=f"Persistent ChromaDB directory (default: {DEFAULT_CHROMA_PATH})")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                   help=f"HuggingFace embedding model (default: {DEFAULT_EMBED_MODEL})")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="Device for the embedding pass (default: cuda)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})")
    p.add_argument("--guidelines-collection", default="eadv_guidelines",
                   help="Collection name for EADV/ (default: eadv_guidelines)")
    p.add_argument("--books-collection", default="books",
                   help="Collection name for books/ (default: books)")
    p.add_argument("--query-instruction", default=None,
                   help="Optional query-side instruction prefix; leave unset to match "
                        "the retrieval scripts, which encode queries without a prefix.")
    p.add_argument("--no-rebuild", action="store_true",
                   help="Append to existing collections instead of dropping them first")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Corpus directory not found: {data_dir}")

    embed_model = get_embed_model(
        args.embed_model, args.device, args.batch_size, args.query_instruction
    )
    chroma_client = chromadb.PersistentClient(path=args.chroma_path)
    rebuild = not args.no_rebuild

    print("=== Dermatology books ===")
    build_collection(
        load_books_documents(data_dir),
        args.books_collection,
        chroma_client,
        embed_model,
        rebuild=rebuild,
    )

    print("=== EADV guidelines ===")
    build_collection(
        load_eadv_documents(data_dir),
        args.guidelines_collection,
        chroma_client,
        embed_model,
        rebuild=rebuild,
    )

    print(f"Done. ChromaDB written to: {args.chroma_path}")


if __name__ == "__main__":
    main()
