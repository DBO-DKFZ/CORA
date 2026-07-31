"""
Find questions where at least one used_source is from a clinical guideline,
determined by matching document texts against the ChromaDB 'eadv_guidelines'
collection (as opposed to the 'books' collection).
"""

import ast
import re
import sys
from pathlib import Path

import chromadb
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results" / "results_mistrallarge2_qwenagent_reranked_rag.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "guideline_sourced_questions.csv"
DEFAULT_CHROMA = REPO_ROOT / "chromadb_snowflakev2"
GUIDELINES_COLLECTION = "eadv_guidelines"
BOOKS_COLLECTION = "books"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text_fingerprint(text: str, length: int = 120) -> str:
    """Normalize whitespace and return first `length` lowercase chars as a fingerprint."""
    return re.sub(r"\s+", " ", text).strip()[:length].lower()


def build_fingerprint_set(collection) -> set[str]:
    """Fetch all documents from a ChromaDB collection and return their fingerprints."""
    docs = collection.get(include=["documents"])["documents"]
    return {text_fingerprint(t) for t in docs}


def parse_used_doc_ids(used_sources_str: str) -> list[int]:
    """Extract 1-based document IDs from '[Document ID 3], [Document ID 5]' style strings."""
    return [int(m) for m in re.findall(r"Document ID\s+(\d+)", str(used_sources_str))]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def process(input_csv: Path, output_csv: Path, chroma_path: Path) -> None:
    # Build fingerprint sets from ChromaDB
    print(f"Loading ChromaDB from {chroma_path} ...")
    client = chromadb.PersistentClient(path=str(chroma_path))
    guideline_fps = build_fingerprint_set(client.get_collection(GUIDELINES_COLLECTION))
    books_fps = build_fingerprint_set(client.get_collection(BOOKS_COLLECTION))
    print(f"  {GUIDELINES_COLLECTION}: {len(guideline_fps):,} chunks")
    print(f"  {BOOKS_COLLECTION}:      {len(books_fps):,} chunks")

    df = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df):,} rows from {input_csv.name}")

    results = []
    unknown_count = 0

    for _, row in df.iterrows():
        try:
            retrieved = ast.literal_eval(row["retrieved_documents"])
        except Exception:
            retrieved = []

        used_ids = parse_used_doc_ids(row["used_sources"])

        guideline_ids, book_ids, unknown_ids = [], [], []
        for doc_id in used_ids:
            idx = doc_id - 1  # 1-based → 0-based
            if not (0 <= idx < len(retrieved)):
                continue
            fp = text_fingerprint(retrieved[idx])
            if fp in guideline_fps:
                guideline_ids.append(doc_id)
            elif fp in books_fps:
                book_ids.append(doc_id)
            else:
                unknown_ids.append(doc_id)
                unknown_count += 1

        if guideline_ids:
            results.append({
                **row.to_dict(),
                "guideline_doc_ids": guideline_ids,
                "book_doc_ids": book_ids,
                "unknown_doc_ids": unknown_ids,
                "n_guideline_sources": len(guideline_ids),
                "n_book_sources": len(book_ids),
                "n_used_sources": len(used_ids),
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)

    print(f"\nFound {len(out_df):,} questions with ≥1 guideline source "
          f"({len(out_df) / len(df) * 100:.1f}% of total)")
    if unknown_count:
        print(f"  WARNING: {unknown_count} used-source docs could not be matched to either collection")
    print(f"Saved to {output_csv}")

    if not out_df.empty:
        print("\nBreakdown by number of guideline sources used:")
        print(out_df["n_guideline_sources"].value_counts().sort_index().to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_csv  = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    chroma_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CHROMA

    if not input_csv.exists():
        sys.exit(f"Input file not found: {input_csv}")
    if not chroma_path.exists():
        sys.exit(f"ChromaDB path not found: {chroma_path}")

    process(input_csv, output_csv, chroma_path)
