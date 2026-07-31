#!/usr/bin/env python3
"""
Ingest PMC case reports into the retrieval corpus, with leakage exclusion.

The PubMed vignettes were GENERATED FROM case reports, so any report whose
PMID/PMCID is a vignette `source_pmid`/`source_pmcid` must NOT enter the corpus —
otherwise RAG retrieves the source article, which states the gold answer verbatim
(trivial leakage, inflated results).

This matches the *live* retrieval corpus, not the stale build_index.py:
  - chroma path   : chromadb_snowflakev2   (NOT ./chromadb)
  - embed model   : Snowflake/snowflake-arctic-embed-l-v2.0
  - new collection: case_reports           (queried alongside eadv_guidelines/books)

Modes:
  --dry-run (default): report leakage exclusion + chunk counts, and write the
                       chunk texts to results/case_report_chunks.jsonl so
                       analyze_corpus_coverage.py can measure the coverage lift
                       WITHOUT a GPU embedding pass.
  --embed            : embed the leakage-free chunks and add them to the
                       `case_reports` collection in chromadb_snowflakev2.

Usage:
  python ingest_case_reports.py --input pubmed/pmc_oa_case_reports.jsonl --dry-run
  python ingest_case_reports.py --input pubmed/pmc_oa_case_reports_fresh.jsonl --embed
"""

import os
import json
import argparse

CHROMA_PATH = "chromadb_snowflakev2"
COLLECTION = "case_reports"
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
VIGNETTES = "pubmed/vignettes.jsonl"
CHUNKS_OUT = "results/case_report_chunks.jsonl"


def norm_id(x) -> str:
    return str(x).strip().replace("PMC", "") if x else ""


def leakage_ids(vignettes_path: str) -> set:
    ids = set()
    with open(vignettes_path) as f:
        for line in f:
            v = json.loads(line)
            for k in ("source_pmid", "source_pmcid"):
                nid = norm_id(v.get(k))
                if nid:
                    ids.add(nid)
    return ids


def load_reports(path: str, leak: set):
    kept, excluded, no_body = [], 0, 0
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            if norm_id(c.get("pmid")) in leak or norm_id(c.get("pmcid")) in leak:
                excluded += 1
                continue
            if not c.get("body"):
                no_body += 1
                continue
            kept.append(c)
    return kept, excluded, no_body


def to_documents(reports: list):
    """Build llama_index Documents; body already carries '## <title>' headers."""
    from llama_index.core import Document
    docs = []
    for c in reports:
        text = f"# {c.get('title','')}\n\n"
        if c.get("abstract"):
            text += f"## Abstract\n\n{c['abstract']}\n\n"
        text += c.get("body", "")
        docs.append(Document(
            text=text,
            metadata={
                "source": "case_report",
                "pmid": norm_id(c.get("pmid")),
                "pmcid": norm_id(c.get("pmcid")),
                "title": c.get("title", ""),
                "journal": c.get("journal", ""),
                "year": c.get("year", ""),
            },
            excluded_embed_metadata_keys=["pmid", "pmcid", "year"],
            excluded_llm_metadata_keys=["pmid", "pmcid", "year"],
        ))
    return docs


def chunk(documents: list):
    from llama_index.core.node_parser import MarkdownNodeParser
    nodes = MarkdownNodeParser().get_nodes_from_documents(documents)
    return nodes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="pubmed/pmc_oa_case_reports.jsonl")
    ap.add_argument("--vignettes", default=VIGNETTES)
    ap.add_argument("--chroma", default=CHROMA_PATH)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--embed", action="store_true",
                    help="Actually embed + write to Chroma (default is dry-run).")
    ap.add_argument("--rebuild", action="store_true",
                    help="Drop the collection before adding.")
    args = ap.parse_args()

    leak = leakage_ids(args.vignettes)
    reports, excluded, no_body = load_reports(args.input, leak)
    print(f"Leakage IDs from vignettes:      {len(leak)}")
    print(f"Excluded as leakage (source):    {excluded}")
    print(f"Dropped (no body text):          {no_body}")
    print(f"Leakage-free reports to ingest:  {len(reports)}")

    if not reports:
        print("\nNothing to ingest. The input file appears to BE the vignette source set.")
        print("Fetch an INDEPENDENT set first, e.g.:")
        print("  python pubmed/fetch_pmc_oa_case_reports.py --exclude-pmids pubmed/vignettes.jsonl \\")
        print("      --query '\"Skin Diseases\"[majr] AND \"Case Reports\"[pt] AND hasabstract "
              "AND \"2020\"[dp]:\"2024\"[dp] AND \"Humans\"[mesh] AND \"pubmed pmc open access\"[sb]' \\")
        print("      --max-results 3000 --output pubmed/pmc_oa_case_reports_fresh.jsonl")
        return

    docs = to_documents(reports)
    nodes = chunk(docs)
    print(f"Chunks after markdown splitting: {len(nodes)}")

    # Always emit chunk texts for the coverage-lift measurement (no GPU needed).
    os.makedirs(os.path.dirname(CHUNKS_OUT), exist_ok=True)
    with open(CHUNKS_OUT, "w") as f:
        for n in nodes:
            f.write(json.dumps({"text": n.get_content()}) + "\n")
    print(f"Wrote chunk texts -> {CHUNKS_OUT}  (feed to analyze_corpus_coverage.py --extra-chunks)")

    if not args.embed:
        print("\n[dry-run] Skipped embedding. Re-run with --embed to write to Chroma.")
        return

    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device="cuda")
    client = chromadb.PersistentClient(path=args.chroma)
    if args.rebuild:
        try:
            client.delete_collection(args.collection)
            print(f"Dropped existing collection '{args.collection}'")
        except Exception:
            pass
    collection = client.get_or_create_collection(args.collection)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context,
                     embed_model=embed_model, show_progress=True)
    print(f"\nCollection '{args.collection}' now has {collection.count()} chunks in {args.chroma}")
    print("NOTE: wire this collection into the retriever (see run_agentic_retrieval.py "
          "books_collection_name handling) so RAG actually queries it.")


if __name__ == "__main__":
    main()
