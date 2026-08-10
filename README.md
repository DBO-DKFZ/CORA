# CORA

An agentic retrieval-augmented generation (RAG) pipeline for dermatology question
answering, evaluated against physicians in a reader study.

The pipeline has five stages, each a standalone script driven by a YAML config:

```
corpus → ChromaDB index → agentic retrieval → cross-encoder rerank → answer generation → LLM judge
```

This README is written so a reviewer can install the code, run a self-contained demo
that needs none of the licensed study data, and then reproduce the manuscript's
quantitative results. Sections 1–4 follow the Nature Research code submission
checklist.

**Contents**

- [1. System requirements](#1-system-requirements)
- [2. Installation guide](#2-installation-guide)
- [3. Demo](#3-demo)
- [4. Instructions for use](#4-instructions-for-use)
- [Repository layout](#repository-layout)
- [Data availability](#data-availability)

---

## 1. System requirements

### Software dependencies

| Component | Version used | Notes |
|---|---|---|
| Python | 3.12.3 (`~3.12` required) | 3.13 is not supported by all pinned dependencies |
| Poetry | 2.1.4 | dependency manager; `pip` also works (see below) |
| PyTorch | 2.7.1 + CUDA 12.6 | CPU-only PyTorch is sufficient for the demo |
| transformers | 4.57.1 (pinned) | |
| sentence-transformers | 5.2.0 | cross-encoder reranking |
| chromadb | 1.4.1 | persistent vector store |
| llama-index | 0.14.13 | chunking, embedding, Chroma vector store |
| pandas | 2.3.3 | |
| openai | ≥1.100 | OpenAI-compatible clients (also used for local vLLM/TGI) |
| together | ≥1.5.35 | Together-hosted open-weight models |
| anthropic | 0.104.1 | LLM judge and annotation scripts — see the note in §2 |
| statsmodels, scikit-learn, nltk, sacrebleu, bert-score, rouge-score | see `pyproject.toml` | metrics and statistics |

The complete, exactly resolved dependency set is in [pyproject.toml](pyproject.toml)
and [poetry.lock](poetry.lock).

### Operating systems

- **Tested on:** Ubuntu 24.04.2 LTS (x86-64), Linux kernel 6.x/7.x, Python 3.12.3.
- Expected to work on any Linux distribution and on macOS (Apple Silicon included;
  the embedding and reranking stages then run on CPU or MPS instead of CUDA).
- Not tested on Windows; use WSL2 there.

### Hardware

No non-standard hardware is required.

- **Demo (§3):** CPU only. ~8 GB RAM, ~2 GB free disk beyond the environment.
- **Full reproduction (§4):** the answering, retrieval-agent, and judge models are
  called over HTTP APIs, so no local GPU is needed for them. A CUDA GPU is
  recommended for the two local model stages — embedding the corpus
  (`build_index.py`) and cross-encoder reranking (`run_reranker.py`). Any GPU with
  ≥16 GB memory is sufficient; both stages also run on CPU, more slowly. The
  reported runs used NVIDIA H100 NVL GPUs (driver 580.173.02, CUDA 12.6).
- Running an open-weight answerer locally instead of via Together/Fireworks (the
  `provider: local` option, e.g. behind vLLM) is the only setting that requires
  large GPU memory; it is optional and not needed to reproduce the reported numbers.

### API access

All LLM stages call hosted models. Set only the keys for the providers you intend
to use, in the shell or in a `.env` file at the repository root:

```bash
export OPENAI_API_KEY=...      # GPT-5 family; used by the demo
export TOGETHER_API_KEY=...    # Qwen, Llama, DeepSeek, Mistral, MiniMax
export ANTHROPIC_API_KEY=...   # LLM judge and question-annotation scripts
export FIREWORKS_API_KEY=...
export NCBI_API_KEY=...        # PubMed/PMC corpus building only
```

Local OpenAI-compatible endpoints (vLLM, TGI) are addressed with `provider: local`
plus `local_base_url` in the relevant config. The main pipeline scripts (retrieval,
answering) don't require an API key for `local`; the judge and annotation scripts
under `scripts/` additionally honour `LOCAL_BASE_URL`/`LOCAL_API_KEY` env vars as
defaults when no `base_url` is passed explicitly.

---

## 2. Installation guide

```bash
git clone <repository-url> dermrag
cd dermrag

poetry install          # creates ./.venv from poetry.lock
poetry run pip install anthropic==0.104.1   # see note below
```

Then either prefix commands with `poetry run`, or activate the environment
(`source .venv/bin/activate`). All commands below assume the repository root as the
working directory.

> **Note on `anthropic`.** The judge (`run_answer_judge.py`) and the LLM annotation
> scripts under `scripts/` import the `anthropic` SDK, which is not yet in
> `poetry.lock`; install it with the `pip` line above. Everything else, including the
> whole demo, works without it.

Without Poetry:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r <(poetry export --without-hashes -f requirements.txt)   # or install
                                                                       # pyproject deps manually
```

### Typical install time

On a normal desktop with a 100 Mbit connection: **10–25 minutes**, dominated by the
PyTorch/CUDA and Hugging Face wheels. The resolved environment is ~10 GB on disk.
The demo additionally downloads two small models on first use (~150 MB total,
cached in `~/.cache/huggingface`).

---

## 3. Demo

The demo runs the complete pipeline — index → agentic retrieval → rerank → answer —
on a **small real slice of the study's own data, shipped in [demo/](demo/)**, so a
reviewer needs no external downloads. The demo questions are multiple-choice, so it
stops at answer generation; the LLM judge (§4) grades free-text answers and has
nothing to do on a fixed answer set.

- [demo/corpus/](demo/corpus/) — two real EADV acne guideline documents copied
  verbatim from `Data/EADV/Acne/` (the European S3 guideline on acne treatment, and a
  cross-guideline comparison of oral isotretinoin recommendations). EADV guidelines
  are publicly available and redistributable, unlike the copyrighted textbooks, so
  only `EADV/` is included here; the `books` collection is built empty.
- [demo/questions_demo.json](demo/questions_demo.json) — six real questions copied
  verbatim from the study's own question set (`medqa_derma_final.json`), all about
  acne/isotretinoin management, so their answers are at least partly grounded in the
  two guideline documents above.
- [demo/configs/](demo/configs/) — demo counterparts of the configs in `configs/`,
  pointing at a small CPU-friendly embedder (`all-MiniLM-L6-v2`), the xsmall
  cross-encoder, and `gpt-5-mini` for the agent/answerer/judge.

Because this is real guideline text rather than a passage written to guarantee a
clean retrieval, the demo also shows a real failure mode: on one question the
planning step routed to the (empty) `books` collection and came back with zero
documents. That is expected — see [Expected output](#expected-output) below.

Requires `OPENAI_API_KEY` (about 20 LLM calls in total, all with `gpt-5-mini`).

```bash
# 1. Build the demo ChromaDB index (CPU)
python build_index.py --data-dir demo/corpus --chroma-path demo/chromadb_demo \
                      --embed-model sentence-transformers/all-MiniLM-L6-v2 \
                      --device cpu --batch-size 8

# 2. Agentic retrieval (plan → decompose → retrieve → self-critique → reformulate)
python run_agentic_retrieval_gaps.py --config demo/configs/agentic_retrieval_demo.yaml

# 3. Cross-encoder reranking
python run_reranker.py --config demo/configs/reranker_demo.yaml

# 4a. Answer the questions with the retrieved context (multiple choice)
python run_answerer.py --config demo/configs/answerer_demo_rag.yaml
# 4b. Same questions, context withheld (no-retrieval baseline)
python run_answerer.py --config demo/configs/answerer_demo_base.yaml

# 5. Score the two conditions
python demo/score_demo.py demo/outputs/results_demo_rag.csv \
                          demo/outputs/results_demo_base.csv
```

Single questions can be put through the same pipeline interactively:

```bash
python query.py --config demo/configs/query_demo.yaml \
  "What is the first-line topical treatment for mild to moderate papulopustular acne?"
```

### Expected output

Step 1 loads 0 book documents (the `books` collection is created empty — no
copyrighted textbook content ships with this repo) and 2 EADV guideline documents
(58 chunks) and writes `demo/chromadb_demo/` (~1 MB).

Step 2 writes one JSON record per question to
`demo/outputs/retrieved_docs_demo/{0..5}.json`, each containing the agent's plan,
sub-queries, per-iteration critiques (`sufficient`, `confidence`, `gaps`), and the
retrieved passages. In our run it reported ~6 documents and 1.5 iterations per
question on average. Four of the six records ended `final_sufficient=False` — the
real EADV guideline text doesn't spell out a US-style iPLEDGE pregnancy-prevention
protocol as explicitly as a synthetic passage would, so the self-critique correctly
flagged those retrievals as incomplete even though the answerer still had enough to
work with. One record (`2`) got zero documents: the planning step routed that
question to the (empty) `books` collection instead of `eadv_guidelines`. The demo
configs deliberately keep such records (`sufficient_only: false`) so every stage
still processes all six questions — this is what a real, imperfect corpus looks
like, not a bug.

Step 3 writes the same records to `demo/outputs/retrieved_docs_demo_reranked/`, with
`documents` re-sorted by cross-encoder score, plus `reranker_scores`, truncated to
`top_k: 5`.

Steps 4–5 write one CSV row per question (`question_id`, `question`,
`answer_options`, `correct_choice`, `llm_response`, `used_sources`,
`retrieval_iterations`, `final_doc_count`, `final_sufficient`, `retrieved_documents`,
…) and print:

```
=== demo/outputs/results_demo_rag.csv ===
answered: 6   correct: 6   accuracy: 100.00%
 question_id correct_choice llm_response  correct  final_doc_count                                          used_sources
           0              B            B     True                5                            [Document ID 2], [Document ID 4]
           1              B            B     True                5           [Document ID 3], [Document ID 4], [Document ID 5]
           2              B            B     True                0                                                       NaN
           ...
```

Both the RAG and no-retrieval-baseline conditions score 6/6 in our run — these are
real MedQA-style questions with a small, fixed answer set, and `gpt-5-mini` has
enough general medical knowledge to get isotretinoin management right from the
options alone. That the baseline matches RAG here is expected on multiple-choice
questions the model already knows; the manuscript's RAG-vs-base comparison is run on
the full open-ended reader-study format (§4), where the model can't rely on option
elimination and grading requires the LLM judge (`run_answer_judge.py`).

Reference outputs from our run are committed under [demo/outputs/](demo/outputs/) for
comparison. Exact agreement is not expected: the retrieval agent and the answerer are
non-deterministic LLM calls (only the sampling code is seeded), so the retrieved
documents and the wording of sub-queries can differ between runs. What should
reproduce is the shape of the output — six records at every stage, cited
`used_sources` for the RAG condition — and roughly this pattern of results: strong
multiple-choice accuracy in both conditions, and at least one record that is not
perfectly retrieved. The demo verifies that the pipeline runs end to end on real
data, not the manuscript's effect size.

### Expected run time

On a normal desktop (CPU-only, models already downloaded), measured end to end:

| Step | Time |
|---|---|
| 1 — build demo index | ~15 s |
| 2 — agentic retrieval (6 questions) | ~3 min |
| 3 — rerank | ~15 s |
| 4a/4b — answer, MCQ + baseline | ~10 s each |
| 5 — scoring | instant |
| **Total** | **~3.5 minutes** |

Add ~2–5 minutes on the first run for the Hugging Face model downloads.

---

## 4. Instructions for use

### Running the pipeline on your own data

Two inputs are needed: a corpus to retrieve from, and a question set.

**1. Corpus.** Markdown files laid out as the demo corpus is:

```
<corpus-root>/EADV/<condition>/<document_title>/vlm/*.md   → collection "eadv_guidelines"
<corpus-root>/books/<book_title>/vlm/*.md                  → collection "books"
```

Files are chunked on markdown headers (`MarkdownNodeParser`), so keep `##`/`###`
structure. Index them with:

```bash
python build_index.py --data-dir <corpus-root> --chroma-path <db-dir> \
                      --embed-model <hf-model> [--device cuda|cpu] [--batch-size N]
```

`--embed-model` **must** match the `embed_model` in the retrieval config that later
queries this index; querying with a different embedder silently returns meaningless
neighbours. Defaults reproduce the paper's index
(`Snowflake/snowflake-arctic-embed-l-v2.0` into `./chromadb_snowflakev2`).
`python build_index.py --help` lists all flags. PMC case reports are added to a
third collection (`case_reports`) by [ingest_case_reports.py](ingest_case_reports.py),
which also enforces the leakage exclusion described in its docstring.

**2. Question set.** JSON in the schema used throughout:

```json
{"questions": {"0": {"id": "...",
                     "question": "clinical stem …",
                     "answer_options": {"A": "…", "B": "…", "C": "…", "D": "…"},
                     "correct_choice": ["A"],
                     "correct_answer": "text of the correct option"}}}
```

See [demo/questions_demo.json](demo/questions_demo.json) for a complete example.
The manuscript's question sets, including the PMC-case-report-derived vignettes,
are distributed via figshare — see [Data availability](#data-availability).

**3. Run the stages.** Copy the configs in [demo/configs/](demo/configs/) or
[configs/](configs/) and edit the paths. The knobs that matter most:

| Config key | Stage | Meaning |
|---|---|---|
| `input_json`, `output_dir` | retrieval | question set in, one JSON record per question out |
| `chroma_path`, `collection_name`, `books_collection_name`, `embed_model`, `topk` | retrieval | index and per-query document budget |
| `agent_model`, `provider`, `max_iterations`, `confidence_threshold`, `enable_query_decomposition`, `enable_self_critique` | retrieval | the agent loop; `max_iterations: 1` with both flags off approximates naive RAG |
| `reranker_model`, `top_k`, `sufficient_only` | rerank | cross-encoder and how many passages survive |
| `rag_mode`, `open_ended`, `retrieved_docs_dir`, `model`, `provider`, `sufficient_only` | answerer | `rag_mode: false` is the no-retrieval baseline; `open_ended: true` withholds the options |
| `input_csv`, `model`, `provider`, `include_question` | judge | grades free-text answers as Correct / Partially correct / Incorrect |

`provider` is `openai`, `together`, `anthropic` (judge only), or `local`; with
`local`, point `local_base_url` at any OpenAI-compatible server. Every stage
checkpoints and resumes — re-running skips questions that already have output — so
interrupted runs are safe to restart.

### Reproducing the manuscript results

Download the question sets and de-identified reader-study data from figshare (see
[Data availability](#data-availability)) and place the question-set JSONs under
`data/derma/`, which is where the configs in `configs/` expect them:

```bash
mkdir -p data/derma   # then copy medqa_derma_final.json, pubmed_vignettes_medqa.json, … here
```

The retrieval corpora (`Data/`) cannot be redistributed (see below), so the retrieval
stage cannot be re-run verbatim; every downstream stage can be reproduced from the
committed per-model result CSVs in [results/](results/).

Full pipeline, per benchmark (`medqa` = curated exam-style questions, `pubmed` =
case-report vignettes):

```bash
# Index (needs Data/; defaults match the paper)
python build_index.py
python ingest_case_reports.py --input <path-to-pmc-case-reports.jsonl> --embed

# Retrieval: agentic (multi-step, gap-driven) or naive single-shot
python run_agentic_retrieval_gaps.py        --config configs/agentic_retrieval_gaps.yaml
python run_agentic_retrieval_gaps_pubmed.py --config configs/agentic_retrieval_gaps_pubmed.yaml
python run_naive_retrieval.py               --config configs/naive_retrieval.yaml

# Rerank
python run_reranker.py --config configs/reranker_mixedbread.yaml
python run_reranker.py --config configs/reranker_naiverag.yaml

# Answer — one config per model × condition (base / RAG / naive-RAG), e.g.
python run_answerer.py --config configs/gpt5/answerer_gpt5_pubmed_rag.yaml
python run_answerer.py --config configs/gpt5/answerer_gpt5_pubmed_base.yaml
#   … repeat for configs/{gpt5mini,qwen2.5,llama4,gemma3,deepseekv3.1,mistrallarge2,minimax2.7}/

# Score
python run_answer_judge.py       --config configs/answer_judge.yaml        # per results CSV
python run_faithfulness_judge.py --config configs/faithfulness_judge.yaml
```

Analyses and figures, once the result CSVs exist:

| Analysis | Entry point |
|---|---|
| Benchmark accuracy, base vs RAG, per model | [notebooks/llm_accuracy.ipynb](notebooks/llm_accuracy.ipynb) |
| Question metadata (disease category, type, structure, quality) | [scripts/categorize_questions.py](scripts/categorize_questions.py), [scripts/label_question_structure.py](scripts/label_question_structure.py), [scripts/flag_question_quality.py](scripts/flag_question_quality.py), [scripts/assign_subcategories.py](scripts/assign_subcategories.py), [scripts/dedup_diseases.py](scripts/dedup_diseases.py) |
| Question-type and disease-category distributions | [notebooks/question_type_distribution.ipynb](notebooks/question_type_distribution.ipynb), [notebooks/dermatology_radial_tree.ipynb](notebooks/dermatology_radial_tree.ipynb) |
| Retrieval quality: document relevance and context sufficiency | [scripts/run_relevance_scoring.py](scripts/run_relevance_scoring.py) → [scripts/analyze_doc_relevance.py](scripts/analyze_doc_relevance.py), [scripts/analyze_context_sufficiency.py](scripts/analyze_context_sufficiency.py) |
| Citation faithfulness | [run_faithfulness_judge.py](run_faithfulness_judge.py) → [analyze_faithfulness.py](analyze_faithfulness.py); counterfactual evidence swaps: [analyze_counterfactual.py](analyze_counterfactual.py) |
| Citation precision / support vs correctness | [notebooks/citation_precision_support_plots.ipynb](notebooks/citation_precision_support_plots.ipynb), [notebooks/support_correctness_relationship.ipynb](notebooks/support_correctness_relationship.ipynb), [notebooks/citation_agreement_accuracy.ipynb](notebooks/citation_agreement_accuracy.ipynb) |
| Error taxonomy of wrong answers | [scripts/classify_error_types.py](scripts/classify_error_types.py) |
| Reader-study figures (accuracy, reliance, change flows, agreement) | [notebooks/reader_study_accuracy.ipynb](notebooks/reader_study_accuracy.ipynb), [notebooks/reader_study_reliance.ipynb](notebooks/reader_study_reliance.ipynb), [notebooks/reader_study_change_flows.ipynb](notebooks/reader_study_change_flows.ipynb), [notebooks/rsr_by_citation_support.ipynb](notebooks/rsr_by_citation_support.ipynb) |
| Skin-tone and sex subgroup analyses | [scripts/classify_skin_tone.py](scripts/classify_skin_tone.py), [scripts/analyze_skin_gender_llm.py](scripts/analyze_skin_gender_llm.py) |

Free-text answers from open-weight models occasionally arrive with commentary around
the answer; [clean_llm_outputs.py](clean_llm_outputs.py) extracts the selected answer into a
`*_cleaned.csv` before scoring, and the `_cleaned` files are what the analyses read.

---

## Repository layout

| Path | Contents |
|---|---|
| [build_index.py](build_index.py), [ingest_case_reports.py](ingest_case_reports.py) | corpus indexing |
| `run_agentic_retrieval*.py`, [run_naive_retrieval.py](run_naive_retrieval.py) | retrieval stage |
| [run_reranker.py](run_reranker.py) | cross-encoder reranking |
| [run_answerer.py](run_answerer.py) | answer generation (base / RAG, MCQ / open-ended) |
| [run_answer_judge.py](run_answer_judge.py), [run_faithfulness_judge.py](run_faithfulness_judge.py) | LLM judges |
| [query.py](query.py) | interactive single-question tool over the same pipeline |
| [configs/](configs/) | YAML configs for every stage and model |
| [demo/](demo/) | small real corpus/question slice, demo configs, reference outputs |
| [scripts/](scripts/) | question annotation, retrieval/faithfulness analyses, error typing |
| [notebooks/](notebooks/) | figure generation |
| [results/](results/) | model outputs and judge ratings per model and condition |

## Data availability

- **Question sets and de-identified reader-study data:** figshare,
  https://figshare.com/s/fa2ff8eb45984acfecf6
- **Retrieval corpora (`Data/`) are not distributed here.** The EADV guidelines are
  publicly available from the EADV; the dermatology textbooks are copyrighted and
  cannot be redistributed. See the data availability statement in the paper.
- **Demo data** in [demo/](demo/) is a small real slice of the data above: two public
  EADV guideline documents (`demo/corpus/`, copied from `Data/EADV/Acne/`) and six
  real questions (`demo/questions_demo.json`, copied from `medqa_derma_final.json`).
  It is not the copyrighted textbook content and not the full question set.
