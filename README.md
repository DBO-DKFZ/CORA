# CORA

An agentic retrieval-augmented generation pipeline for dermatology question answering, evaluated against physicians in a reader study.

## Setup

```bash
poetry install
```

Requires Python 3.12. Set the API keys for whichever providers you use:

```bash
export OPENAI_API_KEY=...      # GPT-5 family
export TOGETHER_API_KEY=...    # Qwen, Llama, DeepSeek, Mistral
export ANTHROPIC_API_KEY=...
export FIREWORKS_API_KEY=...
export NCBI_API_KEY=...        # PubMed/PMC corpus building
```

## Pipeline

Every stage is driven by a YAML config in [configs/](configs/); model-specific configs live
in per-model subdirectories (`configs/gpt5/`, `configs/deepseekv3.1/`, …).

```bash
# 1. Build the ChromaDB index over the guideline and textbook corpora
python build_index.py

# 2. Retrieval — agentic (multi-step, gap-driven) or naive single-shot
python run_agentic_retrieval_gaps.py --config configs/agentic_retrieval_gaps.yaml

# 3. Rerank the pooled passages
python run_reranker.py --config configs/reranker_mixedbread.yaml

# 4. Generate answers
python run_answerer.py --config configs/gpt5/answerer_gpt5_pubmed_rag.yaml

# 5. Score
python run_answer_judge.py       --config configs/answer_judge.yaml
```

## Layout

| Path | Contents |
|---|---|
| [configs/](configs/) | YAML configs for every stage and model |
| [pubmed/](pubmed/) | Case-report retrieval from PMC and question generation |
| [scripts/](scripts/) | Question metadata annotation, error typing, auxiliary analyses |
| [scripts/reader_study/](scripts/reader_study/) | Reader-study analysis and figures |
| [fairness/](fairness/) | Counterfactual demographic-perturbation experiment |
| [guideline_adherence/](guideline_adherence/) | Guideline-answerability pipeline (see its README) |
| [notebooks/](notebooks/) | Figure generation |
| [results/](results/) | Retrieved passages, model outputs, judge ratings |

The retrieval corpora (`Data/`) are not distributed here: the EADV guidelines are publicly
available and the four dermatology textbooks are copyrighted. See the data availability
statement in the paper.

## Data

LLM results files in results/

Question sets and de-identified reader-study data are on figshare at https://figshare.com/s/fa2ff8eb45984acfecf6.
