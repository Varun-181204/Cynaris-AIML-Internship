# W7D2 - Haystack BM25 and Dense Retrieval

## Objective
Build and evaluate a document retrieval pipeline using Haystack 3.5.0.
Compare sparse BM25 retrieval and dense embedding-based retrieval over
five PDF reference documents.

## Dataset
Five PDFs are stored in `data/pdfs/`:
- haystack_pipeline_architecture.pdf
- prompt_engineering.pdf
- responsible_ai_model_card.pdf
- retrieval_methods_and_evaluation.pdf
- vector_databases_guide.pdf

The pipeline indexed 22 document chunks.

## Architecture
1. Load and convert PDF documents.
2. Split documents into chunks.
3. Generate document embeddings using Ollama and `nomic-embed-text`.
4. Store documents in an in-memory document store.
5. Evaluate BM25 and dense retrieval independently.
6. Deduplicate retrieved results by source PDF for source-level evaluation.
7. Generate extractive answers and save evaluation evidence.

## Evaluation
Ten questions were evaluated using top-k = 3.

| Metric | BM25 | Dense |
|---|---:|---:|
| Mean Precision@3 | 0.3333 | 0.3333 |
| Mean latency (ms) | 5.79 | 61.56 |
| Questions | 10 | 10 |

**Result:** Both methods achieved equal source-level Precision@3.
BM25 had lower observed mean latency in this run.

## Evaluation methodology and limitations
- Relevance is judged at PDF source level, not independently at passage level.
- Retrieved passages from the same PDF are deduplicated.
- Precision@3 uses a fixed denominator of three, even if fewer than
  three unique sources are returned.
- These are proxy relevance judgments, not independent passage annotations.
- Ten questions are a small evaluation set.
- Retrieval precision does not establish answer correctness.
- The recorded latency is an observed per-query measurement, not a
  repeated-run statistical benchmark.
- Reader outputs should be manually assessed separately.

## Run
From the repository root, activate the project virtual environment and run:

```powershell
python week-07/day-02-haystack-retrieval/src/haystack_pipeline.py
python -m pytest week-07/day-02-haystack-retrieval/tests/ -v
```

Ollama must be running and `nomic-embed-text` must be available locally.

## Evidence
The generated results are saved in:
`outputs/retrieval_evidence.json`

## Tests
12 automated tests passed during the W7D2 validation run.
