# W7D1 — Haystack Pipeline Architecture

## Objective

Build a Haystack document retrieval pipeline using five PDF documents.
Implement BM25 and dense retrieval, execute ten identical questions,
and compare retrieval precision.

## Technology Stack

- Python 3.14
- Haystack AI 3.5.0
- Ollama Haystack 7.0.2
- Ollama
- nomic-embed-text
- pypdf
- pytest
- ReportLab

## Architecture

### Document Indexing

PDF files
  -> PyPDFToDocument
  -> DocumentSplitter
  -> OllamaDocumentEmbedder
  -> InMemoryDocumentStore

### BM25 Retrieval

User Query
  -> InMemoryBM25Retriever
  -> ExtractiveReader
  -> Answer and Source

### Dense Retrieval

User Query
  -> OllamaTextEmbedder
  -> InMemoryEmbeddingRetriever
  -> ExtractiveReader
  -> Answer and Source

## Dataset

Five PDF documents are included:

1. haystack_pipeline_architecture.pdf
2. prompt_engineering.pdf
3. responsible_ai_model_card.pdf
4. retrieval_methods_and_evaluation.pdf
5. vector_databases_guide.pdf

The documents are converted into 22 chunks during indexing.

## Installation

Activate the project virtual environment.

Install dependencies:

```powershell
python -m pip install haystack-ai ollama-haystack pypdf reportlab pytest
```

Ensure Ollama is running and the embedding model is available:

```powershell
ollama pull nomic-embed-text
ollama list
```

## Execution

Run from the repository root:

```powershell
python week-07/day-01-haystack-pipeline/src/haystack_pipeline.py
```

The pipeline:

1. Converts five PDFs into Haystack Documents.
2. Splits documents into 180-word chunks with 30-word overlap.
3. Generates embeddings using nomic-embed-text.
4. Writes document chunks into InMemoryDocumentStore.
5. Builds BM25 and dense retrieval pipelines.
6. Executes the same ten questions with both methods.
7. Calculates Precision@3.
8. Saves the comparison evidence.

## Testing

```powershell
python -m pytest week-07/day-01-haystack-pipeline/tests -v
```

## Evaluation Results

| Metric | BM25 | Dense |
|---|---:|---:|
| Queries | 10 | 10 |
| Precision@3 | 0.5000 | 0.5667 |

Dense retrieval achieved a 0.0667 absolute improvement in Precision@3
on this evaluation dataset.

This is a small experimental comparison, not proof that dense retrieval
will outperform BM25 on every dataset.

## Evaluation Methodology

The evaluation uses expected source PDF labels as initial relevance
judgments. Precision@3 is calculated from the three retrieved chunks.

A result is considered relevant when its source filename matches the
question's expected source.

The current evaluation is therefore source-level proxy evaluation.
Independent chunk-level human relevance judgments would provide a
stronger evaluation.

## Reader

The custom ExtractiveReader selects a query-relevant sentence from
the highest-ranked retrieved document.

It is an extractive demonstration component, not a generative LLM.
It does not use qwen2.5:3b for answer generation.

## Evidence

The execution evidence is saved to:

`outputs/retrieval_evidence.json`

It includes:
- Ten BM25 query results.
- Ten dense query results.
- Retrieved source filenames.
- Precision@3 for each query.
- Mean latency.
- Aggregate comparison.
- Extractive Reader responses.

## Limitations

- InMemoryDocumentStore is not persistent.
- Five documents represent a small evaluation dataset.
- Source-level relevance labels are initial manual judgments.
- ExtractiveReader does not generate complete contextual answers.
- Dense retrieval requires the local Ollama embedding model.
- PDF parser warnings occur for two source PDFs, although their text
  extraction completed successfully.

## Viva Questions

### 1. Explain what you built and your design decisions.

I built a Haystack-based PDF retrieval system using five documents.
I implemented PDF conversion, document splitting, indexing, BM25
retrieval and dense retrieval.

I used BM25 for lexical matching and Ollama embeddings for semantic
retrieval. Both methods were evaluated using the same ten questions
and Precision@3.

### 2. What was the hardest part?

The hardest part was integrating the retrievers and Reader outputs
correctly in Haystack and ensuring that the evaluation used the
actual top-three retrieved chunks.

I resolved the pipeline output issue and corrected the evaluation
logic before running the automated tests.

### 3. What would you improve with one more day?

I would introduce a persistent document store, improve chunk-level
relevance judgments, test more embedding models, and implement a
generative answer component with source citations.

## Status

Implementation completed. Automated tests passed.
CIA mentor feedback has been reviewed and incorporated; Git submission is the remaining step.

## Final Evaluation Update — CIA Feedback Incorporated

The initial evaluation counted repeated chunks from the same PDF separately. Following the CIA review, the evaluation now deduplicates retrieved results by PDF source before calculating source-level Precision@3.

A dense retrieval warm-up query is also executed before latency measurement.

| Metric | BM25 | Dense |
|---|---:|---:|
| Source-deduplicated Precision@3 | 0.3333 | 0.3333 |
| Mean query latency | 3.88 ms | 51.29 ms |

**Conclusion:** Both methods tied on this 10-question, 5-PDF, 22-chunk dataset. The results are preliminary and do not establish that either retrieval method is generally superior.

**Evaluation limitation:** Relevance is judged at PDF-source level, not by independently annotated passages. Precision@3 uses a fixed denominator of three; missing unique sources count as non-relevant. The dense warm-up is excluded from the reported query timings.

The earlier scores of BM25 0.5000 and Dense 0.5667 are superseded by this corrected evaluation.

