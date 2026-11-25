# W8D4 Technical Documentation

## Purpose
This day documents, tests, and reviews the Week 8 production RAG work.

## System Components

### W8D1 – Production RAG API
- FastAPI endpoints: `/health`, `/ingest`, and `/ask`.
- Reuses the Week 7 RAG implementation.
- Isolates Week 8 runtime data and configuration.

### W8D2 – Ragas Evaluation
- LangChain and ChromaDB retrieval pipeline.
- Five source documents.
- Ten evaluation questions and ground-truth references.
- Ragas evaluation and MLflow experiment tracking.
- Retrieval top-k optimisation from 3 to 2.

### W8D3 – Haystack Retrieval and Deployment
- BM25 and dense retrieval.
- Five PDF documents and 22 chunks.
- Ten evaluation questions.
- FastAPI endpoints: `/`, `/health`, and `/query`.
- Docker and Railway deployment configuration.

## Validation
- W8D1: 7 tests passed.
- W8D2: 5 tests passed.
- W8D3: 19 tests passed.
- Total: 31 tests passed.
- Ruff lint and formatting checks: verified after final cleanup.

## Deployment Limitations
The W8D3 Docker build and live Railway deployment remain pending verification. Configuration files are present, but configuration alone does not establish a successful deployment.

## Future Improvements
- Add root-level test configuration to support a single combined pytest run.
- Add integration tests for deployed services.
- Complete Docker and Railway deployment verification.
- Expand retrieval evaluation datasets and perform deeper error analysis.
