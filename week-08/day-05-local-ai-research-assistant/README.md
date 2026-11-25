# W8D5 - Local AI Research Assistant

## Overview
A local AI research assistant using retrieval-augmented generation, agent review, evaluation, and experiment tracking.

## Technology Stack
- Python 3.12
- LangGraph: workflow orchestration
- CrewAI: research answer review
- ChromaDB: vector database
- Ollama: local LLM and embeddings
- MLflow: experiment tracking
- Ragas: faithfulness evaluation
- Pytest and Ruff: quality assurance

## Architecture
1. Load research documents.
2. Split documents into chunks.
3. Generate embeddings using Ollama.
4. Store embeddings in ChromaDB.
5. Retrieve relevant context through LangGraph.
6. Generate answers using Ollama.
7. Review answers using CrewAI.
8. Evaluate faithfulness using Ragas.
9. Track experiments using MLflow.

## Run Commands
Run these commands from the repository root:

- Index: `python week-08/day-05-local-ai-research-assistant/main.py --ingest`
- Ask: `python week-08/day-05-local-ai-research-assistant/main.py --question "What is RAG?" --no-review`
- Review: `python week-08/day-05-local-ai-research-assistant/main.py --question "Explain AI agent workflows."`
- Evaluate: `python week-08/day-05-local-ai-research-assistant/main.py --evaluate`
- Test: `python -m pytest week-08/day-05-local-ai-research-assistant/tests -v`

## Validation
- 10 automated tests passed.
- Ruff lint passed.
- Ruff formatting passed.
- Python compilation passed.
- Ollama document ingestion and research query succeeded.
- CrewAI answer review succeeded.
- Latest real Ragas Faithfulness result: 1.0000.
- MLflow SQLite tracking succeeded.
- Automated Ragas test uses mocked components for deterministic validation.

Detailed reports are stored in `outputs/evidence/`.

## CIA Mentor Reviews
Two independent CIA mentor reviews were completed. Their findings, implementation decisions, and remaining recommendations are documented in `docs/cia_review_log.md`.

## Limitations
The prototype uses a small sample dataset and local models. Larger datasets, broader evaluation metrics, production deployment, structured monitoring, and additional integration testing remain future improvements.
