# W7D5 - Multi-Document RAG System

## Overview
A multi-document Retrieval-Augmented Generation system combining CrewAI, LangGraph, MLflow, Ragas, ChromaDB and Ollama.

## Architecture
1. Ingestion: Reads multiple text documents and splits them into overlapping chunks.
2. Embeddings: Generates vector embeddings using nomic-embed-text through Ollama.
3. Storage: Stores document chunks and metadata in ChromaDB.
4. Retrieval: Retrieves the top three relevant chunks for a user question.
5. Generation: Uses llama3.2:3b to generate a context-grounded answer.
6. Workflow: LangGraph coordinates retrieval and generation.
7. Agent review: CrewAI reviews the generated answer.
8. Evaluation: Ragas Faithfulness evaluates answer grounding.
9. Experiment tracking: MLflow records parameters, metrics and evaluation runs.

## Technology Stack
- Python 3.12
- CrewAI
- LangGraph
- MLflow
- Ragas
- ChromaDB
- Ollama

## Setup
Create and activate the dedicated Python environment, then install the project dependencies.

Start Ollama and ensure these models are available:
- llama3.2:3b
- nomic-embed-text:latest

## Usage

Ingest documents:

    python src/rag_system.py --ingest

Ask a question:

    python src/rag_system.py --question "What is retrieval-augmented generation?"

Run CrewAI review:

    python src/rag_system.py --question "What is retrieval-augmented generation?" --crew

Run Ragas evaluation:

    python src/evaluate_ragas.py

Run tests:

    python -m pytest tests -v

## Evaluation Evidence
- Seven unit tests passed.
- Live RAG question successfully answered.
- CrewAI answer review completed.
- Ragas Faithfulness score: 1.0000.
- MLflow recorded the evaluation metric.

## Limitations
- Local model inference depends on available hardware.
- Current retrieval uses vector similarity with a fixed top-k.
- Evaluation currently focuses on faithfulness.
- Further testing with a larger evaluation dataset is recommended.

