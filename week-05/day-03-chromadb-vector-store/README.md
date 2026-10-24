# W5D3 — ChromaDB Vector Store and PDF RAG

## Objective
Build a semantic search pipeline using ChromaDB and Ollama embeddings, then demonstrate retrieval-augmented generation (RAG) over a sample PDF.

## Features
- Store 20 learning documents in a ChromaDB collection.
- Generate embeddings locally with Ollama `nomic-embed-text`.
- Search using cosine distance and return the top 3 results.
- Filter results using document metadata.
- Extract text from a PDF, split it into overlapping chunks, embed the chunks, and retrieve the top 3 relevant chunks.
- Pass retrieved context to `llama3.2:3b` and save evidence as JSON.

## Project Structure
- `src/vector_store.py` — document collection, embedding, semantic search, and evidence.
- `src/create_sample_pdf.py` — creates the sample PDF.
- `src/pdf_rag.py` — PDF extraction, chunking, retrieval, and answer generation.
- `sample_data/vector_databases_guide.pdf` — generated sample PDF.
- `tests/` — automated tests.
- `outputs/` — JSON evidence from searches and RAG.
- `docs/CIA_INTERACTIONS.md` — actual CIA review interactions.
- `SELF_REVIEW.md` — implementation review and limitations.

## Setup
Run these commands from the repository root in an activated virtual environment:

```powershell
python -m pip install -r week-05/day-03-chromadb-vector-store/requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Make sure Ollama is running before executing the scripts.

## Run
```powershell
python week-05/day-03-chromadb-vector-store/src/vector_store.py
python week-05/day-03-chromadb-vector-store/src/create_sample_pdf.py
python week-05/day-03-chromadb-vector-store/src/pdf_rag.py
python -m pytest week-05/day-03-chromadb-vector-store/tests -q
```

## Concepts
- **Embeddings:** numerical vectors representing text meaning.
- **Cosine similarity:** compares vector directions; smaller cosine distance indicates greater similarity in this ChromaDB collection.
- **Metadata filtering:** limits eligible results using fields such as category.
- **RAG:** retrieves relevant source chunks and supplies them as context to a language model.

## Evidence
The scripts write `outputs/vector_search_results.json` and `outputs/pdf_rag_evidence.json`. Review the retrieved chunks and generated answer to assess grounding. Retrieval and context instructions reduce unsupported claims but do not guarantee factual correctness.

## Limitations
The demo uses a local Ollama server and an in-memory ChromaDB client, so collections are recreated when the scripts run. This is a learning demo, not a persistent production deployment.