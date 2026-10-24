# W5D4: Semantic Search with ChromaDB

## Objective
Implement semantic search using ChromaDB, Ollama embeddings, metadata filtering, and PDF Retrieval-Augmented Generation (RAG).

## Implementation
- `src/vector_store.py` — Creates a ChromaDB collection configured for cosine distance, embeds and stores 20 documents, performs similarity search, supports metadata filtering, and saves search evidence.
- `src/pdf_rag.py` — Extracts PDF text, chunks it with page and source metadata, retrieves the top 3 chunks, and passes context to Ollama.
- `src/create_sample_pdf.py` — Generates the sample PDF.
- `tests/test_vector_store.py` — Tests document count, cosine-search ordering, metadata filtering, and invalid inputs.
- `tests/test_pdf_rag.py` — Tests PDF chunking and RAG behavior.
- `sample_data/vector_databases_guide.pdf` — Sample source document.
- `outputs/vector_search_results.json` — Semantic-search evidence.
- `outputs/pdf_rag_evidence.json` — PDF RAG evidence.

## Requirements
Install dependencies from `requirements.txt`. Ollama must be running with `nomic-embed-text` for embeddings and `llama3.2:3b` for text generation.

## Run from the repository root

```powershell
python -m pytest week-05/day-04-semantic-search-chromadb/tests -q
python week-05/day-04-semantic-search-chromadb/src/vector_store.py
python week-05/day-04-semantic-search-chromadb/src/pdf_rag.py
```

## Verified Results
- **Unit tests:** 8 passed, with one non-blocking ChromaDB dependency deprecation warning.
- **Semantic search:** Indexed 20 documents and returned the top 3 results.
- **PDF RAG:** Indexed 3 chunks, retrieved 3 chunks, and generated an answer using Ollama.
- Both evidence JSON files were generated successfully.

## Key Concepts
- **Embeddings:** Numeric vector representations of text that capture semantic features.
- **Cosine distance:** Measures vector dissimilarity; lower distance indicates greater similarity.
- **Metadata filtering:** Restricts search results to documents matching specified metadata.
- **RAG:** Retrieves relevant context and includes it in a language-model prompt to ground the answer in source material.

## Limitations and Future Improvements
Retrieval quality depends on the embedding model, document quality, and query. Future improvements include evaluating a larger query set, adding live-Ollama integration tests, and checking generated answers against retrieved context.