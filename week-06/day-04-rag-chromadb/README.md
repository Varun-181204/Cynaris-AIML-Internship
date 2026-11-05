# W6D4 â€” RAG Pipeline with LangChain, ChromaDB and Ollama

## Objective
Build a local retrieval-augmented generation pipeline with a persistent ChromaDB cosine collection, 20 embedded sample documents, metadata-filtered similarity search, and PDF-grounded answers from Ollama.

## Environment
Run commands from this W6D4 folder with the project virtual environment activated.

```powershell
python -m pip install chromadb langchain-community langchain-text-splitters pypdf
```

The implementation also uses the already-installed `langchain-ollama` and `langchain-core` packages. Ollama must be running, with these models available:

```powershell
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

## Run the complete demonstration

```powershell
python -m src.rag_pipeline
```

Optional question:

```powershell
python -m src.rag_pipeline --question "Explain few-shot prompting using this PDF."
```

The script:
1. Creates/reuses the `w6d4_rag` ChromaDB collection with cosine distance.
2. Adds 20 sample documents using `nomic-embed-text` embeddings.
3. Runs top-3 semantic search and metadata-filtered search.
4. Loads `data/Module-3-PE.pdf`, splits it into overlapping chunks, and indexes the chunks with source/page metadata.
5. Retrieves the top 3 PDF chunks and passes only those chunks to `qwen2.5:3b`.
6. Saves reproducible run evidence to `outputs/rag_evidence.json`.

## Tests

```powershell
python -m pytest tests -q
```

Unit tests use deterministic fake vectors and do not require an Ollama call. The full demonstration is the live integration test.

## Design notes
- ChromaDB collection metadata sets `hnsw:space` to `cosine`.
- PDF chunks retain source filename and zero-based page metadata from `PyPDFLoader`.
- The answer prompt requires the model to use retrieved context only and cite source pages.
- The database is generated locally and should not be committed; the evidence JSON is the reviewable output.

## Viva quick notes
- **Cosine similarity:** compares vector direction rather than raw magnitude; useful for comparing semantic embedding vectors.
- **Embedding vs one-hot:** embeddings are dense, lower-dimensional representations that can encode semantic relationships; one-hot vectors are sparse and represent category identity without semantic proximity.
- **ChromaDB vs FAISS:** ChromaDB provides collections, metadata, persistence, and filtering in an integrated vector-database interface; FAISS is a fast vector-search library with more manual responsibility for metadata and persistence.
