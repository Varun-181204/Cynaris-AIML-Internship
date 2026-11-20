# W7D3 — LlamaIndex Document Indexing and Querying

## Objective
Index local text documents with LlamaIndex, build a query engine, connect it to ChromaDB, and compare query latency.

## Implementation
- Five local text knowledge files are loaded with `SimpleDirectoryReader`.
- `VectorStoreIndex` builds the in-memory index.
- Ollama provides local embeddings and LLM generation.
- `ChromaVectorStore` connects LlamaIndex to persistent ChromaDB storage.
- The same ten questions are run through both query engines.
- End-to-end latency and retrieved source filenames are recorded.

## Run
```powershell
python week-07/day-03-llamaindex-document-rag/src/run_rag.py
pytest week-07/day-03-llamaindex-document-rag/tests -v
```

## Evaluation
See `outputs/retrieval_evidence.json` for the measured answers, retrieved source files, and latency.

## Limitations
The evaluation uses five small text documents and ten questions. Source retrieval is tracked automatically; answer correctness still requires manual review against the source documents. Results are specific to the local hardware, models, and settings used.
