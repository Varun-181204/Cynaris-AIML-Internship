# W7D3 Self Review

## Completed Work
- [x] Loaded five local TXT files using LlamaIndex.
- [x] Built an in-memory VectorStoreIndex and query engine.
- [x] Connected LlamaIndex to persistent ChromaDB.
- [x] Configured Ollama embeddings and local LLM.
- [x] Ran the same ten questions through both query engines.
- [x] Recorded generated answers, retrieved source files, and latency.
- [x] Added Ollama checks, embedding warm-up, and document validation.
- [x] Completed two genuine CIA mentor interactions and documented feedback.

## Testing
- [x] Both query engines completed ten queries.
- [x] Expected source appeared in the top two retrieved results for all ten questions in both engines.
- [x] Four pytest tests passed.
- [x] Evidence JSON contains mean, median, p95, and standard deviation.

## Results
In-memory mean latency: 19,133.34 ms.
ChromaDB mean latency: 11,040.17 ms.
ChromaDB was approximately 42.3% faster by mean latency in this run.

## Reflection
Hardest part: Connecting LlamaIndex with Ollama embeddings and persistent ChromaDB while keeping the comparison consistent.

How I solved it: Used the same embedding and language models, identical questions, the same retrieval depth, a persistent Chroma collection, and recorded query-level evidence.

What I learned: How LlamaIndex builds vector indexes and query engines, how ChromaDB stores vectors persistently, and how to compare two retrieval backends.

Future improvement: Expand the dataset and query set, add negative retrieval tests, evaluate answer factuality, and repeat latency measurements for more reliable conclusions.
