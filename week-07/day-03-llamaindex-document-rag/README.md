# W7D3 — LlamaIndex Document Indexing and Querying

## Objective
Index local text documents with LlamaIndex, build a query engine, connect it to ChromaDB, and compare query latency.

## Implementation
- Five local TXT knowledge documents loaded using SimpleDirectoryReader.
- In-memory VectorStoreIndex used as the baseline.
- ChromaVectorStore connected to a persistent ChromaDB collection.
- Ollama embedding model: nomic-embed-text:latest.
- Ollama language model: qwen2.5:3b.
- Same ten questions run through both query engines.
- Retrieval depth: similarity_top_k=2.
- Expected source filenames, generated answers, and end-to-end latency recorded.
- Chroma collection is recreated for each benchmark run.

## Benchmark Results

| Metric | In-memory | ChromaDB |
|---|---:|---:|
| Queries | 10 | 10 |
| Expected source found in top 2 | 10/10 | 10/10 |
| Mean latency | 19,133.34 ms | 11,040.17 ms |
| Median latency | 18,895.56 ms | 11,304.88 ms |
| P95 latency | 26,065.06 ms | 19,117.44 ms |
| Standard deviation | 5,241.97 ms | 5,369.95 ms |

ChromaDB was approximately 42.3% faster by mean latency in this particular run. These are end-to-end query timings that include embedding and LLM response generation, not isolated vector-search timings.

## Testing
Four pytest tests passed:
- Source documents exist.
- Both engines have ten recorded queries.
- Answers and latency statistics are present.
- Expected-source tracking values are boolean.

## CIA Mentor Reviews
Two genuine Full Stack Mentor Mode reviews were completed and recorded in `docs/CIA_INTERACTIONS.md`.

## Limitations
- The corpus contains only five small text documents.
- The evaluation contains ten questions.
- Expected-source verification checks presence within the top two retrieved sources, not top-one accuracy or full relevance.
- Generated answers are captured but are not formally scored for factuality or hallucination.
- Latency measurements are from one sequential run on one machine; no statistical significance is established.
- Results should not be generalized to larger datasets or concurrent workloads.

## Future Improvements
Use a larger and more diverse corpus, add paraphrased and negative queries, evaluate answer factuality and retrieval relevance, and repeat benchmarks to support more robust statistical comparisons.

## Run
```powershell
python week-07/day-03-llamaindex-document-rag/src/run_rag.py
python -m pytest week-07/day-03-llamaindex-document-rag/tests -v
```
