# W5D4 Self-Review

## Completed Tasks
- [x] Created a ChromaDB collection configured for cosine distance.
- [x] Generated embeddings and stored 20 documents.
- [x] Implemented similarity search and metadata filtering.
- [x] Added tests for search ordering, metadata filtering, corpus size, and invalid inputs.
- [x] Implemented PDF extraction, chunking, top-3 retrieval, and Ollama answer generation.
- [x] Saved semantic-search and PDF RAG evidence.
- [x] Ran the W5D4 tests successfully: 8 passed.

## Evidence
- `outputs/vector_search_results.json`
- `outputs/pdf_rag_evidence.json`

## Known Warning
ChromaDB emits a dependency deprecation warning involving `asyncio.iscoroutinefunction`. The warning does not cause test failures.

## Areas for Improvement
- Add integration tests using a live Ollama instance.
- Evaluate retrieval quality against a representative query set.
- Test whether generated answers are supported by retrieved context.
- Investigate a compatible dependency update to resolve the warning.

## CIA Code Review
Record the actual feedback from two CIA Full Stack Mentor Mode review interactions in `docs/CIA_INTERACTIONS.md` before committing.