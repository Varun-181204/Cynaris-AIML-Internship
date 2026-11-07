# W6D4 Self-Review â€” RAG Pipeline with ChromaDB

## Assignment Requirements

- [x] Installed ChromaDB and created a persistent collection.
- [x] Added 20 sample documents with embeddings.
- [x] Configured cosine distance for similarity search.
- [x] Implemented metadata-filtered retrieval.
- [x] Loaded the supplied PDF using PyPDFLoader.
- [x] Split PDF content into chunks with metadata.
- [x] Retrieved the top 3 relevant PDF chunks.
- [x] Passed retrieved context to Ollama for answer generation.
- [x] Verified the generated answer against the retrieved PDF content.

## Testing

- [x] Verified the 20-document sample dataset.
- [x] Tested cosine similarity search.
- [x] Tested metadata filtering.
- [x] Tested PDF chunk metadata.
- [x] Tested unmatched metadata filters.
- [x] Tested empty retrieval context without invoking the LLM.
- [x] All 5 pytest tests passed.

## Evidence

- [x] Pipeline execution completed successfully.
- [x] 20 sample documents indexed.
- [x] 28 PDF chunks indexed.
- [x] Collection contains 48 documents.
- [x] Embedding dimension verified as 768.
- [x] Top 3 PDF chunks retrieved.
- [x] Generated answer saved in `outputs/rag_evidence.json`.

## Documentation

- [x] README includes setup and execution instructions.
- [x] Viva questions and answers documented.
- [x] CIA interaction record created.
- [x] Self-review checklist created.

## Final Review

- [x] Implementation runs successfully with the installed dependencies.
- [x] Tests pass.
- [x] Two actual CIA interactions recorded and reviewed.
- [ ] Final Git review and commits completed.
- [ ] Changes pushed to the existing W6 branch and PR.

**Known non-blocking warnings:** Deprecation warnings from the LangChain community integration and ChromaDB telemetry dependencies.

**Status:** Implementation and tests complete. Documentation and submission workflow still require final verification.
