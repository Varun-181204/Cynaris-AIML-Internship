# CIA Full Stack Mentor Mode — W5D3 Review Log

## Interaction 1 — `src/vector_store.py`

**Review scope:** ChromaDB vector storage, Ollama embeddings, collection creation, similarity search, and metadata filtering.

**CIA feedback:**
- Verify that the Ollama `/api/embed` response uses the expected `embeddings` field and returns the correct number of vectors.
- Validate empty inputs and add type hints/docstrings where useful.
- Consider whether resetting the collection by default is appropriate.
- Consider persistent ChromaDB storage, configurable Ollama URL, logging, and more specific exception handling when deleting collections.

**Follow-up and verification:**
- Inspected the implementation: it validates non-empty embedding inputs and checks the returned embedding count and vector structure.
- The implementation uses Ollama's `/api/embed` endpoint and reads the `embeddings` response field.
- Similarity search validates the query and result count and supports metadata filtering.
- Optional persistence, logging, and configuration improvements were treated as suggestions, not confirmed defects.
- No changes were made solely on the basis of unverified suggestions.

## Interaction 2 — `src/pdf_rag.py`

**Review scope:** PDF text extraction and chunking, ChromaDB indexing, top-3 retrieval, metadata filtering, and Ollama answer generation.

**CIA feedback:**
- Requested more source excerpts to verify chunk-loop boundaries, retrieval filtering, and prompt construction.
- Recommended checking chunk overlap validation, unique chunk IDs, text cleanup, source/page metadata, and empty-answer handling.
- Suggested tests for empty PDFs, invalid overlap, and duplicate IDs.
- Confirmed that `collection.upsert` with precomputed embeddings is a valid approach.

**Follow-up and verification:**
- The implementation validates chunk size and overlap, cleans extracted text, and records source filename and page number.
- PDF chunks are indexed with precomputed Ollama embeddings.
- Retrieval requests the top three chunks with the PDF category filter.
- The prompt instructs the model to answer only from the supplied context and acknowledge insufficient context.
- Empty model responses are rejected.
- The existing W5D3 test suite passed: **3 passed, 1 dependency deprecation warning**.
- The warning originates from the installed ChromaDB telemetry dependency; it is not a test failure.
- Additional tests should be added only where a specific uncovered behavior warrants them.

## Overall outcome

Both CIA interactions were documented. Suggestions were assessed against the implementation rather than treated automatically as defects. Current test result: **3 passed**.
