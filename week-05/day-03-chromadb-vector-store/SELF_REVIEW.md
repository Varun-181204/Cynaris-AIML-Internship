# W5D3 Self-Review

## Completed
- Created a ChromaDB collection containing 20 learning documents.
- Generated embeddings with the local Ollama embedding model `nomic-embed-text`.
- Configured cosine distance and implemented top-3 semantic retrieval.
- Implemented category-based metadata filtering.
- Added PDF extraction, overlapping text chunking, embedding, and retrieval for a RAG workflow.
- Passed retrieved context to a local chat model and saved JSON evidence.
- Added automated tests for PDF chunking and retrieval behavior.

## Validation
- Vector-store script successfully indexed 20 documents and saved search evidence.
- PDF RAG script successfully retrieved three chunks and generated a context-based answer.
- Previously executed PDF tests: 3 passed. Re-run the complete test suite before committing.

## Limitations and improvements
- ChromaDB uses an in-memory client; data does not persist between independent runs.
- Ollama must be installed and running locally, and the required models must be available.
- A generated answer can still be incorrect despite context restrictions. Source attribution and manual evidence review are important.
- Future improvements include persistent storage, stronger relevance evaluation, and tests for empty or irrelevant model responses.

## Final status
Implementation is in place. Confirm the latest complete test results and record actual CIA review feedback before marking the day fully verified.