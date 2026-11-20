# W7D5 Self-Review

## Strengths
- Implemented a multi-document ingestion and retrieval pipeline.
- Used LangGraph to organize the RAG workflow.
- Used Ollama for local embeddings and answer generation.
- Integrated CrewAI for answer review.
- Integrated Ragas Faithfulness evaluation.
- Used MLflow with SQLite tracking.
- Added unit tests for chunking and workflow construction.

## Test Results
- 7 unit tests passed.
- Live RAG query completed successfully.
- CrewAI review completed successfully.
- Ragas Faithfulness score: 1.0000.

## Areas for Improvement
- Evaluate retrieval quality with a larger question dataset.
- Add context precision and context recall metrics.
- Improve retrieval using hybrid search and reranking.
- Add API endpoints and structured logging.
- Benchmark latency and resource usage across models.

## Review Integrity
Mentor review feedback must be added after the actual review interactions. No mentor feedback is assumed or fabricated.

## Final Verification
- Final automated test suite: 7/7 passed.
- Python compilation: successful.
- CrewAI review: executed successfully.
- Ragas Faithfulness: 1.0000 on the recorded evaluation.
- A later live query returned the safe fallback "I do not know", highlighting the need for further retrieval tuning.
