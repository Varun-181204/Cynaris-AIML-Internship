# Architecture

Question -> LangGraph -> Retrieval -> ChromaDB
                         |
                         v
                  Retrieved Context
                         |
                         v
                  Ollama LLM
                         |
                         v
                    RAG Answer
                         |
              +----------+----------+
              |                     |
              v                     v
          CrewAI Review        Ragas Evaluation
                                    |
                                    v
                               MLflow Tracking

Document ingestion:
Text documents -> Chunking -> Ollama Embeddings -> ChromaDB
