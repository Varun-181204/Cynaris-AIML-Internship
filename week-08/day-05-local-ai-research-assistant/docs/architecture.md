# W8D5 Architecture

## Workflow

Research documents
-> Text chunking
-> Ollama embeddings
-> ChromaDB storage
-> LangGraph retrieval
-> Ollama answer generation
-> CrewAI answer review
-> Ragas evaluation
-> MLflow tracking
-> Evidence outputs

## Modules

- config.py: application settings and paths.
- ingestion.py: document processing and vector indexing.
- workflow.py: LangGraph retrieval and generation.
- agents.py: CrewAI research reviewer.
- research_assistant.py: application orchestration.
- tracking.py: MLflow tracking.
- evaluation.py: Ragas Faithfulness evaluation.
- main.py: command-line interface.

## Design Decisions

- Local Ollama models avoid hosted inference dependencies.
- ChromaDB stores persistent document embeddings.
- LangGraph separates retrieval and generation.
- CrewAI provides a separate quality-review step.
- MLflow uses a dedicated SQLite database.
- Ragas evaluates answer faithfulness.

## Data Isolation
W8D5 uses its own documents, ChromaDB collection, MLflow database, and evidence directory. W7D5 implementation files remain unchanged.
