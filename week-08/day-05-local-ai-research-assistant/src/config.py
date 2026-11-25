from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "documents"
CHROMA_DIR = PROJECT_DIR / "outputs" / "chroma_db"
MLFLOW_DIR = PROJECT_DIR / "outputs" / "mlflow"
EVIDENCE_DIR = PROJECT_DIR / "outputs" / "evidence"

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text:latest"

COLLECTION_NAME = "w8d5_research_documents"
EXPERIMENT_NAME = "W8D5-Local-AI-Research-Assistant"
TOP_K = 3
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
