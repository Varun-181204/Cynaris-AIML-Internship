"""W8D1: Production-ready API wrapper for the multi-document RAG system."""

import os
import sys
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Resolve project directories.
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

WEEK7_SOURCE = REPO_ROOT / "week-07" / "day-05-multi-document-rag" / "src"

if not WEEK7_SOURCE.exists():
    raise FileNotFoundError(f"Week 7 RAG source directory not found: {WEEK7_SOURCE}")

# Reuse the original RAG implementation without modifying Week 7.
sys.path.insert(0, str(WEEK7_SOURCE))
import rag_system  # noqa: E402


# Override configuration so Week 8 has isolated runtime data.
rag_system.ROOT = PROJECT_DIR
rag_system.DOCUMENTS_DIR = PROJECT_DIR / "data" / "documents"
rag_system.CHROMA_DIR = PROJECT_DIR / "outputs" / "chroma_db"
rag_system.MLFLOW_DIR = PROJECT_DIR / "outputs" / "mlruns"
rag_system.OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
rag_system.COLLECTION_NAME = "w8d1_documents"


app = FastAPI(
    title="Cynaris W8D1 Production RAG API",
    description="FastAPI service for the Week 8 multi-document RAG system.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question to answer using the indexed documents.",
    )
    use_crew: bool = False


@app.get("/health")
def health_check():
    """Check API and Ollama availability."""
    try:
        response = requests.get(
            f"{rag_system.OLLAMA_URL}/api/tags",
            timeout=5,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="Ollama service is unavailable.",
        ) from exc

    return {
        "status": "healthy",
        "ollama": "connected",
        "chat_model": rag_system.CHAT_MODEL,
        "embedding_model": rag_system.EMBED_MODEL,
    }


@app.post("/ingest")
def ingest_documents():
    """Index the Week 8 documents into ChromaDB."""
    try:
        chunk_count = rag_system.ingest_documents()
        return {
            "status": "success",
            "message": "Documents indexed successfully.",
            "chunks_indexed": chunk_count,
        }
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not generate embeddings using Ollama.",
        ) from exc
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/ask")
def ask_question(request: AskRequest):
    """Retrieve relevant chunks and generate an answer."""
    try:
        return rag_system.ask_question(
            request.question,
            use_crew=request.use_crew,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="An Ollama request failed while processing the question.",
        ) from exc
