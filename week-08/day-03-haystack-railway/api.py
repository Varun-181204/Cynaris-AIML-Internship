import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from haystack_pipeline import (
    DenseRetrievalUnavailableError,
    HaystackRetrievalSystem,
    resolve_embedding_model,
    resolve_ollama_timeout,
    resolve_ollama_url,
    sanitize_url_for_display,
)

system = HaystackRetrievalSystem()


def _refresh_system_config(target: HaystackRetrievalSystem) -> None:
    """Refresh environment-driven Ollama settings at application startup."""
    target.ollama_url = resolve_ollama_url()
    target.embedding_model = resolve_embedding_model()
    target.ollama_timeout = resolve_ollama_timeout()


@asynccontextmanager
async def lifespan(app: FastAPI):
    active_system: HaystackRetrievalSystem = getattr(app.state, "system", system)
    _refresh_system_config(active_system)
    active_system.index_pdfs(require_dense=False)
    active_system.build_retrieval_pipelines()
    app.state.system = active_system
    app.state.ready = active_system.bm25_ready
    yield
    app.state.ready = False


def create_app(
    retrieval_system: HaystackRetrievalSystem | None = None,
) -> FastAPI:
    application = FastAPI(
        title="Cynaris Haystack Retrieval API",
        description="BM25 and dense retrieval over five indexed PDF documents.",
        version="1.0.0",
        lifespan=lifespan,
    )
    if retrieval_system is not None:
        application.state.system = retrieval_system

    @application.get("/")
    def root():
        active_system: HaystackRetrievalSystem = getattr(
            application.state, "system", system
        )
        return {
            "service": "Cynaris Haystack Retrieval API",
            "version": "1.0.0",
            "ready": getattr(application.state, "ready", False),
            "bm25_ready": active_system.bm25_ready,
            "dense_ready": active_system.dense_ready,
            "indexed_chunks": active_system.indexed_documents,
            "endpoints": {
                "health": "/health",
                "query": "/query",
                "docs": "/docs",
            },
        }

    @application.get("/health")
    def health():
        active_system: HaystackRetrievalSystem = getattr(
            application.state, "system", system
        )
        ready = getattr(application.state, "ready", False)
        return {
            "status": "ok" if ready else "starting",
            "ready": ready,
            "bm25_ready": active_system.bm25_ready,
            "dense_ready": active_system.dense_ready,
            "indexed_chunks": active_system.indexed_documents,
            "embedding_model": active_system.embedding_model,
            "ollama_url": sanitize_url_for_display(active_system.ollama_url),
            "dense_error": active_system.dense_error,
        }

    @application.post("/query")
    def query(request: QueryRequest):
        if not getattr(application.state, "ready", False):
            raise HTTPException(status_code=503, detail="Pipeline is not ready.")

        active_system: HaystackRetrievalSystem = getattr(
            application.state, "system", system
        )
        try:
            result = active_system.run_query(request.method, request.question)
            return {
                "question": request.question,
                "method": request.method,
                "answer": result["reader"]["answer"],
                "source": result["reader"]["source"],
                "retrieved_sources": [
                    doc.meta.get("source_name", "Unknown")
                    for doc in result["documents"]
                ],
                "latency_ms": result["latency_ms"],
            }
        except DenseRetrievalUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail="Internal retrieval error."
            ) from exc

    return application


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    method: str = Field(default="dense", pattern="^(bm25|dense)$")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 3:
            raise ValueError(
                "Question must contain at least 3 non-whitespace characters."
            )
        return cleaned


app = create_app()
