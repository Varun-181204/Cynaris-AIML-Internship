"""W5D5: Local PDF Q&A bot using Ollama embeddings, ChromaDB, and Ollama chat."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import chromadb
from pypdf import PdfReader


DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")

INSUFFICIENT_CONTEXT = (
    "I couldn't find enough information in the indexed document to answer that question."
)

SYSTEM_PROMPT = """You are a careful document question-answering assistant.
Answer the user's question using only the supplied document context.
Do not invent facts or rely on outside knowledge.
If the context does not contain enough information, clearly say so.
Keep the answer concise and mention relevant source page numbers when possible."""


class LocalQABotError(RuntimeError):
    """Raised when document indexing or local inference fails."""


def _post_json(base_url: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send a JSON POST request to the local Ollama API."""
    url = f"{base_url.rstrip('/')}{endpoint}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LocalQABotError(
            f"Could not reach Ollama at {base_url}. Check that Ollama is running. "
            f"Details: {exc}"
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LocalQABotError(f"Ollama returned invalid JSON: {exc}") from exc

    if not isinstance(result, dict):
        raise LocalQABotError("Ollama returned an unexpected response format.")

    return result


def embed_texts(
    texts: list[str],
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_EMBED_MODEL,
) -> list[list[float]]:
    """Generate local embeddings with Ollama's /api/embed endpoint."""
    if (
        not isinstance(texts, list)
        or not texts
        or any(not isinstance(text, str) or not text.strip() for text in texts)
    ):
        raise ValueError("texts must contain non-empty strings.")

    response = _post_json(
        base_url,
        "/api/embed",
        {"model": model, "input": texts},
    )
    embeddings = response.get("embeddings")

    if (
        not isinstance(embeddings, list)
        or len(embeddings) != len(texts)
        or any(not isinstance(vector, list) or not vector for vector in embeddings)
    ):
        raise LocalQABotError("Ollama returned missing or invalid embeddings.")

    try:
        converted = [[float(value) for value in vector] for vector in embeddings]
    except (TypeError, ValueError) as exc:
        raise LocalQABotError("Ollama returned non-numeric embedding values.") from exc

    if any(not math.isfinite(value) for vector in converted for value in vector):
        raise LocalQABotError("Ollama returned non-finite embedding values.")

    return converted


def extract_pdf_chunks(
    pdf_path: str | Path,
    *,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    """Extract PDF text and split it into overlapping chunks with page metadata."""
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    try:
        reader = PdfReader(str(path))
        chunks: list[dict[str, Any]] = []
        step = chunk_size - overlap

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            for start in range(0, len(text), step):
                content = text[start : start + chunk_size].strip()
                if content:
                    chunks.append(
                        {
                            "text": content,
                            "page": page_number,
                            "source": path.name,
                        }
                    )

        if not chunks:
            raise LocalQABotError(f"No extractable text found in PDF: {path}")

        return chunks
    except LocalQABotError:
        raise
    except Exception as exc:
        raise LocalQABotError(f"Could not read PDF {path}: {exc}") from exc


class LocalQABot:
    """Index a PDF and answer questions using local semantic retrieval and generation."""

    def __init__(
        self,
        *,
        storage_dir: str | Path | None = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        collection_name: str = "w5d5_local_qa",
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model

        if storage_dir is None:
            storage_dir = Path(__file__).resolve().parents[1] / "data" / "chroma"

        storage_path = Path(storage_dir).resolve()
        storage_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(storage_path))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_pdf(self, pdf_path: str | Path, *, reset: bool = True) -> int:
        """Embed and index PDF chunks. Reset replaces this bot's collection contents."""
        chunks = extract_pdf_chunks(pdf_path)

        # Generate embeddings before changing the collection, avoiding a partial reset
        # if Ollama is unavailable.
        embeddings = embed_texts(
            [chunk["text"] for chunk in chunks],
            base_url=self.ollama_url,
            model=self.embed_model,
        )

        if reset:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        self.collection.upsert(
            ids=[f"{chunks[i]['source']}-p{chunks[i]['page']}-c{i}" for i in range(len(chunks))],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                {"source": chunk["source"], "page": chunk["page"]}
                for chunk in chunks
            ],
            embeddings=embeddings,
        )
        return len(chunks)

    def retrieve(self, question: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the most relevant indexed chunks for a question."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if self.collection.count() == 0:
            raise LocalQABotError("No documents are indexed. Call index_pdf() first.")

        query_embedding = embed_texts(
            [question.strip()],
            base_url=self.ollama_url,
            model=self.embed_model,
        )[0]

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            {
                "text": document,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page"),
                "distance": distance,
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
            if document
        ]

    def answer(self, question: str, *, top_k: int = 3) -> dict[str, Any]:
        """Retrieve context and generate an answer grounded in the indexed document."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be empty.")

        matches = self.retrieve(question, top_k=top_k)
        if not matches:
            return {
                "question": question,
                "answer": INSUFFICIENT_CONTEXT,
                "sources": [],
                "model": self.chat_model,
            }

        context = "\n\n".join(
            f"[Source: {item['source']}, page {item['page']}]\n{item['text']}"
            for item in matches
        )

        response = _post_json(
            self.ollama_url,
            "/api/chat",
            {
                "model": self.chat_model,
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Document context:\n{context}\n\n"
                            f"Question: {question.strip()}\n\n"
                            "Answer only from the context. If it is insufficient, "
                            "say that the document does not provide enough information."
                        ),
                    },
                ],
            },
        )

        message = response.get("message", {})
        answer = message.get("content") if isinstance(message, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise LocalQABotError("Ollama returned an empty chat answer.")

        return {
            "question": question.strip(),
            "answer": answer.strip(),
            "sources": [
                {
                    "source": item["source"],
                    "page": item["page"],
                    "distance": item["distance"],
                }
                for item in matches
            ],
            "model": self.chat_model,
        }


def main() -> None:
    """Run two local PDF Q&A examples and save their evidence."""
    project_dir = Path(__file__).resolve().parents[1]
    pdf_path = project_dir / "sample_data" / "vector_databases_guide.pdf"
    output_path = project_dir / "outputs" / "qa_evidence.json"

    bot = LocalQABot()
    chunk_count = bot.index_pdf(pdf_path)

    questions = [
        "What is a vector database, and how does semantic search use it?",
        "How does retrieval-augmented generation (RAG) work?",
    ]
    results = [bot.answer(question) for question in questions]

    evidence = {
        "indexed_pdf": pdf_path.name,
        "indexed_chunks": chunk_count,
        "embedding_model": bot.embed_model,
        "chat_model": bot.chat_model,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print(json.dumps(evidence, indent=2))
    print(f"\nEvidence saved to: {output_path}")


if __name__ == "__main__":
    main()
