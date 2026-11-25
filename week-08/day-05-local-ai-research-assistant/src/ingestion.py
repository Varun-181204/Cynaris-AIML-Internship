"""Load local research documents, embed them with Ollama, and store them in ChromaDB."""

from pathlib import Path

import chromadb
import requests

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL,
    OLLAMA_URL,
)


def split_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    text = text.strip()
    return [
        text[start : start + chunk_size]
        for start in range(0, len(text), chunk_size - overlap)
        if text[start : start + chunk_size].strip()
    ]


def embed_text(text: str) -> list[float]:
    """Generate an embedding using the local Ollama embedding model."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    response.raise_for_status()
    embeddings = response.json().get("embeddings", [])
    if not embeddings:
        raise RuntimeError("Ollama returned no embedding.")
    return embeddings[0]


def get_collection():
    """Open the isolated W8D5 ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def ingest_documents(directory: Path = DATA_DIR) -> int:
    """Index all text documents and return the number of indexed chunks."""
    collection = get_collection()
    files = sorted(directory.glob("*.txt"))

    if not files:
        raise FileNotFoundError(f"No .txt research documents found in {directory}")

    ids, texts, embeddings, metadatas = [], [], [], []

    for file_path in files:
        chunks = split_text(file_path.read_text(encoding="utf-8"))
        for index, chunk in enumerate(chunks):
            ids.append(f"{file_path.stem}-{index}")
            texts.append(chunk)
            embeddings.append(embed_text(chunk))
            metadatas.append({"source": file_path.name, "chunk": index})

    # Replace existing IDs so repeated ingestion does not create duplicates.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)
