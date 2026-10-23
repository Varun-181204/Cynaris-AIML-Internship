"""W5D3: ChromaDB vector storage with local Ollama embeddings."""

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import chromadb

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

DOCUMENTS = [
    ("d01", "Python is a programming language used for automation and data science.", "programming"),
    ("d02", "JavaScript adds interactivity to web pages and applications.", "programming"),
    ("d03", "SQL is used to query and manage relational databases.", "data"),
    ("d04", "Pandas provides tools for data cleaning and analysis.", "data"),
    ("d05", "Machine learning models learn patterns from training data.", "machine-learning"),
    ("d06", "Supervised learning uses labelled examples to train predictive models.", "machine-learning"),
    ("d07", "Unsupervised learning discovers structure in unlabelled data.", "machine-learning"),
    ("d08", "Neural networks contain layers of interconnected artificial neurons.", "deep-learning"),
    ("d09", "Deep learning is useful for image, audio, and language tasks.", "deep-learning"),
    ("d10", "Natural language processing helps computers understand human language.", "language-ai"),
    ("d11", "Tokenization splits text into smaller units for language processing.", "language-ai"),
    ("d12", "Text embeddings represent semantic meaning as numerical vectors.", "vectors"),
    ("d13", "Cosine similarity compares the direction of two vectors.", "vectors"),
    ("d14", "Semantic search retrieves content based on meaning rather than exact keywords.", "search"),
    ("d15", "ChromaDB stores embeddings and supports vector similarity queries.", "vector-database"),
    ("d16", "Metadata filters restrict vector search results to matching attributes.", "vector-database"),
    ("d17", "Retrieval augmented generation supplies retrieved context to a language model.", "generative-ai"),
    ("d18", "Large language models generate text from prompts and context.", "llm"),
    ("d19", "PDF text extraction makes document content available for retrieval.", "document-processing"),
    ("d20", "Chunking divides long documents into smaller searchable text passages.", "document-processing"),
]


def ollama_embed(texts, model=EMBED_MODEL):
    """Create embeddings locally using Ollama's /api/embed endpoint."""
    if isinstance(texts, str):
        texts = [texts]
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts must contain non-empty strings")

    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = Request(
        f"{OLLAMA_URL.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError("Ollama returned an invalid number of embeddings")
    if any(not isinstance(vector, list) or not vector for vector in embeddings):
        raise RuntimeError("Ollama returned an invalid embedding vector")
    return embeddings


def create_collection(client=None, name="w5d3_learning_docs", reset=True):
    """Create a cosine-distance collection and populate 20 learning documents."""
    client = client or chromadb.Client()
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [item[0] for item in DOCUMENTS]
    documents = [item[1] for item in DOCUMENTS]
    metadatas = [{"category": item[2]} for item in DOCUMENTS]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=ollama_embed(documents),
    )
    return collection


def similarity_search(collection, query, n_results=3, where=None):
    """Return nearest documents using cosine distance, optionally filtered by metadata."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if n_results < 1:
        raise ValueError("n_results must be at least 1")

    arguments = {
        "query_embeddings": ollama_embed([query]),
        "n_results": min(n_results, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        arguments["where"] = where
    return collection.query(**arguments)


def save_evidence(collection):
    """Save search and metadata-filter evidence as JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    searches = {
        "cosine_similarity_search": similarity_search(
            collection, "How do vector databases find text by meaning?", 3
        ),
        "metadata_filtered_search": similarity_search(
            collection, "learning algorithms and models", 3,
            where={"category": "machine-learning"},
        ),
    }
    path = OUTPUT_DIR / "vector_search_results.json"
    path.write_text(json.dumps(searches, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    collection = create_collection()
    evidence_path = save_evidence(collection)
    print(f"Collection: {collection.name}")
    print(f"Documents stored: {collection.count()}")
    print(f"Evidence saved: {evidence_path}")

    result = similarity_search(collection, "How do computers understand human language?", 3)
    print("\nTop 3 results:")
    for document, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        print(f"- [{metadata['category']}] {document} (cosine distance: {distance:.4f})")
