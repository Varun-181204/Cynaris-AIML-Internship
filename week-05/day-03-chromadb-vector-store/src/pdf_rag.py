"""W5D3: PDF retrieval-augmented generation using ChromaDB and Ollama."""
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAMPLE_PDF = ROOT / "sample_data" / "vector_databases_guide.pdf"
OUTPUT_DIR = ROOT / "outputs"

# Allow importing the vector store when this file is run directly.
sys.path.insert(0, str(HERE))
from vector_store import create_collection, ollama_embed, similarity_search


def extract_pdf_chunks(pdf_path, chunk_size=500, overlap=80):
    """Extract PDF text and split it into overlapping chunks with page metadata."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")

    reader = PdfReader(str(pdf_path))
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            content = text[start:start + chunk_size].strip()
            if content:
                chunks.append({
                    "id": f"pdf-p{page_number}-c{len(chunks) + 1}",
                    "text": content,
                    "source": Path(pdf_path).name,
                    "page": page_number,
                })
            if start + chunk_size >= len(text):
                break
            start += chunk_size - overlap
    if not chunks:
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")
    return chunks


def index_pdf(collection, pdf_path):
    """Add extracted PDF chunks to a ChromaDB collection."""
    chunks = extract_pdf_chunks(pdf_path)
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {"source": chunk["source"], "page": chunk["page"], "category": "pdf"}
            for chunk in chunks
        ],
        embeddings=ollama_embed([chunk["text"] for chunk in chunks]),
    )
    return chunks


def ask_ollama(question, context, model="llama3.2:3b"):
    """Ask Ollama to answer from retrieved context."""
    import os
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied document context. "
                    "If the answer is not in the context, say the context is insufficient. "
                    "Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": f"Document context:\n{context}\n\nQuestion: {question}",
            },
        ],
    }
    request = Request(
        f"{os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama chat request failed: {exc}") from exc
    answer = result.get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer")
    return answer


def run_rag(pdf_path=SAMPLE_PDF, question="How does retrieval-augmented generation work?"):
    """Index the PDF, retrieve the top three chunks, and answer with Ollama."""
    collection = create_collection(name="w5d3_pdf_rag")
    chunks = index_pdf(collection, pdf_path)
    search = similarity_search(collection, question, n_results=3, where={"category": "pdf"})
    documents = search["documents"][0]
    metadata = search["metadatas"][0]
    distances = search["distances"][0]
    context = "\n\n".join(
        f"[Source: {item['source']}, page {item['page']}]\n{doc}"
        for doc, item in zip(documents, metadata)
    )
    answer = ask_ollama(question, context)

    evidence = {
        "pdf": Path(pdf_path).name,
        "question": question,
        "chunks_indexed": len(chunks),
        "retrieved_top_3": [
            {"text": doc, "metadata": item, "cosine_distance": distance}
            for doc, item, distance in zip(documents, metadata, distances)
        ],
        "answer": answer,
        "verification": {
            "three_chunks_retrieved": len(documents) == 3,
            "answer_nonempty": bool(answer.strip()),
            "context_included_in_prompt": bool(context),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "pdf_rag_evidence.json"
    path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Top chunks retrieved: {len(documents)}")
    print(f"\nQuestion: {question}\nAnswer: {answer}")
    print(f"\nEvidence saved: {path}")
    return evidence


if __name__ == "__main__":
    run_rag()
