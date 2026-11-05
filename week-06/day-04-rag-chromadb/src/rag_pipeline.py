"""W6D4: ChromaDB + Ollama retrieval-augmented generation pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import chromadb
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "chroma_db"
OUTPUT_PATH = ROOT / "outputs" / "rag_evidence.json"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:3b"

SAMPLE_DOCUMENTS = [
    ("RAG combines retrieval with generation to ground model answers in external knowledge.", "rag", "rag-basics"),
    ("An embedding is a dense numerical vector representing semantic information in text.", "embeddings", "embedding-basics"),
    ("Cosine similarity compares vector direction and is commonly used for semantic retrieval.", "similarity", "cosine-search"),
    ("ChromaDB is an open-source vector database that stores embeddings and metadata.", "vector-db", "chromadb"),
    ("A collection groups vectors, documents, identifiers, and metadata in ChromaDB.", "vector-db", "collections"),
    ("Metadata filtering narrows retrieval to records matching structured conditions.", "retrieval", "metadata-filter"),
    ("Top-k retrieval returns the k most relevant records for a query.", "retrieval", "top-k"),
    ("A chunk is a smaller segment of a source document used for retrieval.", "chunking", "text-chunks"),
    ("Recursive character splitting divides text while attempting to preserve natural boundaries.", "chunking", "recursive-splitter"),
    ("Ollama can run language models locally on a developer machine.", "ollama", "local-llm"),
    ("nomic-embed-text is an embedding model that creates vectors from text.", "embeddings", "nomic"),
    ("A RAG prompt should instruct the model to answer only from supplied context.", "rag", "grounded-prompt"),
    ("If retrieved context does not contain an answer, the system should say it is unavailable.", "guardrails", "abstention"),
    ("PDF loaders extract page text and can retain page metadata for citations.", "documents", "pdf-loader"),
    ("A text splitter creates overlapping chunks to reduce loss of context at boundaries.", "chunking", "overlap"),
    ("Persistent vector storage allows a collection to be reused between program runs.", "vector-db", "persistence"),
    ("Retrieval quality can be checked by manually inspecting returned documents.", "evaluation", "manual-check"),
    ("A hallucination is fluent model output that is unsupported by available evidence.", "guardrails", "hallucination"),
    ("RAG evaluation can consider relevance, groundedness, correctness, and response quality.", "evaluation", "rag-evaluation"),
    ("A source identifier and page number help trace an answer back to its document.", "documents", "source-tracing"),
]

class ChromaRAG:
    def __init__(self, persist_dir: Path = DB_PATH, embedding_model: str = EMBED_MODEL):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="w6d4_rag",
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: Iterable[Document], ids: list[str] | None = None) -> int:
        docs = list(documents)
        if not docs:
            return 0
        if ids is None:
            start = self.collection.count()
            ids = [f"doc-{start+i:04d}" for i in range(len(docs))]
        vectors = self.embeddings.embed_documents([d.page_content for d in docs])
        self.collection.upsert(
            ids=ids,
            documents=[d.page_content for d in docs],
            metadatas=[{str(k): str(v) for k, v in d.metadata.items()} for d in docs],
            embeddings=vectors,
        )
        return len(docs)

    def seed_demo_documents(self) -> int:
        docs = [
            Document(page_content=text, metadata={"category": category, "source": source})
            for text, category, source in SAMPLE_DOCUMENTS
        ]
        return self.add_documents(docs, [f"sample-{i+1:02d}" for i in range(len(docs))])

    def search(self, query: str, top_k: int = 3, where: dict | None = None) -> list[dict]:
        vector = self.embeddings.embed_query(query)
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        matches = []
        for i, doc in enumerate(result["documents"][0]):
            matches.append({
                "id": result["ids"][0][i],
                "document": doc,
                "metadata": result["metadatas"][0][i],
                "cosine_distance": result["distances"][0][i],
            })
        return matches

    def ingest_pdf(self, pdf_path: Path) -> int:
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        pages = PyPDFLoader(str(pdf_path)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        docs = [
            Document(
                page_content=chunk.page_content,
                metadata={
                    "category": "pdf",
                    "source": pdf_path.name,
                    "page": str(chunk.metadata.get("page", "unknown")),
                },
            )
            for chunk in chunks if chunk.page_content.strip()
        ]
        ids = [f"pdf-{pdf_path.stem}-{i:04d}" for i in range(len(docs))]
        return self.add_documents(docs, ids)

    def answer(self, question: str, top_k: int = 3) -> tuple[str, list[dict]]:
        matches = self.search(question, top_k=top_k, where={"category": "pdf"})
        context = "\n\n".join(
            f"[Source: {m['metadata'].get('source')}, page: {m['metadata'].get('page')}]\n{m['document']}"
            for m in matches
        )
        if not context:
            return "Information not found in the indexed PDF.", matches
        prompt = (
            "Answer the question using only the retrieved context below. "
            "Do not add outside facts. If the context does not contain the answer, "
            "say: Information not found in the indexed PDF. Cite the source page.\n\n"
            f"Retrieved context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = ChatOllama(model=CHAT_MODEL, temperature=0).invoke(prompt)
        return str(response.content), matches


def run_all(pdf_path: Path, question: str) -> dict:
    rag = ChromaRAG()
    demo_added = rag.seed_demo_documents()
    demo_results = rag.search("How does cosine similarity support semantic search?", top_k=3)
    filtered_results = rag.search(
        "How does retrieval work?", top_k=5, where={"category": "retrieval"}
    )
    pdf_chunks = rag.ingest_pdf(pdf_path)
    answer, pdf_results = rag.answer(question, top_k=3)
    evidence = {
        "collection": "w6d4_rag",
        "distance_metric": "cosine",
        "embedding_model": EMBED_MODEL,
        "embedding_dimensions": len(rag.embeddings.embed_query("dimension check")),
        "sample_documents_added": demo_added,
        "sample_search_results": demo_results,
        "metadata_filtered_results": filtered_results,
        "pdf": str(pdf_path),
        "pdf_chunks_added": pdf_chunks,
        "pdf_top_3_retrieval": pdf_results,
        "question": question,
        "answer": answer,
        "total_collection_count": rag.collection.count(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    print(f"\nEvidence saved to: {OUTPUT_PATH}")
    return evidence


def main():
    parser = argparse.ArgumentParser(description="Run the W6D4 ChromaDB + Ollama RAG demo.")
    parser.add_argument("--pdf", default=str(ROOT / "data" / "Module-3-PE.pdf"))
    parser.add_argument(
        "--question",
        default="What are the four types of generative AI guardrails, and what does each one do?",
    )
    args = parser.parse_args()
    run_all(Path(args.pdf), args.question)


if __name__ == "__main__":
    main()
