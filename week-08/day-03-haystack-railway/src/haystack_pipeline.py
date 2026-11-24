import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from haystack import Document, Pipeline, component
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder,
    OllamaTextEmbedder,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
OUTPUT_DIR = BASE_DIR / "outputs"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
OLLAMA_URL = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_TIMEOUT = 30
TOP_K = 3


class DenseRetrievalUnavailableError(RuntimeError):
    """Raised when dense retrieval cannot run because Ollama embeddings are unavailable."""


def resolve_ollama_url(url: str | None = None) -> str:
    """Resolve and normalize the Ollama base URL from an argument or environment."""
    candidate = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL) if url is None else url
    normalized = candidate.strip().rstrip("/")
    return normalized or DEFAULT_OLLAMA_URL


def resolve_embedding_model(model: str | None = None) -> str:
    """Resolve the embedding model name from an argument or environment."""
    candidate = (
        os.getenv("OLLAMA_EMBEDDING_MODEL", EMBEDDING_MODEL) if model is None else model
    )
    normalized = candidate.strip()
    return normalized or EMBEDDING_MODEL


def resolve_ollama_timeout(timeout: int | None = None) -> int:
    """Resolve the Ollama HTTP timeout in seconds."""
    if timeout is not None:
        return max(1, int(timeout))
    raw = os.getenv("OLLAMA_TIMEOUT", str(DEFAULT_OLLAMA_TIMEOUT)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT


def sanitize_url_for_display(url: str) -> str:
    """Strip embedded credentials from a URL before returning or logging it."""
    parts = urlsplit(url)
    if not parts.username and not parts.password:
        return url
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


QUESTIONS = [
    {
        "id": 1,
        "question": "How are components connected in a Haystack pipeline?",
        "relevant_sources": ["haystack_pipeline_architecture.pdf"],
    },
    {
        "id": 2,
        "question": "What is the role of a document store in a RAG workflow?",
        "relevant_sources": ["haystack_pipeline_architecture.pdf"],
    },
    {
        "id": 3,
        "question": "How does BM25 calculate document relevance?",
        "relevant_sources": ["retrieval_methods_and_evaluation.pdf"],
    },
    {
        "id": 4,
        "question": "How does dense retrieval use embeddings to find documents?",
        "relevant_sources": ["retrieval_methods_and_evaluation.pdf"],
    },
    {
        "id": 5,
        "question": "What is few-shot prompting and how does it guide an LLM?",
        "relevant_sources": ["prompt_engineering.pdf"],
    },
    {
        "id": 6,
        "question": "What makes an effective prompt for a language model?",
        "relevant_sources": ["prompt_engineering.pdf"],
    },
    {
        "id": 7,
        "question": "What information does a model card provide about a machine learning model?",
        "relevant_sources": ["responsible_ai_model_card.pdf"],
    },
    {
        "id": 8,
        "question": "Why should model limitations and intended use be documented?",
        "relevant_sources": ["responsible_ai_model_card.pdf"],
    },
    {
        "id": 9,
        "question": "What is a vector database and how does it store information?",
        "relevant_sources": ["vector_databases_guide.pdf"],
    },
    {
        "id": 10,
        "question": "How does similarity search find related vectors?",
        "relevant_sources": ["vector_databases_guide.pdf"],
    },
]


@component
class ExtractiveReader:
    """Extract a query-relevant sentence from the highest-ranked document."""

    @component.output_types(answer=str, source=str, documents=list[Document])
    def run(self, query: str, documents: list[Document]):
        if not documents:
            return {
                "answer": "No relevant document was retrieved.",
                "source": "None",
                "documents": [],
            }

        query_terms = set(re.findall(r"\b\w+\b", query.lower()))
        best_document = documents[0]

        sentences = re.split(
            r"(?<=[.!?])\s+",
            best_document.content or "",
        )

        def sentence_score(sentence):
            terms = set(re.findall(r"\b\w+\b", sentence.lower()))
            return len(query_terms.intersection(terms))

        best_sentence = max(sentences, key=sentence_score, default="")

        return {
            "answer": best_sentence.strip() or "No readable answer found.",
            "source": best_document.meta.get("source_name", "Unknown"),
            "documents": documents,
        }


class HaystackRetrievalSystem:
    """Build and evaluate BM25 and dense Haystack retrieval."""

    def __init__(
        self,
        ollama_url: str | None = None,
        embedding_model: str | None = None,
        ollama_timeout: int | None = None,
    ):
        self.ollama_url = resolve_ollama_url(ollama_url)
        self.embedding_model = resolve_embedding_model(embedding_model)
        self.ollama_timeout = resolve_ollama_timeout(ollama_timeout)

        self.document_store = InMemoryDocumentStore(
            embedding_similarity_function="cosine"
        )

        self.indexing_pipeline = None
        self.bm25_pipeline = None
        self.dense_pipeline = None
        self.chunks: list[Document] = []
        self.indexed_documents = 0
        self.bm25_ready = False
        self.dense_ready = False
        self.dense_error: str | None = None

    def _create_document_embedder(self):
        return OllamaDocumentEmbedder(
            model=self.embedding_model,
            url=self.ollama_url,
            timeout=self.ollama_timeout,
        )

    def _create_text_embedder(self):
        return OllamaTextEmbedder(
            model=self.embedding_model,
            url=self.ollama_url,
            timeout=self.ollama_timeout,
        )

    def prepare_chunks(self) -> list[Document]:
        """Convert the five reference PDFs into split text chunks."""
        pdf_files = sorted(PDF_DIR.glob("*.pdf"))

        if len(pdf_files) != 5:
            raise ValueError(f"Expected exactly 5 PDFs, found {len(pdf_files)}.")

        converter = PyPDFToDocument()
        splitter = DocumentSplitter(
            split_by="word",
            split_length=180,
            split_overlap=30,
        )

        prep_pipeline = Pipeline()
        prep_pipeline.add_component("file_converter", converter)
        prep_pipeline.add_component("splitter", splitter)
        prep_pipeline.connect("file_converter.documents", "splitter.documents")

        result = prep_pipeline.run(
            {
                "file_converter": {
                    "sources": pdf_files,
                    "meta": [{"source_name": pdf.name} for pdf in pdf_files],
                }
            }
        )
        chunks = result["splitter"]["documents"]
        if not chunks:
            raise RuntimeError("No document chunks were produced from the PDFs.")

        self.chunks = chunks
        return chunks

    def build_indexing_pipeline(self):
        converter = PyPDFToDocument()

        splitter = DocumentSplitter(
            split_by="word",
            split_length=180,
            split_overlap=30,
        )

        embedder = self._create_document_embedder()

        writer = DocumentWriter(
            document_store=self.document_store,
            policy=DuplicatePolicy.OVERWRITE,
        )

        pipeline = Pipeline()

        pipeline.add_component("file_converter", converter)
        pipeline.add_component("splitter", splitter)
        pipeline.add_component("embedder", embedder)
        pipeline.add_component("document_writer", writer)

        pipeline.connect("file_converter.documents", "splitter.documents")
        pipeline.connect("splitter.documents", "embedder.documents")
        pipeline.connect("embedder.documents", "document_writer.documents")

        self.indexing_pipeline = pipeline
        return pipeline

    def index_bm25_only(self) -> int:
        """Write split PDF chunks to the in-memory store for BM25 without Ollama."""
        chunks = self.chunks or self.prepare_chunks()
        writer = DocumentWriter(
            document_store=self.document_store,
            policy=DuplicatePolicy.OVERWRITE,
        )
        result = writer.run(documents=chunks)
        written = result["documents_written"]
        if written == 0:
            raise RuntimeError("No document chunks were indexed.")

        self.indexed_documents = self.document_store.count_documents()
        self.bm25_ready = True
        return written

    def index_dense_embeddings(self) -> int:
        """Compute Ollama embeddings for indexed chunks and store them."""
        chunks = self.chunks or self.prepare_chunks()
        embedder = self._create_document_embedder()
        writer = DocumentWriter(
            document_store=self.document_store,
            policy=DuplicatePolicy.OVERWRITE,
        )

        embedded = embedder.run(documents=chunks)["documents"]
        result = writer.run(documents=embedded)
        written = result["documents_written"]
        if written == 0:
            raise RuntimeError("No embedded document chunks were indexed.")

        self.indexed_documents = self.document_store.count_documents()
        self.bm25_ready = True
        self.dense_ready = True
        self.dense_error = None
        return written

    def index_pdfs(self, require_dense: bool = True):
        """Index the five PDFs for BM25 and (optionally) dense retrieval."""
        print("\nStarting PDF conversion and indexing...")
        self.prepare_chunks()
        self.index_bm25_only()
        print("Indexed document chunks (BM25):", self.indexed_documents)

        try:
            written = self.index_dense_embeddings()
            print("Indexed document chunks (Dense):", written)
            return {"document_writer": {"documents_written": written}}
        except Exception as exc:
            self.dense_ready = False
            self.dense_error = str(exc)
            if require_dense:
                raise
            logger.warning(
                "Dense indexing skipped during startup (%s); BM25 remains active.",
                exc,
            )
            return {"document_writer": {"documents_written": self.indexed_documents}}

    def build_retrieval_pipelines(self):
        bm25_retriever = InMemoryBM25Retriever(
            document_store=self.document_store,
            top_k=TOP_K,
        )

        dense_retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=TOP_K,
        )

        text_embedder = self._create_text_embedder()

        # BM25 retrieval and Reader
        bm25_pipeline = Pipeline()
        bm25_pipeline.add_component("retriever", bm25_retriever)
        bm25_pipeline.add_component("reader", ExtractiveReader())
        bm25_pipeline.connect("retriever.documents", "reader.documents")

        # Dense retrieval and Reader
        dense_pipeline = Pipeline()
        dense_pipeline.add_component("text_embedder", text_embedder)
        dense_pipeline.add_component("retriever", dense_retriever)
        dense_pipeline.add_component("reader", ExtractiveReader())

        dense_pipeline.connect(
            "text_embedder.embedding",
            "retriever.query_embedding",
        )
        dense_pipeline.connect(
            "retriever.documents",
            "reader.documents",
        )

        self.bm25_pipeline = bm25_pipeline
        self.dense_pipeline = dense_pipeline

    def ensure_dense_ready(self) -> None:
        """Lazily index dense embeddings if Ollama was unavailable at startup."""
        if self.dense_ready:
            return
        try:
            self.index_dense_embeddings()
        except Exception as exc:
            self.dense_ready = False
            self.dense_error = str(exc)
            safe_url = sanitize_url_for_display(self.ollama_url)
            raise DenseRetrievalUnavailableError(
                f"Dense retrieval is unavailable because Ollama at {safe_url} "
                f"with model '{self.embedding_model}' could not be reached: {exc}"
            ) from exc

    @staticmethod
    def unique_source_results(documents):
        """Keep the first highest-ranked chunk from each distinct PDF source."""
        unique_documents = []
        seen_sources = set()

        for document in documents:
            source = document.meta.get("source_name", "Unknown")
            if source not in seen_sources:
                seen_sources.add(source)
                unique_documents.append(document)

            if len(unique_documents) == TOP_K:
                break

        return unique_documents

    def run_query(self, method, question):
        start = time.perf_counter()

        if method == "bm25":
            if not self.bm25_pipeline:
                raise RuntimeError("BM25 retrieval pipeline is not initialized.")
            result = self.bm25_pipeline.run(
                {
                    "retriever": {"query": question},
                    "reader": {"query": question},
                }
            )

        elif method == "dense":
            if not self.dense_pipeline:
                raise RuntimeError("Dense retrieval pipeline is not initialized.")
            self.ensure_dense_ready()
            try:
                result = self.dense_pipeline.run(
                    {
                        "text_embedder": {"text": question},
                        "reader": {"query": question},
                    }
                )
            except DenseRetrievalUnavailableError:
                raise
            except Exception as exc:
                safe_url = sanitize_url_for_display(self.ollama_url)
                raise DenseRetrievalUnavailableError(
                    f"Dense query failed against Ollama at {safe_url}: {exc}"
                ) from exc

        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        documents = result["reader"]["documents"]
        unique_documents = self.unique_source_results(documents)

        return {
            "documents": unique_documents,
            "reader": result["reader"],
            "latency_ms": round(elapsed_ms, 2),
        }

    def evaluate(self):
        all_results = {}

        # Warm up Ollama before measuring dense-query latency.
        self.run_query("dense", QUESTIONS[0]["question"])

        for method in ("bm25", "dense"):
            print(f"\nEvaluating {method.upper()} retrieval...")

            method_results = []
            total_precision = 0.0

            for item in QUESTIONS:
                result = self.run_query(
                    method,
                    item["question"],
                )

                retrieved = result["documents"]
                retrieved_sources = [
                    doc.meta.get("source_name", "Unknown") for doc in retrieved
                ]

                relevant = set(item["relevant_sources"])

                relevant_hits = sum(source in relevant for source in retrieved_sources)

                precision = relevant_hits / TOP_K
                total_precision += precision

                record = {
                    "question_id": item["id"],
                    "question": item["question"],
                    "expected_sources": item["relevant_sources"],
                    "retrieved_sources": retrieved_sources,
                    "relevant_hits": relevant_hits,
                    "precision_at_3": round(precision, 4),
                    "latency_ms": result["latency_ms"],
                    "reader_answer": result["reader"]["answer"],
                    "reader_source": result["reader"]["source"],
                }

                method_results.append(record)

                print(
                    f"Q{item['id']:02d} | "
                    f"Precision@3: {precision:.2f} | "
                    f"Sources: {retrieved_sources}"
                )

            all_results[method] = {
                "queries": method_results,
                "mean_precision_at_3": round(total_precision / len(QUESTIONS), 4),
                "mean_latency_ms": round(
                    sum(r["latency_ms"] for r in method_results) / len(method_results),
                    2,
                ),
            }

        bm25_score = all_results["bm25"]["mean_precision_at_3"]
        dense_score = all_results["dense"]["mean_precision_at_3"]

        comparison = {
            "bm25_precision_at_3": bm25_score,
            "dense_precision_at_3": dense_score,
            "difference": round(dense_score - bm25_score, 4),
            "higher_precision": (
                "Dense"
                if dense_score > bm25_score
                else "BM25"
                if bm25_score > dense_score
                else "Tie"
            ),
        }

        evidence = {
            "assignment": "Cynaris W8D3 - Haystack BM25 and Dense Retrieval",
            "haystack_version": "3.5.0",
            "embedding_model": self.embedding_model,
            "pdf_count": 5,
            "indexed_chunks": self.indexed_documents,
            "top_k": TOP_K,
            "evaluation_method": (
                "Source-level relevance judgments with top-K results "
                "deduplicated by PDF filename. Precision@3 uses a fixed "
                "denominator of 3. These are proxy judgments, not "
                "independent passage-level annotations."
            ),
            "comparison": comparison,
            "results": all_results,
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = OUTPUT_DIR / "retrieval_evidence.json"
        output_path.write_text(
            json.dumps(evidence, indent=4),
            encoding="utf-8",
        )

        print("\n========== FINAL COMPARISON ==========")
        print("BM25 Precision@3:", bm25_score)
        print("Dense Precision@3:", dense_score)
        print("Higher precision:", comparison["higher_precision"])
        print("Evidence saved:", output_path)

        return evidence


def main():
    system = HaystackRetrievalSystem()

    system.index_pdfs()
    system.build_retrieval_pipelines()
    system.evaluate()


if __name__ == "__main__":
    main()
