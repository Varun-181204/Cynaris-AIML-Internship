import json
import logging
import statistics
import time
import urllib.request
from pathlib import Path

import chromadb

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "texts"
OUTPUT_DIR = BASE_DIR / "outputs"
CHROMA_DIR = OUTPUT_DIR / "chroma_db"
EVIDENCE_FILE = OUTPUT_DIR / "retrieval_evidence.json"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen2.5:3b"

QUESTIONS = [
    {"question": "What is LlamaIndex used for?", "expected_source": "llamaindex.txt"},
    {"question": "What are the main stages of a typical LlamaIndex workflow?", "expected_source": "llamaindex.txt"},
    {"question": "What does Retrieval-Augmented Generation combine?", "expected_source": "rag_fundamentals.txt"},
    {"question": "What are the main stages of a basic RAG pipeline?", "expected_source": "rag_fundamentals.txt"},
    {"question": "What information can a ChromaDB record contain?", "expected_source": "chromadb.txt"},
    {"question": "How does PersistentClient help a ChromaDB application?", "expected_source": "chromadb.txt"},
    {"question": "What are text embeddings?", "expected_source": "embeddings.txt"},
    {"question": "Why should the same embedding model be used for documents and queries?", "expected_source": "embeddings.txt"},
    {"question": "What does Precision at k measure?", "expected_source": "evaluation.txt"},
    {"question": "What conditions make a comparison between two RAG systems fair?", "expected_source": "evaluation.txt"},
]


def configure_models():
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_URL,
    )
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        request_timeout=180.0,
        temperature=0,
        additional_kwargs={"num_predict": 120},
    )
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    # Warm up the shared embedding model before indexing and measurement.
    logger.info("Warming up Ollama embedding model...")
    Settings.embed_model.get_text_embedding("W7D3 embedding warm-up")


def build_in_memory_index(documents):
    return VectorStoreIndex.from_documents(documents)


def build_chroma_index(documents):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection_name = "w7d3_llamaindex"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )


def run_queries(label, index):
    """Measure end-to-end query latency using time.perf_counter()."""
    logger.info("Starting %s query benchmark...", label)
    query_engine = index.as_query_engine(similarity_top_k=2)
    results = []

    for item in QUESTIONS:
        started = time.perf_counter()
        response = query_engine.query(item["question"])
        latency_ms = (time.perf_counter() - started) * 1000

        source_files = sorted({
            Path(node.node.metadata.get("file_path", node.node.metadata.get("file_name", "unknown"))).name
            for node in response.source_nodes
        })

        results.append({
            "question": item["question"],
            "expected_source": item["expected_source"],
            "retrieved_sources": source_files,
            "expected_source_retrieved": item["expected_source"] in source_files,
            "answer": str(response),
            "latency_ms": round(latency_ms, 2),
        })

        print(f"[{label}] {len(results)}/{len(QUESTIONS)} "
              f"({latency_ms:.0f} ms) {item['question']}")

    latencies = [item["latency_ms"] for item in results]
    return {
        "engine": label,
        "query_count": len(results),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(
            statistics.quantiles(latencies, n=100, method="inclusive")[94], 2
        ),
        "stdev_latency_ms": round(statistics.stdev(latencies), 2),
        "results": results,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Fail early with a clear message if Ollama or either model is unavailable.
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as response:
            available_models = {
                model.get("name")
                for model in json.load(response).get("models", [])
            }
    except Exception as exc:
        raise SystemExit(
            f"Ollama is unavailable at {OLLAMA_URL}. Start Ollama and retry. Details: {exc}"
        ) from exc

    missing_models = {EMBED_MODEL, LLM_MODEL} - available_models
    if missing_models:
        raise SystemExit(
            f"Required Ollama models are missing: {sorted(missing_models)}. "
            "Download them with ollama pull."
        )

    configure_models()

    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".txt"],
        encoding="utf-8",
    ).load_data()

    documents = [document for document in documents if document.text.strip()]

    if not documents:
        raise RuntimeError(f"No non-empty text documents found in {DATA_DIR}")

    logger.info("Loaded %s non-empty source documents.", len(documents))

    print("\nBuilding in-memory VectorStoreIndex...")
    memory_index = build_in_memory_index(documents)

    print("\nBuilding persistent ChromaDB index...")
    chroma_index = build_chroma_index(documents)

    # Warm up both query engines before measuring the ten-question set.
    print("\nWarming up both query engines...")
    memory_index.as_query_engine(similarity_top_k=2).query(
        "What is a vector embedding?"
    )
    chroma_index.as_query_engine(similarity_top_k=2).query(
        "What is a vector embedding?"
    )

    print("\nRunning ten in-memory queries...")
    memory_results = run_queries("in_memory", memory_index)

    print("\nRunning ten ChromaDB queries...")
    chroma_results = run_queries("chromadb", chroma_index)

    evidence = {
        "project": "W7D3 LlamaIndex Document Indexing and Querying",
        "models": {
            "embedding": EMBED_MODEL,
            "llm": LLM_MODEL,
        },
        "source_document_count": len(documents),
        "chunk_size": Settings.chunk_size,
        "chunk_overlap": Settings.chunk_overlap,
        "similarity_top_k": 2,
        "timing_method": (
            "End-to-end query latency measured with time.perf_counter; "
            "both engines warmed up before the measured questions."
        ),
        "in_memory": memory_results,
        "chromadb": chroma_results,
    }

    EVIDENCE_FILE.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- COMPARISON ---")
    for result in (memory_results, chroma_results):
        print(
            f"{result['engine']}: mean={result['mean_latency_ms']} ms, "
            f"median={result['median_latency_ms']} ms, "
            f"p95={result['p95_latency_ms']} ms, "
            f"stddev={result['stdev_latency_ms']} ms"
        )
    print(f"Evidence saved: {EVIDENCE_FILE}")


if __name__ == "__main__":
    main()
