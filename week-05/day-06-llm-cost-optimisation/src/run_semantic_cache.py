"""Run a 20-pair semantic cache experiment using local Ollama embeddings."""

import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from semantic_cache import SemanticCache

BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "semantic_cache_results.json"

QA_PAIRS = [
    ("What is machine learning?", "Machine learning enables systems to learn patterns from data."),
    ("What is supervised learning?", "Supervised learning trains on labelled examples."),
    ("What is unsupervised learning?", "Unsupervised learning discovers patterns in unlabelled data."),
    ("What is deep learning?", "Deep learning uses multilayer neural networks."),
    ("What is a neural network?", "A neural network is a model made of interconnected computational units."),
    ("What is overfitting?", "Overfitting occurs when a model memorizes training data and generalizes poorly."),
    ("What is underfitting?", "Underfitting occurs when a model cannot capture important patterns."),
    ("What is cross-validation?", "Cross-validation estimates generalization by evaluating multiple data splits."),
    ("What is a training dataset?", "Training data is used to fit model parameters."),
    ("What is a test dataset?", "Test data evaluates a trained model on held-out examples."),
    ("What is precision?", "Precision is the fraction of predicted positives that are correct."),
    ("What is recall?", "Recall is the fraction of actual positives that are identified."),
    ("What is the F1 score?", "The F1 score is the harmonic mean of precision and recall."),
    ("What is data preprocessing?", "Data preprocessing cleans and transforms data before modelling."),
    ("What is feature engineering?", "Feature engineering creates or transforms input variables for a model."),
    ("What is a vector database?", "A vector database stores embeddings and supports similarity search."),
    ("What are embeddings?", "Embeddings represent text or other data as numeric vectors."),
    ("What is retrieval-augmented generation?", "RAG retrieves relevant source context to ground generated answers."),
    ("What is prompt caching?", "Prompt caching reuses eligible repeated prompt content to reduce cost and latency."),
    ("What is model quantization?", "Quantization reduces numeric precision to make model inference more efficient."),
]


def embed(text: str) -> list[float]:
    payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode("utf-8")
    request = Request(
        f"{BASE_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))
    embeddings = result.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"Ollama returned no embeddings for: {text[:60]}")
    return embeddings[0]


def main() -> None:
    started = time.perf_counter()
    cache = SemanticCache(threshold=0.92)

    for question, answer in QA_PAIRS:
        cache.add(question, answer, embed(question))

    matching_question = "What is machine learning?"
    unrelated_question = "How do I bake a chocolate cake?"

    match = cache.lookup(embed(matching_question))
    miss = cache.lookup(embed(unrelated_question))

    report = {
        "embedding_model": EMBED_MODEL,
        "cache_threshold": cache.threshold,
        "qa_pairs_loaded": len(cache),
        "matching_query": matching_question,
        "matching_query_result": match,
        "unrelated_query": unrelated_question,
        "unrelated_query_result": miss,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "note": (
            "Cache hit/miss outcomes depend on the actual embedding model. "
            "Similarity scores and cache behavior are measured, not assumed."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved results to: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
        raise SystemExit(
            f"Semantic cache experiment could not complete: {exc}\n"
            f"Check that Ollama is running and model '{EMBED_MODEL}' is available."
        ) from exc

