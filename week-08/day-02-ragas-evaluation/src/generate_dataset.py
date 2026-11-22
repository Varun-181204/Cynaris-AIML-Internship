import json
from pathlib import Path

from rag_pipeline import answer_question, build_vector_store

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "evaluation_dataset.json"

QUESTIONS = [
    ("What is Retrieval-Augmented Generation?", "rag_fundamentals.txt"),
    ("How does RAG combine retrieval and generation?", "rag_fundamentals.txt"),
    ("What is a vector database?", "chromadb.txt"),
    ("How does ChromaDB store embeddings?", "chromadb.txt"),
    ("What are text embeddings?", "embeddings.txt"),
    ("Why are embeddings useful in semantic search?", "embeddings.txt"),
    ("What is the purpose of RAG evaluation?", "evaluation.txt"),
    ("What does faithfulness measure?", "evaluation.txt"),
    ("What is LlamaIndex used for?", "llamaindex.txt"),
    ("How does LlamaIndex support RAG applications?", "llamaindex.txt"),
]


def main():
    store, chunk_count = build_vector_store()
    samples = []

    for index, (question, expected_source) in enumerate(QUESTIONS, 1):
        result = answer_question(store, question)

        samples.append(
            {
                "id": index,
                "user_input": question,
                "response": result["answer"],
                "retrieved_contexts": result["contexts"],
                "reference": (
                    "Answer the question using the supplied source document. "
                    f"The expected supporting document is {expected_source}."
                ),
                "reference_source": expected_source,
                "actual_sources": result["sources"],
            }
        )

        print(f"[{index}/10] Generated: {question}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDataset saved: {OUTPUT}")
    print(f"Total samples: {len(samples)}")
    print(f"Chunks indexed: {chunk_count}")


if __name__ == "__main__":
    main()
