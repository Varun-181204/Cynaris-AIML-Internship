"""Command-line entry point for the local research assistant."""

import argparse

from src.evaluation import evaluate_research
from src.ingestion import ingest_documents
from src.research_assistant import research


def main():
    parser = argparse.ArgumentParser(description="Local AI Research Assistant")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--question", type=str)
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.ingest:
        print(f"Indexed {ingest_documents()} document chunks.")
    elif args.evaluate:
        question = args.question or "What is retrieval-augmented generation?"
        result = evaluate_research(question)
        print(f"Ragas Faithfulness: {result['faithfulness_score']:.4f}")
    elif args.question:
        result = research(args.question, enable_review=not args.no_review)
        print("\nANSWER\n", result["answer"])
        print("\nSOURCES\n", ", ".join(result["sources"]))
        if result["review"]:
            print("\nCREWAI REVIEW\n", result["review"])
    else:
        parser.error("Provide --ingest, --question, or --evaluate.")


if __name__ == "__main__":
    main()
