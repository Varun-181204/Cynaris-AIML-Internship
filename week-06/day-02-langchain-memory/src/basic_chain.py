
"""W6D2: LangChain prompt-to-output chain using Qwen 2.5."""

from pathlib import Path
from datetime import datetime, timezone
import json
import logging
import time

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = ROOT / "week-06/day-02-langchain-memory/outputs/chain_results.json"

MODEL = "qwen2.5:3b"

QUESTIONS = [
    "Explain the difference between supervised and unsupervised learning.",
    "What is overfitting in machine learning?",
    "Why do we split data into training and testing sets?",
    "Explain the purpose of a vector database in AI applications.",
    "What is retrieval augmented generation (RAG)?",
]


def build_chain():
    """Create a reusable PromptTemplate → Ollama → OutputParser chain."""
    prompt = PromptTemplate.from_template(
        """You are a beginner-friendly AI/ML tutor.

Answer the following question clearly:
{question}

Requirements:
- Use simple English.
- Keep the answer between 3 and 5 sentences.
- Include one practical example when appropriate.
"""
    )

    llm = ChatOllama(
        model=MODEL,
        base_url="http://localhost:11434",
        temperature=0,
        num_predict=180,
    )

    parser = StrOutputParser()
    return prompt | llm | parser


def main():
    chain = build_chain()
    results = []

    for index, question in enumerate(QUESTIONS, start=1):
        start = time.perf_counter()

        try:
            response = chain.invoke({"question": question})
            latency = round(time.perf_counter() - start, 3)

            results.append({
                "test_number": index,
                "question": question,
                "response": response,
                "latency_seconds": latency,
                "status": "success",
            })

            logger.info("Test %d/%d passed (%.2fs)", index, len(QUESTIONS), latency)

        except Exception as exc:
            results.append({
                "test_number": index,
                "question": question,
                "error": str(exc),
                "status": "failed",
            })
            logger.exception("Test %d failed", index)

    successful = sum(item["status"] == "success" for item in results)
    latencies = [
        item["latency_seconds"]
        for item in results
        if item["status"] == "success"
    ]

    evidence = {
        "task": "W6D2 LangChain Fundamentals",
        "model": MODEL,
        "architecture": "PromptTemplate -> ChatOllama -> StrOutputParser",
        "total_tests": len(QUESTIONS),
        "successful_tests": successful,
        "failed_tests": len(QUESTIONS) - successful,
        "average_latency_seconds": (
            round(sum(latencies) / len(latencies), 3) if latencies else None
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(evidence, indent=4), encoding="utf-8")

    logger.info("Completed: %d/%d tests passed", successful, len(QUESTIONS))
    logger.info("Evidence saved to %s", OUTPUT_FILE)

    if successful != len(QUESTIONS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
