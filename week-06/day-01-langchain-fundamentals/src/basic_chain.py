
"""W6D1: LangChain PromptTemplate -> Ollama -> OutputParser."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "week-06" / "day-01-langchain-fundamentals" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "chain_results.json"

MODEL_NAME = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a concise and accurate AI/ML learning assistant.

Explain the following topic to a beginner:
Topic: {topic}

Instructions:
- Use simple English.
- Give a clear definition.
- Include one practical example.
- Keep the answer under 100 words.
- Do not invent technical facts.
"""

TEST_TOPICS = [
    "What is machine learning?",
    "What is LangChain?",
    "What is a prompt template?",
    "What is an LLM?",
    "What is an output parser?",
]


def build_chain() -> Any:
    """Build and return the LangChain processing pipeline."""
    prompt = PromptTemplate(
        input_variables=["topic"],
        template=PROMPT_TEMPLATE,
    )

    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_predict=180,
    )

    parser = StrOutputParser()

    return prompt | llm | parser


def run_chain(topic: str, chain: Any) -> dict:
    """Execute the chain and capture its output and latency."""
    if not topic.strip():
        raise ValueError("Topic cannot be empty.")

    start_time = time.perf_counter()

    response = chain.invoke({"topic": topic})

    latency = time.perf_counter() - start_time

    return {
        "topic": topic,
        "response": response.strip(),
        "latency_seconds": round(latency, 3),
        "success": bool(response.strip()),
    }


def run_experiments() -> list[dict]:
    """Run the chain against all five test topics."""
    chain = build_chain()
    results = []

    for index, topic in enumerate(TEST_TOPICS, start=1):
        logger.info("Running test %d/%d: %s", index, len(TEST_TOPICS), topic)

        try:
            result = run_chain(topic, chain)
            results.append(result)
            logger.info("Completed in %.3f seconds", result["latency_seconds"])

        except Exception as exc:
            logger.exception("Chain failed for topic: %s", topic)
            results.append({
                "topic": topic,
                "response": "",
                "latency_seconds": None,
                "success": False,
                "error": str(exc),
            })

    return results


def save_results(results: list[dict]) -> None:
    """Save experiment evidence as JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    successful = sum(result["success"] for result in results)

    evidence = {
        "experiment": "W6D1 LangChain Basic Chain",
        "model": MODEL_NAME,
        "architecture": [
            "PromptTemplate",
            "ChatOllama",
            "StrOutputParser",
        ],
        "total_tests": len(results),
        "successful_tests": successful,
        "failed_tests": len(results) - successful,
        "average_latency_seconds": (
            round(
                sum(
                    r["latency_seconds"]
                    for r in results
                    if r["success"]
                ) / successful,
                3,
            )
            if successful else None
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    OUTPUT_FILE.write_text(
        json.dumps(evidence, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Evidence saved to %s", OUTPUT_FILE)


def main() -> None:
    """Run the W6D1 chain experiment."""
    logger.info("Starting W6D1 LangChain experiment")

    results = run_experiments()
    save_results(results)

    successful = sum(result["success"] for result in results)

    print("\n" + "=" * 60)
    print("W6D1 LANGCHAIN BASIC CHAIN RESULTS")
    print("=" * 60)

    for index, result in enumerate(results, start=1):
        print(f"\nTest {index}: {result['topic']}")
        print(f"Success: {result['success']}")
        print(f"Latency: {result['latency_seconds']} seconds")
        print(f"Response: {result['response'][:300]}")

    print("\n" + "=" * 60)
    print(f"Successful tests: {successful}/{len(results)}")
    print(f"Evidence: {OUTPUT_FILE}")
    print("=" * 60)

    if successful != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
