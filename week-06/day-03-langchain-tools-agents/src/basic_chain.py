
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "llama3.2:3b"

QUESTIONS = [
    "Explain the difference between supervised and unsupervised learning.",
    "What is overfitting in machine learning?",
    "Why do we split a dataset into training and testing sets?",
    "What is the purpose of a vector database?",
    "Explain Retrieval-Augmented Generation (RAG).",
]


def build_chain():
    prompt = PromptTemplate.from_template(
        "You are a concise AI and machine learning tutor. "
        "Answer the following question clearly in 3 to 5 sentences.\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
    )

    return prompt | llm | StrOutputParser()


def run_chain():
    chain = build_chain()
    results = []

    for index, question in enumerate(QUESTIONS, start=1):
        start_time = time.perf_counter()

        try:
            answer = chain.invoke({"question": question})
            latency = round(time.perf_counter() - start_time, 3)

            results.append({
                "test_id": index,
                "question": question,
                "answer": answer,
                "success": bool(answer.strip()),
                "latency_seconds": latency,
            })

            print(f"\nTest {index}: {question}")
            print(f"Answer: {answer}")
            print(f"Latency: {latency}s")

        except Exception as exc:
            results.append({
                "test_id": index,
                "question": question,
                "answer": "",
                "success": False,
                "error": str(exc),
            })

            print(f"\nTest {index} failed: {exc}")

    successful = sum(item["success"] for item in results)
    latencies = [
        item["latency_seconds"]
        for item in results
        if item["success"]
    ]

    evidence = {
        "task": "W6D3 LangChain Basic Chain",
        "model": MODEL_NAME,
        "chain": "PromptTemplate -> ChatOllama -> StrOutputParser",
        "total_inputs": len(QUESTIONS),
        "successful_inputs": successful,
        "failed_inputs": len(QUESTIONS) - successful,
        "average_latency_seconds": (
            round(sum(latencies) / len(latencies), 3)
            if latencies else None
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    output_file = OUTPUT_DIR / "chain_results.json"
    output_file.write_text(
        json.dumps(evidence, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nChain result: {successful}/{len(QUESTIONS)}")
    print(f"Evidence saved: {output_file}")

    return evidence


if __name__ == "__main__":
    run_chain()
