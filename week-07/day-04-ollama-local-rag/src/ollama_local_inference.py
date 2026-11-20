import json
import time
import urllib.request
from pathlib import Path
from typing import Any


API_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """
You are a helpful AI and machine learning tutor.
Explain technical concepts accurately and clearly.
Use simple language, practical examples, and structured answers.
If you are uncertain, state that clearly instead of inventing facts.
"""

PROMPTS = [
    "What is Retrieval-Augmented Generation (RAG)?",
    "Explain vector databases and why they are useful in AI applications.",
    "What is the difference between embeddings and tokens?",
    "Explain how a local large language model generates a response.",
    "What are the advantages and limitations of using Ollama?"
]

COMPARISON_QUESTIONS = PROMPTS[:3]

MODELS = ["llama3.2:3b", "qwen2.5:3b"]

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "model_comparison.json"
)


def build_payload(
    model: str,
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT
) -> dict[str, Any]:
    """Build a request payload for Ollama's chat API."""

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }


def parse_response(response: dict[str, Any]) -> str:
    """Extract assistant content from an Ollama response."""

    content = response.get("message", {}).get("content")

    if not isinstance(content, str):
        raise ValueError("Ollama response does not contain message content.")

    return content.strip()


def generate_response(
    model: str,
    prompt: str,
    timeout: int = 300
) -> dict[str, Any]:
    """Send a prompt to the local Ollama API and measure latency."""

    payload = build_payload(model, prompt)

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "model": model,
        "prompt": prompt,
        "response": parse_response(result),
        "latency_ms": latency_ms
    }


def run_experiment() -> dict[str, Any]:
    """Run five prompts and compare both models on three questions."""

    print("\nW7D4: Ollama Local LLM Experiment")
    print("=" * 55)

    llama_results = []

    print("\n1. Testing llama3.2:3b with five prompts")

    for index, prompt in enumerate(PROMPTS, start=1):
        print(f"\nLlama prompt {index}/5: {prompt}")

        result = generate_response("llama3.2:3b", prompt)
        llama_results.append(result)

        print(f"Latency: {result['latency_ms']} ms")
        print(f"Response: {result['response'][:350]}...")

    print("\n2. Comparing both models on three identical questions")

    comparison_results = []

    for index, question in enumerate(COMPARISON_QUESTIONS):
        llama_result = llama_results[index]

        print(f"\nComparison question {index + 1}/3")

        qwen_result = generate_response("qwen2.5:3b", question)

        comparison_results.append({
            "question": question,
            "llama3.2:3b": {
                "response": llama_result["response"],
                "latency_ms": llama_result["latency_ms"]
            },
            "qwen2.5:3b": {
                "response": qwen_result["response"],
                "latency_ms": qwen_result["latency_ms"]
            },
            "quality_notes": ""
        })

        print(f"Llama latency: {llama_result['latency_ms']} ms")
        print(f"Qwen latency: {qwen_result['latency_ms']} ms")

    evidence = {
        "experiment": "W7D4 Ollama Local LLM Inference",
        "api": API_URL,
        "system_prompt": SYSTEM_PROMPT.strip(),
        "models": MODELS,
        "five_prompt_test": llama_results,
        "three_question_comparison": comparison_results,
        "comparison_method": (
            "Both models received identical questions and system prompts. "
            "Quality observations are to be completed through manual review."
        )
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(evidence, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nExperiment completed.")
    print(f"Evidence saved to: {OUTPUT_PATH}")

    return evidence


if __name__ == "__main__":
    run_experiment()
