"""Compare two local models using the same three prompts and system prompt."""

import json
from pathlib import Path

from ollama_api import chat

BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / "outputs"
MODELS = ["llama3.2:3b", "qwen2.5:3b"]
SYSTEM = "You are a clear, accurate AI assistant. Answer the question directly."
PROMPTS = [
    "Explain machine learning to a beginner in exactly three sentences.",
    "What is the difference between classification and regression? Give one example of each.",
    "A model correctly predicts 90 out of 120 test examples. Calculate its accuracy as a percentage.",
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []

    for prompt_index, prompt in enumerate(PROMPTS, start=1):
        for model in MODELS:
            print(f"Question {prompt_index}/3 | {model}")
            try:
                result = chat(model, SYSTEM, prompt)
                result.update({
                    "question_number": prompt_index,
                    "status": "success",
                })
                print(f"Latency: {result['latency_seconds']}s")
                print(result["response"])
            except Exception as exc:
                result = {
                    "question_number": prompt_index,
                    "model": model,
                    "system_prompt": SYSTEM,
                    "prompt": prompt,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(result["error"])
            results.append(result)

    json_path = OUTPUT / "model_comparison_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Local LLM Comparison",
        "",
        "Both models received the same system prompt and user questions. "
        "Latency includes local inference time and may vary with model loading.",
        "",
    ]
    for index, prompt in enumerate(PROMPTS, start=1):
        lines.extend([f"## Question {index}", "", prompt, ""])
        for model in MODELS:
            item = next(
                r for r in results
                if r["question_number"] == index and r["model"] == model
            )
            lines.extend([
                f"### {model}",
                "",
                f"Status: {item['status']}",
                "",
                item.get("response", item.get("error", "")),
                "",
                f"Latency (seconds): {item.get('latency_seconds', 'N/A')}",
                "",
            ])

    (OUTPUT / "model_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\nSaved: {json_path}")
    print(f"Saved: {OUTPUT / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
