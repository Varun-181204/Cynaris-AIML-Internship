"""Run five prompt-engineering experiments against a local Ollama model."""

import json
from pathlib import Path

from ollama_api import chat

BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / "outputs"
MODEL = "llama3.2:3b"

EXPERIMENTS = [
    {
        "technique": "Role prompting",
        "system": "You are a patient machine-learning instructor for beginners.",
        "prompt": "Explain overfitting in machine learning.",
    },
    {
        "technique": "Output constraints",
        "system": "You are a concise technical writer. Follow all output constraints.",
        "prompt": "Explain supervised learning in exactly three bullet points. "
                  "Each bullet must contain at most 15 words.",
    },
    {
        "technique": "Few-shot prompting",
        "system": "Classify the sentiment of text as Positive, Negative, or Neutral. "
                  "Return only the label.",
        "prompt": 'Text: "I love this useful application."\nSentiment: Positive\n'
                  'Text: "The service was terrible."\nSentiment: Negative\n'
                  'Text: "The package arrived today."\nSentiment: Neutral\n'
                  'Text: "The support team solved my problem quickly."\nSentiment:',
    },
    {
        "technique": "Structured reasoning",
        "system": "You are a careful problem-solving tutor. Show a short, clear "
                  "calculation and then state the answer.",
        "prompt": "A dataset has 240 records. An 80/20 split is used. "
                  "How many records are in the training and test sets?",
    },
    {
        "technique": "Audience adaptation",
        "system": "Explain technical concepts using simple language for a first-year "
                  "engineering student. Use one practical analogy.",
        "prompt": "What is an API, and why do applications use APIs?",
    },
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        print(f"[{index}/{len(EXPERIMENTS)}] {experiment['technique']}")
        try:
            result = chat(
                MODEL, experiment["system"], experiment["prompt"]
            )
            result["technique"] = experiment["technique"]
            result["status"] = "success"
        except Exception as exc:
            result = {
                **experiment,
                "model": MODEL,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)
        if result["status"] == "success":
            print(f"Latency: {result['latency_seconds']}s")
            print(result["response"])
        else:
            print(result["error"])

    json_path = OUTPUT / "prompt_experiments.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Prompt Engineering Experiments",
        "",
        f"Model: `{MODEL}`",
        "",
        "Five techniques were tested. Review the actual responses before "
        "drawing conclusions about quality.",
        "",
    ]
    for item in results:
        lines.extend([
            f"## {item['technique']}",
            "",
            f"Status: {item['status']}",
            "",
            f"System prompt: {item['system_prompt']}",
            "",
            f"User prompt: {item['prompt']}",
            "",
            f"Response: {item.get('response', item.get('error', ''))}",
            "",
            f"Latency (seconds): {item.get('latency_seconds', 'N/A')}",
            "",
        ])
    (OUTPUT / "prompt_experiments.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\nSaved: {json_path}")
    print(f"Saved: {OUTPUT / 'prompt_experiments.md'}")


if __name__ == "__main__":
    main()

