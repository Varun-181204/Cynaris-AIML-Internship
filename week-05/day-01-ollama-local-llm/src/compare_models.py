import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ollama_client import OllamaClient

questions = [
    "Explain machine learning to a beginner in exactly three sentences.",
    "Explain overfitting and give one effective prevention method.",
    "Write a short Python function that checks whether an integer is prime, and explain how it works."
]

system_prompt = (
    "You are a helpful AI/ML learning assistant. "
    "Be accurate, clear, concise, and follow the user's requested format."
)
models = ["llama3.2:3b", "qwen2.5:3b"]
results = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "system_prompt": system_prompt,
    "questions": questions,
    "models": {}
}

for model in models:
    print(f"\n===== {model} =====")
    client = OllamaClient(model=model, system_prompt=system_prompt, timeout=180)
    model_results = []

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/3] {question}")
        start = time.perf_counter()
        try:
            answer = client.chat(question)
            elapsed = round(time.perf_counter() - start, 2)
            model_results.append({
                "question": question,
                "response": answer,
                "latency_seconds": elapsed,
                "status": "success"
            })
            print(f"Latency: {elapsed}s\n{answer}")
        except Exception as exc:
            model_results.append({
                "question": question,
                "error": str(exc),
                "status": "error"
            })
            print(f"ERROR: {exc}")

    results["models"][model] = model_results

output_dir = Path(__file__).resolve().parents[1] / "outputs"
json_path = output_dir / "model_comparison_results.json"
md_path = output_dir / "model_comparison.md"
json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

lines = [
    "# Llama 3.2 vs Qwen 2.5 — Local Model Comparison",
    "",
    f"- Generated (UTC): {results['generated_at_utc']}",
    "- System prompt and questions were held constant for both models.",
    "- Latency includes the local API request and response generation.",
    "",
    "## Latency results",
    "",
    "| Question | Llama 3.2 (seconds) | Qwen 2.5 (seconds) |",
    "|---|---:|---:|"
]
for i, question in enumerate(questions):
    values = []
    for model in models:
        item = results["models"][model][i]
        values.append(str(item.get("latency_seconds", "ERROR")))
    lines.append(f"| Q{i+1} | {values[0]} | {values[1]} |")

lines += ["", "## Actual responses", ""]
for i, question in enumerate(questions):
    lines += [f"### Question {i+1}", "", question, ""]
    for model in models:
        item = results["models"][model][i]
        lines += [f"**{model}**", ""]
        if item["status"] == "success":
            lines += [f"*Latency: {item['latency_seconds']} seconds*", "", item["response"], ""]
        else:
            lines += [f"Error: {item['error']}", ""]

lines += [
    "## Observations",
    "",
    "Review the captured responses for correctness, clarity, completeness, and instruction-following.",
    "Do not infer that one model is universally better from only these three questions.",
    "Record your evidence-based conclusion here after reviewing the responses.",
    ""
]
md_path.write_text("\n".join(lines), encoding="utf-8")

successes = sum(
    item["status"] == "success"
    for model_results in results["models"].values()
    for item in model_results
)
print(f"\nSaved JSON: {json_path}")
print(f"Saved Markdown report: {md_path}")
print(f"Successful responses: {successes}/6")

if successes != 6:
    sys.exit(1)
