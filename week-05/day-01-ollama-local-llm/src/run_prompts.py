import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ollama_client import OllamaClient

prompts = [
    "Explain machine learning to a beginner in three simple sentences.",
    "Explain the difference between supervised and unsupervised learning, with one example of each.",
    "What is overfitting in machine learning? Give two ways to reduce it.",
    "Write a Python function that checks whether an integer is prime. Include a short explanation.",
    "Explain quantization in large language models and mention one benefit and one trade-off."
]

client = OllamaClient(
    model="llama3.2:3b",
    system_prompt=(
        "You are a helpful AI/ML learning assistant. "
        "Answer accurately in simple language. Follow requested formats "
        "and keep explanations concise."
    ),
    timeout=180,
)

results = {
    "model": client.model,
    "system_prompt": client.system_prompt,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "results": [],
}

for index, prompt in enumerate(prompts, start=1):
    print(f"\n[{index}/{len(prompts)}] {prompt}")
    start = time.perf_counter()
    try:
        answer = client.chat(prompt)
        elapsed = round(time.perf_counter() - start, 2)
        item = {
            "prompt_number": index,
            "prompt": prompt,
            "response": answer,
            "latency_seconds": elapsed,
            "status": "success",
        }
        print(f"\nResponse ({elapsed}s):\n{answer}")
    except Exception as exc:
        item = {
            "prompt_number": index,
            "prompt": prompt,
            "error": str(exc),
            "status": "error",
        }
        print(f"\nERROR: {exc}")
    results["results"].append(item)

output = Path(__file__).resolve().parents[1] / "outputs" / "inference_results.json"
output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
successful = sum(item["status"] == "success" for item in results["results"])
print(f"\nSaved: {output}")
print(f"Successful prompts: {successful}/{len(prompts)}")
if successful != len(prompts):
    sys.exit(1)
