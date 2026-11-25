import asyncio
import json
from pathlib import Path
from statistics import mean

import instructor
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "evaluation_dataset.json"
OUTPUT = ROOT / "outputs" / "optimized_results.json"

client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    timeout=240.0,
)

llm = llm_factory(
    model="qwen2.5:3b",
    provider="openai",
    client=client,
    mode=instructor.Mode.MD_JSON,
    temperature=0.0,
)

metric = Faithfulness(llm=llm)


async def main():
    samples = json.loads(DATASET.read_text(encoding="utf-8"))
    scores = []

    for i, sample in enumerate(samples, 1):
        print(f"Evaluating faithfulness {i}/{len(samples)}", flush=True)

        result = await metric.ascore(
            user_input=sample["user_input"],
            response=sample["response"],
            retrieved_contexts=sample["retrieved_contexts"],
        )

        score = float(result.value)
        scores.append(score)
        print(f"  Score: {score:.4f}", flush=True)

    average = mean(scores)

    evidence = {
        "evaluation": "optimized",
        "model": "qwen2.5:3b",
        "top_k": 2,
        "samples": len(scores),
        "faithfulness": average,
        "individual_scores": scores,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print(f"\nOptimized faithfulness: {average:.4f}")
    print("Baseline faithfulness: 0.5258")
    print(f"Evidence saved: {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
