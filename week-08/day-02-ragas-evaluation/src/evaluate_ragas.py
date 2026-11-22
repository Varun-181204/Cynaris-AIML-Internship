import asyncio
import json
from pathlib import Path
from statistics import mean

import instructor
import mlflow
from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "evaluation_dataset.json"
OUTPUT_DIR = ROOT / "outputs"
EMBED_MODEL = "nomic-embed-text:latest"

client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    timeout=240.0,
)

evaluator_llm = llm_factory(
    model="qwen2.5:3b",
    provider="openai",
    client=client,
    mode=instructor.Mode.MD_JSON,
    temperature=0.0,
)

evaluator_embeddings = embedding_factory(
    "openai",
    model=EMBED_MODEL,
    client=client,
)

faithfulness = Faithfulness(llm=evaluator_llm)
answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
context_precision = ContextPrecision(llm=evaluator_llm)
context_recall = ContextRecall(llm=evaluator_llm)


async def main():
    samples = json.loads(DATASET.read_text(encoding="utf-8"))
    results = []

    for index, sample in enumerate(samples, 1):
        print(f"\nEvaluating {index}/{len(samples)}: {sample['user_input']}")

        user_input = sample["user_input"]
        response = sample["response"]
        contexts = sample["retrieved_contexts"]
        reference = sample["reference"]

        scores = {
            "faithfulness": await faithfulness.ascore(
                user_input=user_input,
                response=response,
                retrieved_contexts=contexts,
            ),
            "answer_relevancy": await answer_relevancy.ascore(
                user_input=user_input,
                response=response,
            ),
            "context_precision": await context_precision.ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=contexts,
            ),
            "context_recall": await context_recall.ascore(
                user_input=user_input,
                retrieved_contexts=contexts,
                reference=reference,
            ),
        }

        row = {"id": sample["id"], "question": user_input}
        for name, result in scores.items():
            row[name] = float(result.value)
            print(f"  {name}: {row[name]:.4f}")

        results.append(row)

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    averages = {name: mean(row[name] for row in results) for name in metric_names}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence = {
        "evaluation": "baseline",
        "samples": len(results),
        "scores": averages,
        "individual_results": results,
    }

    output_file = OUTPUT_DIR / "baseline_results.json"
    output_file.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(
        "sqlite:///" + (OUTPUT_DIR / "mlflow.db").resolve().as_posix()
    )
    mlflow.set_experiment("W8D2-Ragas-Evaluation")

    with mlflow.start_run(run_name="baseline-evaluation"):
        mlflow.log_param("chat_model", "llama3.2:3b")
        mlflow.log_param("embedding_model", EMBED_MODEL)
        mlflow.log_param("samples", len(results))
        for name, score in averages.items():
            mlflow.log_metric(name, score)

    print("\nBASELINE AVERAGES")
    for name, score in averages.items():
        print(f"{name}: {score:.4f}")
    print(f"\nEvidence saved: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
