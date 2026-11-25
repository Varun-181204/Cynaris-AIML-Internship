"""Evaluate research answer faithfulness using Ragas and local Ollama."""

import json
from datetime import UTC, datetime

import instructor
import mlflow
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from src.config import (
    EVIDENCE_DIR,
    EXPERIMENT_NAME,
    OLLAMA_URL,
    PROJECT_DIR,
)
from src.workflow import build_workflow


def evaluate_research(question: str) -> dict:
    """Run the assistant and score its answer against retrieved contexts."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    result = build_workflow().invoke({"question": question.strip()})
    answer = result.get("answer", "")
    contexts = result.get("contexts", [])
    sources = result.get("sources", [])

    if not answer or not contexts:
        raise RuntimeError("Research workflow returned no answer or contexts.")

    client = OpenAI(
        api_key="ollama",
        base_url=f"{OLLAMA_URL}/v1",
        timeout=180.0,
    )
    evaluator = llm_factory(
        model="qwen2.5:3b",
        provider="openai",
        client=client,
        mode=instructor.Mode.JSON,
        temperature=0.0,
    )

    metric = Faithfulness(llm=evaluator)
    score = float(
        metric.score(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        ).value
    )

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    database = PROJECT_DIR / "outputs" / "mlflow.db"
    mlflow.set_tracking_uri("sqlite:///" + database.resolve().as_posix())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="ragas-faithfulness"):
        mlflow.log_param("evaluation_metric", "faithfulness")
        mlflow.log_param("evaluator_model", "qwen2.5:3b")
        mlflow.log_metric("ragas_faithfulness", score)

    evidence = {
        "timestamp": datetime.now(UTC).isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_contexts": contexts,
        "metric": "Ragas Faithfulness",
        "faithfulness_score": score,
        "evaluator_model": "qwen2.5:3b",
    }

    output = EVIDENCE_DIR / "ragas_evaluation.json"
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"Faithfulness: {score:.4f}")
    print(f"Evidence: {output}")
    return evidence
