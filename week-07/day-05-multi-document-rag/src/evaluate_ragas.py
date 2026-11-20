import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
import instructor
import mlflow
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rag_system import build_workflow


def evaluate_rag(question: str) -> dict:
    # Run the existing LangGraph retrieval and generation workflow.
    workflow = build_workflow()
    result = workflow.invoke({"question": question})

    answer = result.get("answer", "")
    contexts = result.get("contexts", [])
    sources = result.get("sources", [])

    if not answer or not contexts:
        raise RuntimeError("RAG workflow returned an empty answer or no contexts.")

    # Connect Ragas to the local Ollama OpenAI-compatible endpoint.
    client = OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        timeout=180.0,
    )
    evaluator_llm = llm_factory(
        model="llama3.2:3b",
        provider="openai",
        client=client,
        mode=instructor.Mode.MD_JSON,
        temperature=0.0,
    )

    metric = Faithfulness(llm=evaluator_llm)
    metric_result = metric.score(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    score = float(metric_result.value)

    # Record the evaluation in MLflow using the same SQLite backend.
    tracking_uri = "sqlite:///" + (
        REPO_ROOT / "week-07" / "day-05-multi-document-rag"
        / "outputs" / "mlflow.db"
    ).resolve().as_posix()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("W7D5-Multi-Document-RAG")

    with mlflow.start_run(run_name="ragas-faithfulness-evaluation"):
        mlflow.log_param("evaluation_metric", "faithfulness")
        mlflow.log_param("evaluator_model", "llama3.2:3b")
        mlflow.log_param("question", question)
        mlflow.log_metric("ragas_faithfulness", score)

    evidence = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_contexts": contexts,
        "metric": "Ragas Faithfulness",
        "faithfulness_score": score,
        "evaluator_model": "llama3.2:3b",
    }

    output_file = OUTPUT_DIR / "ragas_evaluation.json"
    output_file.write_text(
        json.dumps(evidence, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nRAGAS EVALUATION")
    print("=" * 45)
    print("Question:", question)
    print("Answer:", answer)
    print("Sources:", ", ".join(sources))
    print(f"Faithfulness Score: {score:.4f}")
    print("Evidence:", output_file)
    print("MLflow: Evaluation metric logged successfully.")

    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG with Ragas.")
    parser.add_argument(
        "--question",
        default="What is retrieval-augmented generation and how does it work?",
    )
    args = parser.parse_args()
    evaluate_rag(args.question)
