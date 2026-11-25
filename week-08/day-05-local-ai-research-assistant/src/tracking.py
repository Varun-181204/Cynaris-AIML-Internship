"""MLflow experiment tracking using an isolated SQLite database."""

import time

import mlflow

from src.config import (
    CHAT_MODEL,
    EMBED_MODEL,
    EXPERIMENT_NAME,
    PROJECT_DIR,
    TOP_K,
)

MLFLOW_DB = PROJECT_DIR / "outputs" / "mlflow.db"


def track_research(
    question: str, answer: str, source_count: int, review_enabled: bool = False
) -> float:
    """Log research run metadata and return tracking duration."""
    MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)

    tracking_uri = "sqlite:///" + MLFLOW_DB.resolve().as_posix()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    start = time.perf_counter()
    with mlflow.start_run():
        mlflow.log_params(
            {
                "chat_model": CHAT_MODEL,
                "embedding_model": EMBED_MODEL,
                "top_k": TOP_K,
                "review_enabled": review_enabled,
            }
        )
        mlflow.log_metrics(
            {
                "source_count": source_count,
                "answer_length": len(answer),
                "question_length": len(question),
            }
        )
        elapsed = time.perf_counter() - start
        mlflow.log_metric("tracking_elapsed_seconds", elapsed)

    return elapsed
