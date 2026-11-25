import mlflow
from src import tracking


def test_track_research_logs_parameters_and_metrics(tmp_path, monkeypatch):
    database = tmp_path / "test_mlflow.db"
    monkeypatch.setattr(tracking, "MLFLOW_DB", database)

    elapsed = tracking.track_research(
        question="What is RAG?",
        answer="RAG retrieves relevant information.",
        source_count=2,
        review_enabled=False,
    )

    assert elapsed >= 0

    uri = "sqlite:///" + database.resolve().as_posix()
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)

    experiment = client.get_experiment_by_name(tracking.EXPERIMENT_NAME)
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1

    run = runs[0]
    assert run.data.params["chat_model"] == tracking.CHAT_MODEL
    assert run.data.params["embedding_model"] == tracking.EMBED_MODEL
    assert run.data.params["review_enabled"] == "False"
    assert run.data.metrics["source_count"] == 2
    assert run.data.metrics["answer_length"] == len(
        "RAG retrieves relevant information."
    )
