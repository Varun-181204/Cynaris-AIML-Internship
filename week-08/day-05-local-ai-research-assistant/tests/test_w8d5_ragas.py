from contextlib import nullcontext
from types import SimpleNamespace

from src import evaluation


def test_evaluate_research_saves_deterministic_evidence(tmp_path, monkeypatch):
    expected_answer = "RAG retrieves relevant documents."
    expected_contexts = ["RAG uses retrieval to ground generated answers."]

    class FakeWorkflow:
        def invoke(self, state):
            assert state["question"] == "What is RAG?"
            return {
                "answer": expected_answer,
                "contexts": expected_contexts,
                "sources": ["rag_fundamentals.txt"],
            }

    class FakeFaithfulness:
        def __init__(self, llm):
            self.llm = llm

        def score(self, **kwargs):
            assert kwargs["response"] == expected_answer
            assert kwargs["retrieved_contexts"] == expected_contexts
            return SimpleNamespace(value=1.0)

    monkeypatch.setattr(evaluation, "build_workflow", lambda: FakeWorkflow())
    monkeypatch.setattr(evaluation, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(evaluation, "llm_factory", lambda **kwargs: object())
    monkeypatch.setattr(evaluation, "Faithfulness", FakeFaithfulness)

    monkeypatch.setattr(evaluation, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(evaluation, "EVIDENCE_DIR", tmp_path / "evidence")

    monkeypatch.setattr(evaluation.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(evaluation.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(evaluation.mlflow, "start_run", lambda **kwargs: nullcontext())
    monkeypatch.setattr(evaluation.mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation.mlflow, "log_metric", lambda *args, **kwargs: None)

    result = evaluation.evaluate_research("What is RAG?")

    assert result["faithfulness_score"] == 1.0
    assert result["metric"] == "Ragas Faithfulness"
    assert result["answer"] == expected_answer

    evidence_file = tmp_path / "evidence" / "ragas_evaluation.json"
    assert evidence_file.exists()
