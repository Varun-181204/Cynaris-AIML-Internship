"""Application service combining LangGraph, CrewAI, and MLflow."""

from src.agents import review_answer
from src.tracking import track_research
from src.workflow import build_workflow


def research(question: str, enable_review: bool = True) -> dict:
    """Answer a research question, optionally review it, and track the run."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    result = build_workflow().invoke({"question": question.strip()})
    answer = result.get("answer", "")
    contexts = result.get("contexts", [])
    review = None

    if enable_review:
        review = review_answer(question, answer, contexts)

    track_research(
        question=question,
        answer=answer,
        source_count=len(result.get("sources", [])),
        review_enabled=enable_review,
    )

    return {
        "question": question.strip(),
        "answer": answer,
        "sources": result.get("sources", []),
        "review": review,
    }
