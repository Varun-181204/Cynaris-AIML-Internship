"""LangGraph workflow for grounded local research answers."""

from typing import TypedDict

import requests
from langgraph.graph import END, START, StateGraph

from src.config import CHAT_MODEL, OLLAMA_URL, TOP_K
from src.ingestion import embed_text, get_collection


class ResearchState(TypedDict, total=False):
    question: str
    contexts: list[str]
    sources: list[str]
    answer: str


def retrieve(state: ResearchState) -> dict:
    """Retrieve the most relevant document chunks."""
    question = state["question"].strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("No documents indexed. Run ingestion first.")

    result = collection.query(
        query_embeddings=[embed_text(question)],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas"],
    )

    contexts = result["documents"][0] or []
    metadatas = result["metadatas"][0] or []
    sources = sorted({item["source"] for item in metadatas})

    return {"contexts": contexts, "sources": sources}


def generate(state: ResearchState) -> dict:
    """Generate an answer using only retrieved research context."""
    context = "\n\n".join(state.get("contexts", []))
    if not context:
        return {"answer": "I could not find relevant information in the documents."}

    prompt = (
        "Answer the question using only the supplied research context. "
        "If the context does not contain the answer, say so. "
        "Do not invent facts.\n\n"
        f"Research context:\n{context}\n\n"
        f"Question: {state['question']}"
    )

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    answer = response.json().get("message", {}).get("content", "")
    return {"answer": answer.strip()}


def build_workflow():
    """Build and compile the retrieval-generation graph."""
    graph = StateGraph(ResearchState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def ask_question(question: str) -> dict:
    """Run the research assistant and return its answer and sources."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    result = build_workflow().invoke({"question": question.strip()})
    return {
        "question": question.strip(),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }
