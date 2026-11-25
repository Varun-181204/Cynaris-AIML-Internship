"""CrewAI reviewer for checking whether an answer is grounded."""

from crewai import LLM, Agent, Crew, Process, Task

from src.config import CHAT_MODEL, OLLAMA_URL


def review_answer(question: str, answer: str, contexts: list[str]) -> str:
    """Ask a local CrewAI agent to review answer grounding."""
    llm = LLM(
        model=f"ollama/{CHAT_MODEL}",
        base_url=OLLAMA_URL,
    )

    reviewer = Agent(
        role="Research Quality Reviewer",
        goal="Identify unsupported claims and assess answer grounding.",
        backstory="You carefully compare research answers against source evidence.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=(
            f"Question: {question}\n\nAnswer: {answer}\n\n"
            f"Available evidence: {' '.join(contexts)}\n\n"
            "Identify unsupported claims. Give a concise verdict and explain "
            "whether the answer is supported by the evidence."
        ),
        expected_output="A concise grounding review with a verdict.",
        agent=reviewer,
    )

    crew = Crew(
        agents=[reviewer],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff())
