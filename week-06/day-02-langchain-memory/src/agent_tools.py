
"""W6D2: Two-tool LangChain agent using Qwen 2.5."""

from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import operator
import logging

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = ROOT / "week-06/day-02-langchain-memory/outputs/agent_results.json"

MODEL = "qwen2.5:3b"

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

KNOWLEDGE_BASE = {
    "langchain": (
        "LangChain is a framework for building applications powered by "
        "large language models. It provides prompts, chains, tools, and agents."
    ),
    "ollama": (
        "Ollama allows users to run large language models locally. "
        "It supports models such as Llama and Qwen."
    ),
    "rag": (
        "Retrieval-Augmented Generation retrieves relevant external information "
        "and supplies it to an LLM to improve responses."
    ),
    "python": (
        "Python is a high-level programming language commonly used in "
        "automation, backend development, data science, and AI."
    ),
}


def evaluate_expression(expression: str):
    """Evaluate arithmetic expressions using a restricted AST."""
    if not isinstance(expression, str) or len(expression) > 100:
        raise ValueError("Expression must be a string of at most 100 characters.")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed.")
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent magnitude must not exceed 10.")

            return OPERATORS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](evaluate(node.operand))

        raise ValueError("Unsupported arithmetic expression.")

    return evaluate(tree)


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression. Input must contain only arithmetic."""
    try:
        return str(evaluate_expression(expression))
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculation error: {exc}"


@tool
def web_search_stub(query: str) -> str:
    """Search a small local knowledge base. This is not a live internet search."""
    query_lower = query.lower()

    for topic, information in KNOWLEDGE_BASE.items():
        if topic in query_lower:
            return f"Local knowledge result for '{topic}': {information}"

    return (
        "No matching topic found in the local knowledge base. "
        "This tool is a stub and does not access the live internet."
    )


def build_agent():
    llm = ChatOllama(
        model=MODEL,
        base_url="http://localhost:11434",
        temperature=0,
        num_predict=220,
    )

    return create_agent(
        model=llm,
        tools=[calculator, web_search_stub],
        system_prompt=(
            "You are a helpful AI assistant. "
            "Use calculator for arithmetic instead of calculating mentally. "
            "Use web_search_stub when asked to search the knowledge base. "
            "Clearly disclose that web_search_stub is not live internet search. "
            "Explain tool results in simple English."
        ),
    )


TASKS = [
    {
        "task": "Calculate 275 multiplied by 4, then add 125.",
        "expected_tool": "calculator",
    },
    {
        "task": "Search the local knowledge base for RAG and explain it.",
        "expected_tool": "web_search_stub",
    },
    {
        "task": "Calculate 18 percent of 350 and search for Ollama.",
        "expected_tools": {"calculator", "web_search_stub"},
    },
]


def main():
    agent = build_agent()
    results = []

    for index, item in enumerate(TASKS, start=1):
        logger.info("Running agent task %d/%d", index, len(TASKS))

        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": item["task"]}]},
                config={"recursion_limit": 12},
            )

            messages = response.get("messages", [])
            tool_names = sorted({
                message.name
                for message in messages
                if isinstance(message, ToolMessage)
            })

            expected = item.get("expected_tools", {item.get("expected_tool")})
            if None in expected:
                expected.remove(None)

            tools_verified = expected.issubset(set(tool_names))

            final_answer = next(
                (
                    message.content
                    for message in reversed(messages)
                    if getattr(message, "type", "") == "ai"
                    and message.content
                ),
                "",
            )

            success = bool(final_answer) and tools_verified

            results.append({
                "task_number": index,
                "task": item["task"],
                "expected_tools": sorted(expected),
                "tools_used": tool_names,
                "tools_verified": tools_verified,
                "response": final_answer,
                "status": "success" if success else "failed",
            })

            logger.info(
                "Task %d: %s | Tools: %s",
                index,
                "passed" if success else "failed",
                tool_names,
            )

        except Exception as exc:
            logger.exception("Agent task %d failed", index)
            results.append({
                "task_number": index,
                "task": item["task"],
                "error": str(exc),
                "status": "failed",
            })

    successful = sum(item["status"] == "success" for item in results)

    evidence = {
        "task": "W6D2 Two-Tool Agent",
        "model": MODEL,
        "agent_api": "langchain.agents.create_agent",
        "available_tools": ["calculator", "web_search_stub"],
        "total_tasks": len(TASKS),
        "successful_tasks": successful,
        "all_tasks_passed": successful == len(TASKS),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(evidence, indent=4), encoding="utf-8")

    logger.info("Agent completed: %d/%d tasks passed", successful, len(TASKS))
    logger.info("Evidence saved to %s", OUTPUT_FILE)

    if successful != len(TASKS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
