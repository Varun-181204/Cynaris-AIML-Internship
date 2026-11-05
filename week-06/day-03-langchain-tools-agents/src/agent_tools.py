
import ast
import json
import operator
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.messages import ToolMessage


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "llama3.2:3b"

KNOWLEDGE_BASE = {
    "langchain": (
        "LangChain is a framework for building applications powered by "
        "large language models. It provides chains, prompts, tools, "
        "agents, and memory integrations."
    ),
    "ollama": (
        "Ollama allows users to run large language models locally. "
        "It supports models such as Llama 3.2 and Qwen 2.5."
    ),
    "rag": (
        "Retrieval-Augmented Generation (RAG) retrieves relevant external "
        "information and provides it to an LLM as context before generation."
    ),
    "python": (
        "Python is a high-level programming language commonly used in "
        "automation, data science, artificial intelligence, and web development."
    ),
}


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_evaluate(node):
    if isinstance(node, ast.Expression):
        return safe_evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float)
        ):
            raise ValueError("Only numeric values are allowed.")
        return node.value

    if isinstance(node, ast.BinOp):
        operation = OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported operator.")

        left = safe_evaluate(node.left)
        right = safe_evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent limit exceeded.")

        result = operation(left, right)

        if isinstance(result, (int, float)) and abs(result) > 1e100:
            raise ValueError("Result exceeds the allowed limit.")

        return result

    if isinstance(node, ast.UnaryOp):
        operation = UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported unary operator.")
        return operation(safe_evaluate(node.operand))

    raise ValueError("Unsupported expression.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely.

    Supports addition, subtraction, multiplication, division,
    floor division, modulo, exponentiation, and unary signs.
    """
    if not isinstance(expression, str) or len(expression) > 100:
        return "Calculation error: invalid or overly long expression."

    try:
        tree = ast.parse(expression, mode="eval")
        result = safe_evaluate(tree)
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def web_search_stub(query: str) -> str:
    """Search a small local knowledge base. This is not live internet search."""
    normalized = re.sub(r"[^a-z0-9 ]", " ", query.lower())

    matches = [
        description
        for topic, description in KNOWLEDGE_BASE.items()
        if topic in normalized
    ]

    if not matches:
        return (
            "No matching topic found in the local knowledge base. "
            "This tool does not access the live internet."
        )

    return "\n".join(matches)


def extract_tool_calls(messages):
    return [
        message.name
        for message in messages
        if isinstance(message, ToolMessage)
    ]


def run_agent():
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    agent = create_agent(
        model=llm,
        tools=[calculator, web_search_stub],
        system_prompt=(
            "You are a helpful assistant. Use the calculator for arithmetic "
            "and web_search_stub for information searches. "
            "The search tool uses a local knowledge base, not the live internet. "
            "Explain the results clearly and never invent tool results."
        ),
    )

    tasks = [
        {
            "task": "Calculate 425 * 3 + 175.",
            "expected_tools": ["calculator"],
        },
        {
            "task": "Search the local knowledge base for LangChain.",
            "expected_tools": ["web_search_stub"],
        },
        {
            "task": (
                "Calculate 20 percent of 450 and search the local "
                "knowledge base for RAG."
            ),
            "expected_tools": ["calculator", "web_search_stub"],
        },
    ]

    results = []

    for index, item in enumerate(tasks, start=1):
        print(f"\nTask {index}: {item['task']}")

        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": item["task"]}]},
                config={"recursion_limit": 12},
            )

            messages = response["messages"]
            tool_calls = extract_tool_calls(messages)
            final_answer = str(messages[-1].content)

            expected = set(item["expected_tools"])
            actual = set(tool_calls)

            tools_verified = expected.issubset(actual)

            result = {
                "task_id": index,
                "task": item["task"],
                "expected_tools": item["expected_tools"],
                "tools_used": tool_calls,
                "tools_verified": tools_verified,
                "success": tools_verified and bool(final_answer.strip()),
                "final_answer": final_answer,
            }

            print(f"Tools used: {tool_calls}")
            print(f"Tools verified: {tools_verified}")
            print(f"Answer: {final_answer}")

        except Exception as exc:
            result = {
                "task_id": index,
                "task": item["task"],
                "expected_tools": item["expected_tools"],
                "tools_used": [],
                "tools_verified": False,
                "success": False,
                "error": str(exc),
            }
            print(f"Task failed: {exc}")

        results.append(result)

    successful = sum(item["success"] for item in results)

    evidence = {
        "task": "W6D3 LangChain Tools and Agents",
        "model": MODEL_NAME,
        "agent_api": "langchain.agents.create_agent",
        "available_tools": ["calculator", "web_search_stub"],
        "total_tasks": len(tasks),
        "successful_tasks": successful,
        "failed_tasks": len(tasks) - successful,
        "all_tasks_passed": successful == len(tasks),
        "all_expected_tools_verified": all(
            item["tools_verified"] for item in results
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    output_file = OUTPUT_DIR / "agent_results.json"
    output_file.write_text(
        json.dumps(evidence, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nAgent result: {successful}/{len(tasks)}")
    print(f"All expected tools verified: {evidence['all_expected_tools_verified']}")
    print(f"Evidence saved: {output_file}")

    return evidence


if __name__ == "__main__":
    run_agent()
