"""W6D1 Practical Task 3: LangChain agent with calculator and web-search stub."""

import ast
import json
import logging
import operator
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "llama3.2:3b"
BASE_URL = "http://localhost:11434"

ROOT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT_DIR / "week-06" / "day-01-langchain-fundamentals" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "agent_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Tool 1: Safe calculator
# ---------------------------------------------------------

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


def evaluate_expression(expression: str):
    """Evaluate basic arithmetic without using eval()."""

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ValueError("Only numeric values are allowed.")
            return node.value

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)

            if operator_type not in OPERATORS:
                raise ValueError("Unsupported arithmetic operator.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if operator_type is ast.Pow and abs(right) > 10:
                raise ValueError("Exponent is too large.")

            return OPERATORS[operator_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)

            if operator_type not in UNARY_OPERATORS:
                raise ValueError("Unsupported unary operator.")

            return UNARY_OPERATORS[operator_type](evaluate(node.operand))

        raise ValueError("Invalid arithmetic expression.")

    if not expression or len(expression) > 100:
        raise ValueError("Expression must contain 1–100 characters.")

    try:
        syntax_tree = ast.parse(expression, mode="eval")
        return evaluate(syntax_tree.body)
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.") from None


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression using basic arithmetic.

    Use this tool for addition, subtraction, multiplication, division,
    percentages, and expressions with parentheses.
    """
    try:
        result = evaluate_expression(expression)
        return f"Calculation: {expression} = {result}"
    except (ValueError, SyntaxError, TypeError, OverflowError) as error:
        return f"Calculation error: {error}"


# ---------------------------------------------------------
# Tool 2: Local web-search stub
# ---------------------------------------------------------

SEARCH_DATABASE = {
    "langchain": (
        "LangChain is a framework for developing applications powered "
        "by large language models. It provides components for prompts, "
        "models, tools, agents, and retrieval workflows."
    ),
    "ollama": (
        "Ollama is a tool for running large language models locally. "
        "It provides a local API for interacting with supported models."
    ),
    "python": (
        "Python is a general-purpose programming language widely used "
        "in automation, web development, data science, and AI/ML."
    ),
}


@tool
def web_search_stub(query: str) -> str:
    """Search a small local knowledge base for information about a topic.

    This is a demonstration search tool, not a live internet search.
    Use it when the user asks for information about a topic.
    """
    query_lower = query.lower()

    matches = [
        description
        for keyword, description in SEARCH_DATABASE.items()
        if keyword in query_lower
    ]

    if matches:
        return (
            "Local search results (stub, not live internet): "
            + " ".join(matches)
        )

    return (
        "No matching entry was found in the local search database. "
        "This demonstration tool does not access the live internet."
    )


# ---------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a helpful AI assistant with access to two tools:
1. calculator: for mathematical calculations.
2. web_search_stub: for searching the local demonstration knowledge base.

IMPORTANT RULES:
- You MUST use the calculator tool for mathematical calculations.
- You MUST use web_search_stub when asked to search for information.
- For tasks requiring both calculations and information, use both tools.
- Do not invent calculation results.
- Clearly explain that search results come from a local stub, not live internet.
- Give concise, beginner-friendly answers.
"""


TEST_TASKS = [
    {
        "id": 1,
        "task": "Use the calculator tool to compute (125 * 8) + 40. Show the result.",
        "expected_tools": ["calculator"],
    },
    {
        "id": 2,
        "task": (
            "Use the web search tool to find a short explanation of "
            "LangChain from its local knowledge base."
        ),
        "expected_tools": ["web_search_stub"],
    },
    {
        "id": 3,
        "task": (
            "Use the calculator to calculate 15% of 240. "
            "Then use the web search tool to find a short definition of Ollama."
        ),
        "expected_tools": ["calculator", "web_search_stub"],
    },
]


# ---------------------------------------------------------
# Agent execution
# ---------------------------------------------------------

def message_to_text(content):
    """Convert message content into a readable string."""
    if isinstance(content, str):
        return content

    return json.dumps(content, ensure_ascii=False, default=str)


def run_agent_task(agent, task_details):
    """Run one task and capture its response and tool usage."""

    task_id = task_details["id"]
    task = task_details["task"]

    logger.info("Running agent task %s/3", task_id)
    start_time = time.perf_counter()

    try:
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": task}
                ]
            },
            config={"recursion_limit": 12},
        )

        elapsed = round(time.perf_counter() - start_time, 3)
        messages = result.get("messages", [])

        tool_messages = [
            message
            for message in messages
            if isinstance(message, ToolMessage)
        ]

        used_tools = list(
            dict.fromkeys(
                message.name
                for message in tool_messages
                if message.name
            )
        )

        final_response = ""
        for message in reversed(messages):
            if getattr(message, "type", "") == "ai" and message.content:
                final_response = message_to_text(message.content)
                break

        expected_tools = task_details["expected_tools"]
        tools_verified = all(
            name in used_tools for name in expected_tools
        )

        logger.info(
            "Task %s completed in %.2f seconds. Tools used: %s",
            task_id,
            elapsed,
            used_tools,
        )

        return {
            "task_id": task_id,
            "task": task,
            "response": final_response,
            "expected_tools": expected_tools,
            "tools_used": used_tools,
            "tool_calls_count": len(tool_messages),
            "tools_verified": tools_verified,
            "latency_seconds": elapsed,
            "success": bool(final_response),
        }

    except Exception as error:
        elapsed = round(time.perf_counter() - start_time, 3)
        logger.exception("Agent task %s failed", task_id)

        return {
            "task_id": task_id,
            "task": task,
            "response": "",
            "expected_tools": task_details["expected_tools"],
            "tools_used": [],
            "tool_calls_count": 0,
            "tools_verified": False,
            "latency_seconds": elapsed,
            "success": False,
            "error": str(error),
        }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    logger.info("Starting W6D1 LangChain agent experiment")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=BASE_URL,
        temperature=0,
        num_predict=220,
    )

    tools = [calculator, web_search_stub]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    results = [
        run_agent_task(agent, task)
        for task in TEST_TASKS
    ]

    successful_tasks = sum(
        result["success"] for result in results
    )

    verified_tasks = sum(
        result["tools_verified"] for result in results
    )

    all_expected_tools_used = all(
        result["tools_verified"] for result in results
    )

    output_data = {
        "experiment": "W6D1 LangChain Agent with Two Tools",
        "model": MODEL_NAME,
        "framework": "LangChain create_agent",
        "tools": [
            "calculator",
            "web_search_stub",
        ],
        "search_tool_limitation": (
            "The web search tool uses a local static knowledge base. "
            "It does not access the live internet."
        ),
        "total_tasks": len(TEST_TASKS),
        "successful_tasks": successful_tasks,
        "verified_tasks": verified_tasks,
        "all_expected_tools_used": all_expected_tools_used,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("W6D1 LANGCHAIN AGENT RESULTS")
    print("=" * 60)

    for result in results:
        print(f"\nTask {result['task_id']}: {result['task']}")
        print(f"Response: {result['response']}")
        print(f"Expected tools: {result['expected_tools']}")
        print(f"Tools used: {result['tools_used']}")
        print(f"Tool calls: {result['tool_calls_count']}")
        print(f"Tools verified: {result['tools_verified']}")
        print(f"Latency: {result['latency_seconds']} seconds")

    print("\n" + "=" * 60)
    print(f"Successful tasks: {successful_tasks}/3")
    print(f"Tasks with expected tools verified: {verified_tasks}/3")
    print(f"All expected tools used: {all_expected_tools_used}")
    print(f"Evidence saved: {OUTPUT_FILE}")
    print("=" * 60)

    if successful_tasks != 3 or not all_expected_tools_used:
        raise SystemExit(
            "Agent execution completed, but one or more tasks "
            "did not meet the expected tool-usage checks."
        )


if __name__ == "__main__":
    main()