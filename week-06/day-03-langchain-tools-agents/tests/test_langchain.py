import json
import sys
from pathlib import Path

# Add the W6D3 project directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_tools import calculator, web_search_stub
from src.basic_chain import QUESTIONS, MODEL_NAME
from src.conversation_memory import CONVERSATION

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


def test_chain_has_five_inputs():
    assert len(QUESTIONS) == 5


def test_chain_model_configuration():
    assert MODEL_NAME == "llama3.2:3b"


def test_chain_evidence():
    evidence = json.loads(
        (OUTPUT_DIR / "chain_results.json").read_text(encoding="utf-8")
    )
    assert evidence["successful_inputs"] == 5
    assert evidence["failed_inputs"] == 0


def test_memory_has_five_turns():
    assert len(CONVERSATION) == 5


def test_memory_evidence():
    evidence = json.loads(
        (OUTPUT_DIR / "memory_results.json").read_text(encoding="utf-8")
    )
    assert evidence["successful_turns"] == 5
    assert evidence["retained_messages"] == 10
    assert evidence["all_user_inputs_retained"] is True


def test_calculator_addition():
    assert calculator.invoke({"expression": "10 + 5"}) == "15"


def test_calculator_multiplication():
    assert calculator.invoke({"expression": "12 * 4"}) == "48"


def test_calculator_order_of_operations():
    assert calculator.invoke({"expression": "425 * 3 + 175"}) == "1450"


def test_calculator_division():
    assert calculator.invoke({"expression": "20 / 4"}) == "5.0"


def test_calculator_power():
    assert calculator.invoke({"expression": "2 ** 5"}) == "32"


def test_calculator_blocks_unsafe_expression():
    result = calculator.invoke({
        "expression": "__import__('os').system('echo unsafe')"
    })
    assert "error" in result.lower()


def test_calculator_blocks_unsupported_function():
    result = calculator.invoke({"expression": "open('test.txt')"})
    assert "error" in result.lower()


def test_calculator_blocks_excessive_exponent():
    result = calculator.invoke({"expression": "2 ** 100"})
    assert "error" in result.lower()


def test_web_search_langchain():
    result = web_search_stub.invoke({"query": "LangChain"})
    assert "large language models" in result.lower()


def test_web_search_rag():
    result = web_search_stub.invoke({"query": "RAG"})
    assert "retrieval-augmented generation" in result.lower()


def test_web_search_unknown_topic():
    result = web_search_stub.invoke({"query": "unknown subject"})
    assert "no matching topic found" in result.lower()
    assert "does not access the live internet" in result.lower()


def test_agent_evidence():
    evidence = json.loads(
        (OUTPUT_DIR / "agent_results.json").read_text(encoding="utf-8")
    )
    assert evidence["successful_tasks"] == 3
    assert evidence["all_tasks_passed"] is True
    assert evidence["all_expected_tools_verified"] is True

def test_calculator_blocks_import_execution():
    """Ensure calculator rejects attempts to execute imported system commands."""
    result = calculator.invoke({
        "expression": "__import__('os').system('echo unsafe')"
    })

    assert any(
        word in result.lower()
        for word in ("unsupported", "error", "rejected", "invalid")
    )