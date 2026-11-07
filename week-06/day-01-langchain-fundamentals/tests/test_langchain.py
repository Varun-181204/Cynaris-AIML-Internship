import sys
from pathlib import Path

DAY_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DAY_FOLDER))
import pytest

from src.agent_tools import evaluate_expression
from src.agent_tools import calculator, web_search_stub


# Calculator tests

@pytest.mark.parametrize(
    "expression, expected",
    [
        ("125 * 8 + 40", 1040),
        ("240 * 0.15", 36.0),
        ("(10 + 5) * 2", 30),
        ("100 / 4", 25),
        ("-10 + 20", 10),
    ],
)
def test_calculator_valid_expressions(expression, expected):
    assert evaluate_expression(expression) == expected


def test_calculator_division_by_zero():
    with pytest.raises(ValueError, match="Division by zero"):
        evaluate_expression("10 / 0")


def test_calculator_rejects_unsafe_expression():
    with pytest.raises(ValueError):
        evaluate_expression("__import__('os').system('dir')")


def test_calculator_tool():
    result = calculator.invoke({"expression": "20 * 5"})
    assert "100" in result


# Search stub tests

def test_search_langchain():
    result = web_search_stub.invoke({"query": "LangChain"})
    assert "framework" in result.lower()


def test_search_ollama():
    result = web_search_stub.invoke({"query": "Ollama"})
    assert "locally" in result.lower()


def test_search_unknown_topic():
    result = web_search_stub.invoke({"query": "unknown technology"})
    assert "no matching entry" in result.lower()


def test_search_discloses_stub_limitation():
    result = web_search_stub.invoke({"query": "Python"})
    assert "not live internet" in result.lower()
