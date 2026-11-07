"""Automated tests for W6D2 LangChain tools."""

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.agent_tools import calculator, web_search_stub


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("275 * 4 + 125", 1225),
        ("350 * 0.18", 63),
        ("(10 + 5) * 2", 30),
        ("100 / 4", 25),
        ("-10 + 20", 10),
    ],
)
def test_calculator_valid_expressions(expression, expected):
    result = calculator.invoke({"expression": expression})
    assert float(result) == pytest.approx(expected)


def test_calculator_division_by_zero():
    result = calculator.invoke({"expression": "10 / 0"})
    assert result.startswith("Calculation error:")
    assert "division by zero" in result.lower()


def test_calculator_rejects_unsafe_expression():
    result = calculator.invoke(
        {"expression": "__import__('os').system('echo unsafe')"}
    )
    assert result.startswith("Calculation error:")
    assert "unsupported" in result.lower()


def test_calculator_rejects_unsupported_expression():
    result = calculator.invoke({"expression": "open('test.txt')"})
    assert result.startswith("Calculation error:")
    assert "unsupported" in result.lower()


def test_web_search_stub_rag():
    result = web_search_stub.invoke({"query": "RAG"})
    assert "retrieval-augmented generation" in result.lower()


def test_web_search_stub_ollama():
    result = web_search_stub.invoke({"query": "Ollama"})
    assert "ollama" in result.lower()


def test_web_search_stub_discloses_local_limitation():
    result = web_search_stub.invoke({"query": "unknown topic xyz"})
    assert "does not access the live internet" in result.lower()


def test_web_search_stub_unknown_topic():
    result = web_search_stub.invoke({"query": "unknown topic xyz"})
    assert "no matching topic found" in result.lower()