import pytest

from src.document_chatbot import (
    QUESTIONS,
    CONVERSATION,
    safe_evaluate,
    calculator,
    web_search_stub,
)


def test_five_document_questions_defined():
    assert len(QUESTIONS) == 5


def test_five_memory_turns_defined():
    assert len(CONVERSATION) == 5


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("425 * 3 + 175", 1450),
        ("450 * 0.20", 90),
        ("20 + 5 * 2", 30),
        ("10 // 3", 3),
    ],
)
def test_safe_calculator(expression, expected):
    assert safe_evaluate(__import__("ast").parse(expression, mode="eval")) == expected


def test_calculator_rejects_unsafe_expression():
    result = calculator.invoke({"expression": "__import__('os').system('whoami')"})
    assert "error" in result.lower()


def test_web_search_stub_returns_known_topic():
    result = web_search_stub.invoke({"query": "Explain LangChain"})
    assert "framework" in result.lower()


def test_web_search_stub_discloses_unknown_topic():
    result = web_search_stub.invoke({"query": "unknown subject"})
    assert "no matching topic" in result.lower()


def test_document_pdf_exists():
    from src.document_chatbot import PDF_PATH
    assert PDF_PATH.is_file()
