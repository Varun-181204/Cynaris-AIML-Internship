"""Tests for the semantic cache."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "semantic_cache.py"
)
SPEC = importlib.util.spec_from_file_location("semantic_cache", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SemanticCache = MODULE.SemanticCache


def test_add_and_retrieve_matching_question():
    cache = SemanticCache(threshold=0.92)
    cache.add("What is Python?", "A programming language.", [1.0, 0.0])
    result = cache.lookup([1.0, 0.0])
    assert result is not None
    assert result["answer"] == "A programming language."
    assert result["similarity"] == pytest.approx(1.0)


def test_similar_embedding_meets_threshold():
    cache = SemanticCache(threshold=0.92)
    cache.add("Question A", "Answer A", [1.0, 0.0])
    result = cache.lookup([0.995, 0.1])
    assert result is not None
    assert result["answer"] == "Answer A"


def test_dissimilar_embedding_is_cache_miss():
    cache = SemanticCache(threshold=0.92)
    cache.add("Question A", "Answer A", [1.0, 0.0])
    assert cache.lookup([0.0, 1.0]) is None


def test_empty_cache_is_cache_miss():
    assert SemanticCache().lookup([1.0, 0.0]) is None


def test_add_twenty_question_answer_pairs():
    cache = SemanticCache()
    for i in range(20):
        cache.add(f"Question {i}", f"Answer {i}", [1.0, float(i + 1)])
    assert len(cache) == 20


def test_exact_question_replaces_existing_entry():
    cache = SemanticCache()
    cache.add("What is Python?", "Old answer", [1.0, 0.0])
    cache.add("WHAT IS PYTHON?", "Updated answer", [1.0, 0.0])
    assert len(cache) == 1
    assert cache.lookup([1.0, 0.0])["answer"] == "Updated answer"


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        SemanticCache(threshold=threshold)


@pytest.mark.parametrize("embedding", [[], [0.0, 0.0], [float("nan"), 1.0]])
def test_invalid_embedding_is_rejected(embedding):
    cache = SemanticCache()
    with pytest.raises(ValueError):
        cache.add("Question", "Answer", embedding)


def test_invalid_question_is_rejected():
    with pytest.raises(ValueError):
        SemanticCache().add(" ", "Answer", [1.0, 0.0])


def test_mismatched_embedding_dimensions_are_rejected():
    cache = SemanticCache()
    cache.add("Question A", "Answer A", [1.0, 0.0])
    with pytest.raises(ValueError):
        cache.lookup([1.0, 0.0, 0.0])
