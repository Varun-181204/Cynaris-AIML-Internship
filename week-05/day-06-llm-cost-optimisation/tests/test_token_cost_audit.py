
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "token_cost_audit.py"
)
SPEC = importlib.util.spec_from_file_location("token_cost_audit", MODULE_PATH)
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def test_token_count_is_positive():
    assert audit_module.count_tokens("What is RAG?") > 0


def test_token_count_rejects_non_string():
    with pytest.raises(TypeError):
        audit_module.count_tokens(123)


def test_gpt4o_cost_calculation():
    # 1,000 input tokens at $5/M + 1,000 output at $15/M = $0.02.
    cost = audit_module.estimate_api_cost(1000, 1000, "GPT-4o")
    assert cost == pytest.approx(0.02)


def test_negative_tokens_rejected():
    with pytest.raises(ValueError):
        audit_module.estimate_api_cost(-1, 100, "GPT-4o")


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        audit_module.estimate_api_cost(100, 100, "unknown-model")


def test_audit_has_five_prompt_sizes():
    audit = audit_module.build_audit()
    assert len(audit) == 5
    assert audit["prompt_size"].nunique() == 5


def test_optimized_input_is_lower_or_equal():
    audit = audit_module.build_audit()
    assert (
        audit["optimized_input_tokens_est"]
        <= audit["baseline_input_tokens"]
    ).all()


def test_daily_savings_are_non_negative():
    audit = audit_module.build_audit()
    assert (audit["daily_savings_inr_est"] >= 0).all()


def test_invalid_compression_ratio_rejected():
    with pytest.raises(ValueError):
        audit_module.build_audit(compression_ratio=0)

