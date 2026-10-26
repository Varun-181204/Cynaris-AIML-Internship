"""Tests for assignment-rate cost tracking."""

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "cost_tracker.py"
SPEC = importlib.util.spec_from_file_location("cost_tracker", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CostTracker = MODULE.CostTracker


def test_gpt4o_cost_uses_assignment_rates():
    tracker = CostTracker()
    record = tracker.log_request(
        "gpt-4o", 1_000_000, 1_000_000,
        datetime.fromisoformat("2026-10-26T10:00:00"),
    )
    assert record["cost_usd"] == pytest.approx(20.0)
    assert record["cost_inr"] == pytest.approx(1760.0)


def test_local_model_has_zero_hosted_token_cost():
    record = CostTracker().log_request("llama3.2:3b", 5000, 1000)
    assert record["cost_usd"] == 0
    assert record["cost_inr"] == 0


def test_daily_totals_group_requests():
    tracker = CostTracker()
    tracker.log_request(
        "gpt-4o-mini", 1_000_000, 0,
        datetime.fromisoformat("2026-10-26T10:00:00"),
    )
    tracker.log_request(
        "gpt-4o-mini", 1_000_000, 0,
        datetime.fromisoformat("2026-10-26T11:00:00"),
    )
    assert tracker.daily_totals_inr()["2026-10-26"] == pytest.approx(26.4)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        CostTracker().log_request("unknown-model", 1, 1)


@pytest.mark.parametrize("tokens", [-1, 1.5, True])
def test_invalid_token_counts_are_rejected(tokens):
    with pytest.raises(ValueError):
        CostTracker().log_request("gpt-4o", tokens, 0)


def test_csv_export(tmp_path):
    tracker = CostTracker()
    tracker.log_request("gemini-2.0-flash", 1000, 200)
    path = tracker.save_csv(tmp_path / "usage.csv")
    content = path.read_text(encoding="utf-8")
    assert "input_tokens" in content
    assert "gemini-2.0-flash" in content
