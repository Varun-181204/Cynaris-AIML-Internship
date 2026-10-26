"""Track per-request LLM token costs and daily INR totals."""

import csv
from datetime import datetime
from pathlib import Path

USD_TO_INR = 88.0  # Assignment assumption, not a live exchange rate.

# Illustrative assignment rates, USD per million tokens.
PRICING = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "llama3.2:3b": {"input": 0.0, "output": 0.0},
    "qwen2.5:3b": {"input": 0.0, "output": 0.0},
}


class CostTracker:
    """Record usage, calculate estimated cost, and summarize by day."""

    def __init__(self):
        self.records = []

    def log_request(
        self, model, input_tokens, output_tokens, timestamp=None
    ):
        if model not in PRICING:
            raise ValueError(f"Unknown model: {model}")
        if (
            isinstance(input_tokens, bool)
            or isinstance(output_tokens, bool)
            or not isinstance(input_tokens, int)
            or not isinstance(output_tokens, int)
            or input_tokens < 0
            or output_tokens < 0
        ):
            raise ValueError("Token counts must be non-negative integers")

        moment = timestamp or datetime.now().astimezone()
        if isinstance(moment, str):
            moment = datetime.fromisoformat(moment)
        rates = PRICING[model]
        input_cost_usd = input_tokens * rates["input"] / 1_000_000
        output_cost_usd = output_tokens * rates["output"] / 1_000_000
        total_usd = input_cost_usd + output_cost_usd

        record = {
            "timestamp": moment.isoformat(),
            "date": moment.date().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(total_usd, 10),
            "cost_inr": round(total_usd * USD_TO_INR, 8),
        }
        self.records.append(record)
        return record

    def daily_totals_inr(self):
        totals = {}
        for record in self.records:
            date = record["date"]
            totals[date] = totals.get(date, 0.0) + record["cost_inr"]
        return {date: round(total, 8) for date, total in sorted(totals.items())}

    def save_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "timestamp", "date", "model", "input_tokens",
            "output_tokens", "cost_usd", "cost_inr",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.records)
        return path
