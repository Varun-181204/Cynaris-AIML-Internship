
"""W5D6: Token economics, API cost estimates, and Excel audit."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
EXCEL_PATH = OUTPUT_DIR / "token_cost_audit.xlsx"
CSV_PATH = OUTPUT_DIR / "token_cost_audit.csv"

# Assignment-provided illustrative prices, in USD per million tokens.
# Verify provider pricing before using these values for real budgets.
MODEL_PRICING: dict[str, dict[str, Any]] = {
    "GPT-4o": {
        "input_per_million": 5.00,
        "output_per_million": 15.00,
        "context_window": 128_000,
    },
    "Claude Sonnet 4": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "context_window": 200_000,
    },
    "Gemini 2.0 Flash": {
        "input_per_million": 0.10,
        "output_per_million": 0.40,
        "context_window": 1_000_000,
    },
    "GPT-4o mini": {
        "input_per_million": 0.15,
        "output_per_million": 0.60,
        "context_window": 128_000,
    },
}

# Illustrative conversion rate. Replace with a dated exchange rate for
# financial reporting. This is not a live FX quote.
USD_TO_INR = 88.0

# The assignment asks us to measure five prompt sizes.
PROMPT_SAMPLES = [
    ("small", "What is RAG?"),
    (
        "short",
        "Explain retrieval augmented generation and how a language model "
        "uses retrieved documents to answer a user's question.",
    ),
    (
        "medium",
        "You are a careful document assistant. Use only the supplied context. "
        "Explain vector databases, embeddings, semantic retrieval, chunking, "
        "and how retrieval augmented generation grounds an answer in source "
        "documents. Mention uncertainty and do not invent facts. "
        "Question: How does a RAG pipeline work?",
    ),
    (
        "large",
        (
            "Document context: " + (
                "Vector databases store embeddings and support similarity "
                "search. RAG retrieves relevant chunks before generation. "
            ) * 35
            + "\nQuestion: Explain the role of retrieval in a RAG system."
        ),
    ),
    (
        "rag_2000_approx",
        (
            "You are a careful document assistant. Answer only from context. "
            "Cite source pages and admit uncertainty. "
            + (
                "Vector databases store numerical representations of text. "
                "An embedding model maps text to a vector. Semantic search "
                "retrieves chunks based on similarity. RAG combines retrieved "
                "context with a user question so the language model can answer "
                "with evidence. Chunk overlap may preserve context across "
                "boundaries. Retrieved passages should be checked for relevance. "
            ) * 120
            + "\nQuestion: How does a document Q&A system use RAG?"
        ),
    ),
]


def get_encoder():
    """Return the GPT-4o-compatible tokenizer."""
    return tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    """Count text tokens with the GPT-4o-compatible tokenizer."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return len(get_encoder().encode(text))


def estimate_api_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """Estimate one request's cost in USD using assignment prices."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative")
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model}")

    pricing = MODEL_PRICING[model]
    input_cost = input_tokens * pricing["input_per_million"] / 1_000_000
    output_cost = output_tokens * pricing["output_per_million"] / 1_000_000
    return input_cost + output_cost


def build_audit(
    *,
    output_tokens_per_request: int = 300,
    requests_per_day: int = 1000,
    compression_ratio: float = 0.40,
    selected_model: str = "GPT-4o",
) -> pd.DataFrame:
    """Build baseline and optimized estimates for five prompt sizes.

    compression_ratio=0.40 means the optimized prompt retains approximately
    40% of baseline tokens, i.e. a 60% input-token reduction.
    """
    if output_tokens_per_request < 0 or requests_per_day < 0:
        raise ValueError("Output tokens and request volume must be non-negative")
    if not 0 < compression_ratio <= 1:
        raise ValueError("compression_ratio must be greater than 0 and <= 1")
    if selected_model not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {selected_model}")

    rows = []

    for sample_name, prompt in PROMPT_SAMPLES:
        baseline_input = count_tokens(prompt)
        optimized_input = round(baseline_input * compression_ratio)

        baseline_request_cost = estimate_api_cost(
            baseline_input, output_tokens_per_request, selected_model
        )
        optimized_request_cost = estimate_api_cost(
            optimized_input, output_tokens_per_request, selected_model
        )

        baseline_daily = baseline_request_cost * requests_per_day
        optimized_daily = optimized_request_cost * requests_per_day

        rows.append({
            "audit_date": date.today().isoformat(),
            "prompt_size": sample_name,
            "model": selected_model,
            "baseline_input_tokens": baseline_input,
            "optimized_input_tokens_est": optimized_input,
            "output_tokens_per_request": output_tokens_per_request,
            "requests_per_day": requests_per_day,
            "input_reduction_pct_est": round(
                (1 - optimized_input / baseline_input) * 100, 2
            ) if baseline_input else 0,
            "baseline_cost_per_request_usd": baseline_request_cost,
            "optimized_cost_per_request_usd_est": optimized_request_cost,
            "baseline_daily_cost_usd": baseline_daily,
            "optimized_daily_cost_usd_est": optimized_daily,
            "baseline_daily_cost_inr_est": baseline_daily * USD_TO_INR,
            "optimized_daily_cost_inr_est": optimized_daily * USD_TO_INR,
            "daily_savings_inr_est": (
                baseline_daily - optimized_daily
            ) * USD_TO_INR,
            "pricing_note": "Assignment-provided illustrative prices",
            "fx_usd_inr": USD_TO_INR,
            "fx_note": "Illustrative rate; not a live exchange rate",
        })

    return pd.DataFrame(rows)


def export_audit(audit: pd.DataFrame) -> tuple[Path, Path]:
    """Export the audit to Excel and CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audit.to_csv(CSV_PATH, index=False)
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        audit.to_excel(writer, sheet_name="Token Cost Audit", index=False)

        pricing = pd.DataFrame([
            {"model": model, **details}
            for model, details in MODEL_PRICING.items()
        ])
        pricing.to_excel(writer, sheet_name="Assignment Pricing", index=False)

        notes = pd.DataFrame([
            {"item": "Purpose", "value": "Estimated baseline vs optimized API spend"},
            {"item": "Tokenizer", "value": "tiktoken encoding for GPT-4o"},
            {"item": "Compression assumption", "value": "Retains 40% of input tokens"},
            {"item": "Output tokens", "value": "Fixed assumption per request"},
            {"item": "Daily request volume", "value": "Illustrative scenario"},
            {"item": "INR conversion", "value": f"Illustrative USD/INR {USD_TO_INR}"},
            {"item": "Local Ollama", "value": "No hosted API token fee; hardware/electricity excluded"},
            {"item": "Caveat", "value": "Estimated token compression is not verified LLMLingua output"},
        ])
        notes.to_excel(writer, sheet_name="Assumptions", index=False)

    return EXCEL_PATH, CSV_PATH


def main() -> None:
    """Generate and print the audit."""
    audit = build_audit()
    excel_path, csv_path = export_audit(audit)

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(audit.to_string(index=False))

    print(f"\nExcel audit saved to: {excel_path}")
    print(f"CSV audit saved to:   {csv_path}")


if __name__ == "__main__":
    main()
