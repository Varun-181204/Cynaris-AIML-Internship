"""W5D6: Measured LLMLingua-2 prompt compression experiment.

The input is exactly 2,000 GPT-4o-compatible tiktoken tokens.
Each chunk is kept below the LLMLingua-2 model's 512-token limit.
Quality is assessed with a transparent keyword-retention proxy, not an
LLM answer-quality benchmark.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
from llmlingua import PromptCompressor

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
RESULT_JSON = OUTPUT_DIR / "llmlingua_compression_result.json"
RESULT_TXT = OUTPUT_DIR / "llmlingua_compression_report.txt"

TARGET_TOKENS = 2000
COMPRESSION_RATE = 0.4
CHUNK_TOKENS = 350
MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

FACTS = {
    "embeddings represent text numerically": ["embedding", "vector"],
    "vector databases support similarity search": ["vector database", "similarity search"],
    "RAG retrieves relevant chunks before generation": ["rag", "retriev"],
    "answers should be grounded in source documents": ["source document", "context"],
    "the assistant should admit uncertainty": ["uncertainty"],
    "retrieved passages need relevance checks": ["relevance"],
}

SEED_TEXT = """
RAG, or retrieval-augmented generation, answers questions using evidence retrieved
from external source documents. An embedding model converts text into numerical
vectors; a vector database stores these vectors and supports similarity search.
The retrieval stage finds relevant document chunks and passes them to the language
model before generation. Chunking divides long documents into manageable passages,
and overlap can preserve context across chunk boundaries. The assistant must answer
from the supplied context, cite source material when available, check retrieved
passages for relevance, and clearly admit uncertainty when the evidence is missing.
A retrieval pipeline should not invent facts or treat irrelevant passages as proof.
For reliable document question answering, evaluate both factual correctness and
whether the answer is supported by the retrieved source documents.
""".strip()


def make_exact_token_prompt(encoder, target: int) -> str:
    """Build a deterministic prompt with exactly target tokenizer tokens."""
    repeated = (SEED_TEXT + "\n\n") * (target + 10)
    token_ids = encoder.encode(repeated, disallowed_special=())
    prompt = encoder.decode(token_ids[:target])
    actual = len(encoder.encode(prompt, disallowed_special=()))
    if actual != target:
        token_ids = encoder.encode(prompt, disallowed_special=())
        prompt = encoder.decode(token_ids[:target])
        actual = len(encoder.encode(prompt, disallowed_special=()))
    if actual != target:
        raise RuntimeError(
            f"Could not construct exactly {target} tokens; got {actual}"
        )
    return prompt


def split_into_chunks(text: str, encoder, chunk_tokens: int) -> list[str]:
    """Split text by the measurement tokenizer into bounded chunks."""
    token_ids = encoder.encode(text, disallowed_special=())
    chunks = []
    for start in range(0, len(token_ids), chunk_tokens):
        chunk = encoder.decode(token_ids[start:start + chunk_tokens])
        if chunk.strip():
            chunks.append(chunk)
    if not chunks:
        raise ValueError("Prompt produced no non-empty chunks.")
    return chunks


def assess_fact_retention(text: str) -> dict:
    normalized = " ".join(text.lower().split())
    results = {
        fact: all(marker in normalized for marker in markers)
        for fact, markers in FACTS.items()
    }
    retained = sum(results.values())
    return {
        "facts_checked": len(results),
        "facts_retained": retained,
        "retention_pct": round(100 * retained / len(results), 2),
        "fact_checks": results,
        "method_note": (
            "Heuristic keyword proxy only; it does not prove semantic equivalence "
            "or answer quality. Manually review the compressed prompt."
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    encoder = tiktoken.encoding_for_model("gpt-4o")
    prompt = make_exact_token_prompt(encoder, TARGET_TOKENS)
    input_tokens = len(encoder.encode(prompt, disallowed_special=()))
    chunks = split_into_chunks(prompt, encoder, CHUNK_TOKENS)

    chunk_sizes = [
        len(encoder.encode(chunk, disallowed_special=()))
        for chunk in chunks
    ]
    if max(chunk_sizes) > CHUNK_TOKENS:
        raise RuntimeError(f"Chunk exceeds configured limit: {max(chunk_sizes)}")

    print(f"Compression model: {MODEL_NAME}")
    print(f"Input tokens: {input_tokens}")
    print(f"Chunk count: {len(chunks)}")
    print(f"Maximum chunk size: {max(chunk_sizes)} GPT-4o tokens")
    print("Loading compression model; first run may download model files.")

    start = time.perf_counter()
    compressor = PromptCompressor(
        model_name=MODEL_NAME,
        use_llmlingua2=True,
        device_map="cpu",
    )

    compressed_chunks = []
    print(f"Compressing chunks individually at rate={COMPRESSION_RATE} ...")
    for index, chunk in enumerate(chunks, start=1):
        print(f"  Chunk {index}/{len(chunks)} ({chunk_sizes[index - 1]} tokens)")
        result = compressor.compress_prompt(
            [chunk],
            rate=COMPRESSION_RATE,
            force_tokens=["\n", "?", "."],
        )
        compressed_chunk = result.get("compressed_prompt", "")
        if not compressed_chunk:
            raise RuntimeError(
                f"No compressed_prompt returned for chunk {index}; "
                f"result keys: {list(result.keys())}"
            )
        compressed_chunks.append(compressed_chunk)

    elapsed_seconds = round(time.perf_counter() - start, 2)
    compressed_prompt = "\n".join(compressed_chunks)
    compressed_tokens = len(
        encoder.encode(compressed_prompt, disallowed_special=())
    )
    reduction_pct = round(
        (1 - compressed_tokens / input_tokens) * 100, 2
    )

    report = {
        "experiment": "W5D6 LLMLingua-2 chunked prompt compression",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "compression_model": MODEL_NAME,
        "rate_requested_per_chunk": COMPRESSION_RATE,
        "chunking": {
            "measurement_tokenizer": "tiktoken encoding_for_model('gpt-4o')",
            "chunk_limit_tokens": CHUNK_TOKENS,
            "chunk_count": len(chunks),
            "chunk_input_token_counts": chunk_sizes,
            "largest_chunk_tokens": max(chunk_sizes),
            "note": (
                "The 350-token limit uses GPT-4o tokenization. This is a conservative "
                "approximation, not a guarantee of 350 tokens under the model's "
                "own tokenizer."
            ),
        },
        "tokenizer_for_measurement": "tiktoken encoding_for_model('gpt-4o')",
        "input_tokens": input_tokens,
        "compressed_tokens": compressed_tokens,
        "measured_token_reduction_pct": reduction_pct,
        "measured_token_retention_pct": round(
            100 * compressed_tokens / input_tokens, 2
        ),
        "elapsed_seconds_including_model_load": elapsed_seconds,
        "quality_proxy_before": assess_fact_retention(prompt),
        "quality_proxy_after": assess_fact_retention(compressed_prompt),
        "quality_limitations": (
            "Keyword retention is a lightweight proxy, not a semantic or task-level "
            "quality evaluation. Chunk-by-chunk compression can also lose context "
            "across chunk boundaries. Compare answers using the same downstream "
            "model before making claims about answer quality."
        ),
        "compressed_prompt": compressed_prompt,
    }

    RESULT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "W5D6 LLMLingua-2 Chunked Compression Experiment",
        "=" * 48,
        f"Timestamp (UTC): {report['timestamp_utc']}",
        f"Compression model: {MODEL_NAME}",
        f"Requested compression rate per chunk: {COMPRESSION_RATE}",
        f"Input tokens: {input_tokens}",
        f"Chunk count: {len(chunks)}",
        f"Chunk limit (GPT-4o tokens): {CHUNK_TOKENS}",
        f"Largest chunk: {max(chunk_sizes)} tokens",
        f"Compressed tokens: {compressed_tokens}",
        f"Measured token reduction: {reduction_pct}%",
        f"Measured token retention: {report['measured_token_retention_pct']}%",
        f"Elapsed seconds including model load: {elapsed_seconds}",
        "",
        "Quality proxy before:",
        json.dumps(report["quality_proxy_before"], indent=2),
        "",
        "Quality proxy after:",
        json.dumps(report["quality_proxy_after"], indent=2),
        "",
        "Compressed prompt:",
        compressed_prompt,
        "",
        "Limitations: keyword checks do not establish semantic equivalence or answer quality.",
    ]
    RESULT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\nEXPERIMENT RESULTS")
    print(f"Input tokens: {input_tokens}")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Compressed tokens: {compressed_tokens}")
    print(f"Measured reduction: {reduction_pct}%")
    print(f"Elapsed seconds: {elapsed_seconds}")
    print(f"JSON results: {RESULT_JSON}")
    print(f"Text report: {RESULT_TXT}")


if __name__ == "__main__":
    main()
