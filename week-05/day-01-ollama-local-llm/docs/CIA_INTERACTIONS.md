# W5D1 CIA Interactions

## Interaction 1 — Implementation review

**Topic:** Running LLMs Locally with Ollama  
**Files reviewed:** `src/ollama_client.py`, `src/compare_models.py`, `tests/test_ollama_client.py`

**Summary of CIA feedback:**
- The client validates prompts, sends non-streaming chat requests, measures latency, validates response content, and wraps network errors.
- Suggested an explicit JSON Accept header, a `chat` docstring, clearer timeout handling, and optional retry support.
- Recommended more tests for malformed JSON and optional response-size handling.
- Recommended checking comparison error handling, result persistence, path resolution, and latency aggregation.

**Follow-up assessment:** The review was based on supplied code excerpts. The complete implementation had already produced six successful comparison responses and saved JSON and Markdown results. Persistence and latency aggregation were verified by the executed comparison workflow. Optional changes were not applied blindly; no source changes were justified solely by assumptions about omitted code.

## Interaction 2 — Prioritized correction plan

**Topic:** Installation checks, model availability, error handling, fair comparison, latency, tests, and evidence.

**Summary of CIA feedback:**
- Suggested pre-flight checks for the Ollama executable and API availability.
- Suggested checking installed models and giving a clear `ollama pull <model>` instruction when a model is missing.
- Recommended testing timeout, HTTP/network failures, malformed JSON, and missing response content.
- Recommended identical prompts and settings for both models.
- Recommended documenting hardware and warm-up effects on latency.
- Suggested mocked failure tests and validation of saved comparison evidence.

**Follow-up assessment:** The existing implementation uses Python's standard-library `urllib`; the proposed `requests` examples were not adopted because they would introduce an unnecessary dependency. The actual client and tests should be changed only if a concrete gap is found. The comparison used the same three prompts and system prompt for both models.

## Existing validation evidence

- Unit tests: 4 passed in the previously recorded run.
- Model comparison: 6 successful responses.
- Llama 3.2 total measured latency: 108.24 seconds.
- Qwen 2.5 total measured latency: 158.71 seconds.

**Remaining:** Re-run tests, inspect staged changes, commit, push the feature branch, and open the pull request.
