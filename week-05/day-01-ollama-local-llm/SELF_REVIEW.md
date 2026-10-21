# W5D1 Self-Review

- [x] Ollama installed and local API verified.
- [x] `llama3.2:3b` and `qwen2.5:3b` available locally.
- [x] Python Ollama client implemented.
- [x] Inference responses and latency saved.
- [x] Both models compared using the same three prompts and system prompt.
- [x] Four client unit tests passed in the recorded run.
- [x] Six model comparison responses validated.
- [x] Two CIA reviews completed and summarized in `docs/CIA_INTERACTIONS.md`.
- [ ] Final test run and evidence validation.
- [ ] Two descriptive commits created.
- [ ] Feature branch pushed.
- [ ] Pull request opened and checked.

## Measured comparison

| Model | Successful responses | Total latency |
|---|---:|---:|
| llama3.2:3b | 3 | 108.24 seconds |
| qwen2.5:3b | 3 | 158.71 seconds |

Latency depends on hardware, model loading, and warm-up; these measurements describe this run and are not universal benchmarks.

## Review note

CIA feedback was based on code excerpts. Optional suggestions were assessed against the existing implementation rather than applied without verification.
