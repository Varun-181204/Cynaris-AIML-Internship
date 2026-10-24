# W5D5 — Self-Review Checklist

## Implementation

- [x] Implemented a local document Q&A bot using Ollama and ChromaDB.
- [x] Extracted and chunked text from the sample PDF.
- [x] Generated embeddings using `nomic-embed-text`.
- [x] Retrieved relevant passages with source and page metadata.
- [x] Generated answers using `llama3.2:3b`.
- [x] Added input validation and error-handling tests.
- [x] Saved evidence from real local-model inference.
- [x] Preserved evidence from the initial single-question run.
- [x] Added a README with setup and usage instructions.
- [x] Verified `.gitignore` excludes the persistent ChromaDB database.

## Validation

- [x] Python syntax check passed after updating the demonstration.
- [x] W5D5 automated tests passed: 9 passed in the last recorded W5D5-specific run.
- [x] Full Week 5 test suite passed: 34 passed, 1 non-blocking warning.
- [x] Real Q&A run returned answers and source metadata for two questions.
- [x] Evidence JSON validation passed.

## Quality and limitations

- [x] Documented that prompt-based grounding does not guarantee factual correctness.
- [x] Documented that retrieval distance is not a confidence score.
- [ ] Test additional PDFs and questions beyond the supplied sample.
- [ ] Review the final source and documentation diff before committing.
- [ ] Complete two genuine CIA Full Stack Mentor Mode review interactions and record their actual feedback.

## Git and submission

- [ ] Stage and review the intended W5D5 files.
- [ ] Create the first descriptive W5D5 commit.
- [ ] Create the second descriptive W5D5 commit after final documentation and review.
- [ ] Push the commits to `feat/aiml-W5-Varun-K-S`.
- [ ] Verify the remote branch and update the existing Week 5 pull request as appropriate.

## Known warning

The full Week 5 test run produced one `DeprecationWarning` from ChromaDB telemetry concerning `asyncio.iscoroutinefunction`. All 34 tests passed; the warning did not fail the suite.

## Review status

Implementation and automated validation are complete for the current sample scenario. Additional document coverage, final Git review, and the two actual CIA mentor-review interactions remain pending.
