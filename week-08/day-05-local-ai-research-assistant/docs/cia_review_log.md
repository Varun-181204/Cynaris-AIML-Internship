# W8D5 CIA Mentor Review Log

## Review 1 – Architecture and Code Quality

**Focus:** Architecture, separation of responsibilities, code quality, retrieval-generation workflow, CrewAI, MLflow, Ragas, and production readiness.

### Review outcome

The mentor gave positive assessments of the overall architecture and CrewAI integration. The review also suggested improvements to prompt separation, error handling, logging, and evaluation tracking.

### Findings checked against the implementation

- Prompt templates could be separated into a dedicated module. This remains an optional maintainability improvement.
- Claims that workflow orchestration and MLflow tracking were mixed together did not match the implementation. Tracking is handled separately in `src/tracking.py`.
- Claims that the research service also contains the CLI entry point did not match the implementation. The CLI is in `main.py`.
- The MLflow experiment name is configured centrally in `src/config.py`.
- The retrieval count is clamped to the collection size in `src/workflow.py`.
- Ollama requests already use explicit timeouts.
- Ingestion uses ChromaDB `upsert()` rather than deleting the collection.
- MLflow logs scalar parameters and metrics.

### Decision

The review was treated as advisory. Recommendations were checked against the actual code rather than applied blindly. No unnecessary architectural changes were made to the working pipeline.

---

## Review 2 – Testing, Reliability, and MLOps

**Focus:** Automated test coverage, external-service reliability, Ragas reproducibility, MLflow tracking, logging, resource management, and data isolation.

### Mentor findings

The mentor identified limited automated test coverage, missing error-path tests, potential Ragas variability, and opportunities to improve MLflow run metadata and operational logging.

The review also suggested ChromaDB persistence tests, Ollama failure tests, retrieval edge-case tests, MLflow lifecycle tests, and deterministic Ragas tests.

### Findings checked against the implementation

- The original four tests covered basic text splitting and empty-question validation. They did not exercise the main integration boundaries.
- The existing tests did not call live Ollama or ChromaDB services.
- MLflow experiment naming is configured in `src/config.py`, not hard-coded independently in tracking logic.
- The ingestion implementation uses `upsert()` and does not delete the collection on each run.
- Retrieval already limits `n_results` to the available collection count.
- Ollama embedding and generation calls already specify request timeouts.
- The real Ragas score can vary because it depends on LLM evaluation. Automated testing should not require a fixed live-model score.

### Improvements implemented

The automated test suite was expanded from four to ten tests.

New coverage includes:

1. Ollama embedding timeout propagation.
2. Retrieval result count clamping.
3. Empty collection handling.
4. Ollama generation HTTP failure propagation.
5. MLflow experiment, parameter, and metric logging using a temporary SQLite database.
6. Ragas evaluation workflow and evidence-file creation using mocked components.

### Validation results

- Pytest: 10 passed.
- Ruff lint: passed.
- Ruff formatting: passed; 17 files already formatted.
- Python compilation: passed.
- Real Ollama ingestion and research query: succeeded.
- CrewAI answer review: succeeded.
- Real Ragas Faithfulness evaluation: succeeded; latest score 1.0000.

Two dependency deprecation warnings were observed during testing. They did not cause test failures.

### Remaining opportunities

- Add a real ChromaDB persistence integration test.
- Consider additional MLflow failure-path tests.
- Consider structured logging and improved operational error reporting.
- Track additional run metadata such as dataset version or configuration when the project grows.
- Evaluate resource lifecycle management before deploying as a long-running service.

### Overall decision

The relevant testing and reliability recommendations were implemented without changing the working application pipeline. Optional production improvements remain documented rather than being represented as completed work.

---

## Final review status

Both CIA mentor reviews have been received and analysed. Relevant recommendations have been implemented and validated. Final Git review, commits, push, and PR update remain pending.
