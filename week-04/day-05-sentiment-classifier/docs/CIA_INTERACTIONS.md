# CIA Interaction Log — W4D5

## Interaction 1 — Implementation review
- Date/time: Not recorded.
- Prompt submitted: Review the W4D5 sentiment classifier implementation, model pipelines, train/test split, evaluation metrics, plots, reproducibility, and synthetic dataset limitations.
- CIA feedback: Check pipeline structure and imports, ensure preprocessing is fitted only on training data, generate the required reports and plots, and document the limitations of synthetic data.
- Changes made: Verified the implementation structure, retained fixed random seeds and pipeline-based preprocessing, and documented the synthetic dataset limitations.

## Interaction 2 — Testing and evidence review
- Date/time: Not recorded.
- Prompt submitted: Review W4D5 evaluation results and evidence files, identify missing metrics or documentation, and check consistency between the JSON and CSV outputs.
- CIA feedback: F1-score was missing from the metrics JSON and comparison CSV; include test output in the README evidence list and verify that both models appear in the evaluation evidence.
- Changes made: Added F1-score to the JSON and CSV outputs, included `outputs/test_results.txt` in the README, and reran the classifier and tests. All five tests passed.

## Review status
- [x] Interaction 1 completed and logged.
- [x] Interaction 2 completed and logged.
- [x] Relevant findings addressed.
