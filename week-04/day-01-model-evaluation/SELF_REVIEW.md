# W4D1 Self-Review — Model Evaluation Metrics

## Implementation checklist
- [x] Reproducible binary classification dataset.
- [x] Stratified train/test split.
- [x] Preprocessing and logistic regression in a Pipeline.
- [x] Accuracy, precision, recall, F1, and ROC-AUC.
- [x] Stratified 5-fold cross-validation on the training split.
- [x] Learning curve to inspect training and validation behavior.
- [x] Confusion matrix, ROC curve, classification report, and CSV evidence.
- [x] Four automated unit tests.
- [x] Two CIA mentor interactions recorded in `docs/CIA_INTERACTIONS.md`.

## Latest verification
The experiment ran successfully and all four unit tests passed.

Held-out test metrics:
- Accuracy: 0.8067
- Precision: 0.8511
- Recall: 0.4396
- F1 score: 0.5797
- ROC-AUC: 0.8393

Stratified five-fold cross-validation mean ROC-AUC: 0.8114, standard deviation: 0.0289.

## What the results mean
The test precision is higher than test recall. The model misses a substantial portion of positive cases at the default decision threshold, so accuracy alone would not adequately describe its behavior. Threshold changes should be evaluated against the intended precision/recall trade-off using validation data.

The learning curve helps inspect how model performance changes as training data increases. The dataset is synthetic, so these metrics are educational evidence rather than a claim about real-world performance.

## Test coverage
1. Dataset generation is reproducible and labels are binary.
2. Lowering the threshold from 0.5 to 0.3 does not reduce recall for the same probabilities.
3. Metric calculations match expected values for known predictions.
4. The model pipeline and stratified cross-validation return valid results.

## Remaining review before submission
- [ ] Confirm the CIA interaction log contains both actual interactions and their key takeaways.
- [ ] Review generated output files and inspect the plots.
- [ ] Review Git diff and confirm only Week 4 Day 1 files are included.
- [ ] Create at least two descriptive commits.
- [ ] Push the Week 4 branch and open the pull request.
