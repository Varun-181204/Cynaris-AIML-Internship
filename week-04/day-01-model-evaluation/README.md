# W4D1 — Model Evaluation Metrics

## Objective
Evaluate a binary classification model using a reproducible train/test split, classification metrics, stratified cross-validation, ROC-AUC, and learning curves.

## Dataset and model
- Dataset: reproducibly generated synthetic binary classification dataset.
- Samples: 1,500 total.
- Features: 20, including 8 informative and 4 redundant features.
- Class distribution: approximately 70% negative and 30% positive.
- Split: 80% training and 20% testing, with stratification.
- Model: StandardScaler + LogisticRegression in a scikit-learn Pipeline.
- Cross-validation: StratifiedKFold with 5 folds on the training split.

The scaler is inside the Pipeline so it is fitted independently within each training fold, avoiding leakage from validation folds.

## Held-out test results

| Metric | Score |
|---|---:|
| Accuracy | 0.8067 |
| Precision | 0.8511 |
| Recall | 0.4396 |
| F1 score | 0.5797 |
| ROC-AUC | 0.8393 |

Precision is relatively high, while recall is lower. The model identifies a proportion of positive predictions correctly but misses a meaningful share of actual positive cases. The appropriate balance depends on the application and the cost of false positives versus false negatives.

ROC-AUC measures ranking/discrimination across thresholds; it is not the same as accuracy.

## Stratified 5-fold cross-validation

Cross-validation was performed on the training split. Reported values are mean ± standard deviation.

| Metric | Mean ± standard deviation |
|---|---:|
| Precision | 0.7649 ± 0.0528 |
| Recall | 0.5332 ± 0.0592 |
| F1 score | 0.6238 ± 0.0292 |
| ROC-AUC | 0.8114 ± 0.0289 |

Stratification maintains a similar class proportion across folds. The standard deviation provides a simple indication of variation between folds.

## Learning curve and threshold experiment
The learning curve compares training and validation ROC-AUC at increasing training-set sizes. The shaded bands show one standard deviation across cross-validation folds.

A unit test verifies that reducing the classification threshold from 0.5 to 0.3 does not reduce recall for the same predicted probabilities. Lowering a threshold can also increase false positives. Threshold selection should be performed using validation data, not by repeatedly tuning against the final test set.

## Generated evidence
The `outputs/` directory contains:
- `evaluation_metrics.csv`
- `classification_report.txt`
- `confusion_matrix.png`
- `roc_curve.png`
- `learning_curve.png`

## Run the experiment

From the repository root, with the project virtual environment activated:

```powershell
python .\week-04\day-01-model-evaluation\src\model_evaluation.py
```

## Run tests

```powershell
python -m unittest discover `
    -s .\week-04\day-01-model-evaluation\tests `
    -p "test_*.py" -v
```

## Test status
All four unit tests passed during the latest run:
1. Dataset reproducibility and binary labels.
2. Recall behavior when the decision threshold is lowered.
3. Metric calculations on known predictions.
4. Pipeline and stratified cross-validation behavior.

## Limitations and next steps
- The dataset is synthetic, so results do not establish real-world performance.
- ROC-AUC should be considered alongside precision, recall, F1, and the confusion matrix.
- For imbalanced classification, consider precision-recall curves and average precision.
- If selecting a threshold, use a separate validation strategy and keep the final test set for final evaluation.
- Compare a class-weighted model or other classifier only as a separately documented experiment.
