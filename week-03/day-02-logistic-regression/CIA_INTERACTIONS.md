# W3D2 CIA Interactions — Logistic Regression

## CIA Interaction 1 — Code Review

**Prompt:**  
Review my W3D2 Logistic Regression implementation for correctness. Check Logistic Regression training, classification metrics, confusion matrix, ROC-AUC, ROC curve, multiclass OvR vs Softmax, output files, and code quality. Give only the important issues and fixes. Keep the review concise.

**CIA Response Summary:**  
CIA reviewed the submitted code but incorrectly interpreted the task as W1D3 data cleaning. The review therefore did not provide a valid W3D2 requirement check.

The review was based on a partial code submission rather than the complete implementation.

---

## CIA Interaction 2 — Final Pre-Commit Review

**Prompt:**  
Do a final concise pre-commit review of my W3D2 Logistic Regression implementation.

The complete implementation includes:
- Breast Cancer binary classification
- LogisticRegression training
- coefficients and intercept
- accuracy, precision, recall, F1, ROC-AUC
- confusion matrix
- ROC curve
- multiclass comparison using One-vs-Rest and Softmax
- saved CSV/PNG outputs
- README and SELF_REVIEW.md

I only provided the first 100 lines of the code, so do not assume unseen sections are missing. Identify only genuine critical issues or inconsistencies.

**CIA Response Summary:**  
CIA confirmed the core Logistic Regression pipeline was correctly structured but suggested several improvements based on the partial code, including metrics/output verification, ROC and confusion-matrix visualization, pathlib usage, and a main guard.

Because only part of the implementation was supplied, these findings were checked against the completed W3D2 implementation before committing.

The completed implementation already contains the required evaluation metrics, confusion matrix, ROC curve, multiclass comparison, and saved output files.

---

## CIA Completion Status

- [x] CIA Interaction 1 completed
- [x] CIA Interaction 2 completed
- [x] CIA code review completed
- [x] CIA feedback reviewed
- [x] Feedback checked against completed implementation
- [x] W3D2 implementation successfully tested