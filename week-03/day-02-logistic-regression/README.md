# W3D2 - Logistic Regression & Classification

## Objective

Implement Logistic Regression for binary classification and compare One-vs-Rest with Softmax for multi-class classification.

## Binary Classification

### Dataset

Breast Cancer Wisconsin dataset from scikit-learn.

- Samples: 569
- Features: 30
- Classes: Malignant and Benign
- Train/Test split: 80/20
- Feature scaling: StandardScaler

### Logistic Regression Results

| Metric | Score |
|---|---:|
| Accuracy | 0.9825 |
| Precision | 0.9861 |
| Recall | 0.9861 |
| F1-Score | 0.9861 |
| ROC-AUC | 0.9954 |

### Confusion Matrix

```text
[[41  1]
 [ 1 71]]