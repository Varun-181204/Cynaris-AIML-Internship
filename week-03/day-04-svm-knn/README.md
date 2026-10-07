# W3D4 — SVM & KNN

## Objective

Implement and compare Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) classification models using scikit-learn.

## Dataset

The Breast Cancer dataset from scikit-learn was used.

- Samples: 569
- Features: 30
- Classes: Malignant and Benign
- Train/Test Split: 80/20
- Random State: 42

## Models

### SVM

A Support Vector Classifier using an RBF kernel was trained as the baseline model.

SVM hyperparameters were tuned using `GridSearchCV`.

### KNN

A K-Nearest Neighbors classifier with 5 neighbors was used as the baseline model.

KNN hyperparameters were tuned using `GridSearchCV`.

Both models use `StandardScaler` inside a scikit-learn pipeline.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| SVM | 98.25% | 98.61% | 98.61% | 98.61% |
| KNN | 95.61% | 95.89% | 97.22% | 96.55% |
| Tuned SVM | 98.25% | 98.61% | 98.61% | 98.61% |
| Tuned KNN | 97.37% | 96.00% | 100.00% | 97.96% |

## Best Hyperparameters

### SVM

```text
C = 0.1
gamma = scale
kernel = linear
```

### KNN

```text
metric = euclidean
n_neighbors = 7
weights = uniform
```

## Output Files

The following evidence files are generated in the `outputs` directory:

- `model_comparison.csv`
- `svm_confusion_matrix.png`
- `knn_confusion_matrix.png`

## How to Run

From the repository root:

```powershell
python week-03/day-04-svm-knn/src/svm_knn.py
```

The script trains the models, performs hyperparameter tuning, evaluates the models, prints classification reports, and saves the output files.

## Key Learning

SVM performed strongly on the dataset without tuning, while KNN improved from 95.61% to 97.37% after hyperparameter tuning.

The implementation demonstrates the importance of feature scaling for distance-based and margin-based machine learning algorithms.