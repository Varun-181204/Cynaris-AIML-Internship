# W3D5 Self Review — Hyperparameter Tuning

## 1. Implementation Completed

Implemented hyperparameter tuning using:

- Support Vector Machine (SVM)
- K-Nearest Neighbours (KNN)
- `GridSearchCV`
- `RandomizedSearchCV`
- 5-fold cross-validation
- StandardScaler pipelines
- Stratified train-test split

## 2. Dataset

Used the Scikit-Learn Breast Cancer classification dataset.

- Samples: 569
- Features: 30
- Classes: malignant, benign
- Train/test split: 80/20
- Stratification: enabled
- Random seed: 42

## 3. Hyperparameter Search

### SVM — GridSearchCV

Search space:

- `C`: 0.1, 1, 10, 100
- `gamma`: scale, auto
- `kernel`: rbf, linear

Best parameters:

```text
C = 0.1
gamma = scale
kernel = linear
```

Best cross-validation accuracy:

```text
97.80%
```

Test accuracy:

```text
98.25%
```

### KNN — RandomizedSearchCV

Search space:

- `n_neighbors`: 3, 5, 7, 9, 11, 13, 15
- `weights`: uniform, distance
- `metric`: euclidean, manhattan

Search iterations:

```text
10
```

Best parameters:

```text
n_neighbors = 3
weights = uniform
metric = euclidean
```

Best cross-validation accuracy:

```text
96.92%
```

Test accuracy:

```text
98.25%
```

## 4. Data Leakage Check

No data leakage was identified.

`StandardScaler` is included inside each model pipeline, so scaling is fitted separately within each training fold during cross-validation.

The held-out test set is not used during hyperparameter selection.

## 5. Evaluation

The final test accuracy is calculated using the original held-out test set after the search has selected the best estimator.

Both tuned models achieved:

```text
98.25% test accuracy
```

## 6. Deliverables

The following outputs were generated successfully:

- `tuning_comparison.csv`
- `svm_grid_search_results.csv`
- `knn_random_search_results.csv`
- `tuning_comparison.png`

The summary CSV includes:

- Model
- Search method
- Best parameters
- Best CV accuracy
- Test accuracy

## 7. Code Quality

The implementation:

- Uses clear model pipelines.
- Uses reproducible train/test splitting.
- Uses explicit scoring with accuracy.
- Uses parallel processing with `n_jobs=-1`.
- Saves search results for later inspection.
- Generates a visual comparison of CV and test performance.
- Uses a dedicated output directory.

## 8. CIA Review

Two CIA Full Stack Mentor Mode reviews were completed.

The first review confirmed that the functional requirements were satisfied and identified optional production improvements.

The second review specifically checked correctness, leakage, search configuration, evaluation, and W3D5 deliverables. No critical issues remained.

## 9. Final Status

W3D5 implementation is complete and ready for commit.

### Required W3D5 Deliverables

- [x] Working hyperparameter tuning code
- [x] GridSearchCV
- [x] RandomizedSearchCV
- [x] Cross-validation
- [x] Held-out test evaluation
- [x] Best parameters
- [x] Model comparison CSV
- [x] Detailed search result CSV files
- [x] Accuracy comparison plot
- [x] CIA Interaction 1
- [x] CIA Interaction 2
- [x] Self-review