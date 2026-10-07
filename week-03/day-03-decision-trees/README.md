# W3D3 — Decision Trees & Random Forests

## Objective

Implement and compare Decision Tree and Random Forest classifiers using scikit-learn. The Decision Tree is tuned to reduce overfitting, and the models are evaluated using test accuracy.

## Dataset

The Breast Cancer Wisconsin dataset from scikit-learn was used.

- Samples: 569
- Features: 30
- Classes: malignant, benign
- Train/test split: 80/20
- Stratification: enabled
- Random state: 42

## Models

### 1. Decision Tree

A baseline `DecisionTreeClassifier` was trained to establish a reference performance.

- Accuracy: 91.23%
- Tree depth: 7
- Leaves: 19

### 2. Tuned Decision Tree

The Decision Tree was tuned using:

- `max_depth=5`
- `min_samples_split=10`
- `min_samples_leaf=5`

Results:

- Accuracy: 92.11%
- Tree depth: 5
- Leaves: 12

The reduced depth and number of leaves make the model simpler and help control overfitting.

### 3. Random Forest

A `RandomForestClassifier` with 100 trees was trained.

- Accuracy: 95.61%
- `n_estimators=100`
- `random_state=42`

Random Forest achieved the highest test accuracy among the three models.

## Model Comparison

| Model | Accuracy |
|---|---:|
| Decision Tree | 91.23% |
| Tuned Decision Tree | 92.11% |
| Random Forest | 95.61% |

## Outputs

The following evidence files are generated:

- `outputs/model_comparison.csv`
- `outputs/decision_tree.png`
- `outputs/random_forest_confusion_matrix.png`
- `outputs/feature_importance.csv`

## Feature Importance

The Random Forest feature importance output identifies the features that contributed most to the model's predictions.

The top features included:

1. `worst area`
2. `worst concave points`
3. `worst radius`
4. `mean concave points`
5. `worst perimeter`

## Conclusion

The tuned Decision Tree improved over the baseline Decision Tree while using a shallower tree. Random Forest achieved the best test accuracy at 95.61%, demonstrating the benefit of combining multiple decision trees.

## Run

From the project root:

```powershell
python "week-03/day-03-decision-trees/src/decision_trees.py"
```