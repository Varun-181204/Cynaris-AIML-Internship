# W3D4 Self-Review — SVM & KNN

## Implementation Checklist

- [x] Loaded Breast Cancer classification dataset
- [x] Performed stratified train-test split
- [x] Applied feature scaling using StandardScaler
- [x] Trained baseline SVM
- [x] Trained baseline KNN
- [x] Tuned SVM using GridSearchCV
- [x] Tuned KNN using GridSearchCV
- [x] Evaluated accuracy
- [x] Evaluated precision
- [x] Evaluated recall
- [x] Evaluated F1-score
- [x] Generated SVM confusion matrix
- [x] Generated KNN confusion matrix
- [x] Compared baseline and tuned models
- [x] Saved model comparison CSV
- [x] Saved confusion matrix PNG files
- [x] Printed classification reports
- [x] Tested the implementation successfully

## Results

| Model | Accuracy |
|---|---:|
| SVM | 98.25% |
| KNN | 95.61% |
| Tuned SVM | 98.25% |
| Tuned KNN | 97.37% |

### Best Hyperparameters

**SVM**
- C: 0.1
- Gamma: scale
- Kernel: linear

**KNN**
- Metric: euclidean
- Neighbors: 7
- Weights: uniform

Tuning improved KNN accuracy from 95.61% to 97.37%.

SVM maintained the same test accuracy of 98.25% after tuning.

## CIA Full-Stack Mentor Review

Two CIA Full-Stack Mentor Mode interactions were completed for W3D4.

### Interaction 1

CIA reviewed the SVM and KNN implementation for training, scaling, splitting, tuning, evaluation, confusion matrices, outputs, and code quality.

### Interaction 2

CIA performed a final pre-commit review. The suggestions were checked against the complete implementation, and changes were not made where the implementation already satisfied the recommendation.

Detailed prompts and response summaries are recorded in `CIA_INTERACTIONS.md`.

## Final Verification

The implementation was executed successfully and produced all expected outputs.

- [x] Implementation complete
- [x] Output evidence generated
- [x] CIA interactions completed
- [x] Self-review completed
- [x] Results verified