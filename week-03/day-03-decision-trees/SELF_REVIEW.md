# W3D3 Self-Review — Decision Trees & Random Forests

## Implementation Checklist

- [x] Loaded Breast Cancer classification dataset
- [x] Performed stratified train-test split
- [x] Trained baseline Decision Tree
- [x] Used Gini impurity for Decision Tree splitting
- [x] Tuned Decision Tree to reduce overfitting
- [x] Trained Random Forest
- [x] Compared model accuracy
- [x] Visualized the Decision Tree
- [x] Generated Random Forest confusion matrix
- [x] Calculated Random Forest feature importance
- [x] Saved model comparison CSV
- [x] Saved visualization PNG files
- [x] Saved feature importance CSV
- [x] Tested the implementation successfully
- [x] Added README documentation

## Results

| Model | Accuracy |
|---|---:|
| Decision Tree | 91.23% |
| Tuned Decision Tree | 92.11% |
| Random Forest | 95.61% |

The tuned Decision Tree improved over the baseline while reducing the tree depth from 7 to 5.

Random Forest achieved the highest test accuracy at 95.61%.

## CIA Full-Stack Mentor Review

Two CIA Full-Stack Mentor Mode interactions were completed for W3D3.

### Interaction 1

CIA reviewed the Decision Tree and Random Forest implementation for training, evaluation, tuning, visualization, feature importance, outputs, and code quality.

### Interaction 2

CIA performed a final pre-commit review and suggested several improvements. The suggestions were checked against the completed implementation.

Several suggested items were already implemented, including the fixed random state, train-test split, figure cleanup, and output handling. No unnecessary changes were made based on findings that did not apply to the completed code.

Detailed prompts and response summaries are recorded in `CIA_INTERACTIONS.md`.

## Final Verification

The implementation was executed successfully and produced all expected outputs.

- [x] Implementation complete
- [x] Output evidence generated
- [x] README completed
- [x] Self-review completed
- [x] CIA interactions completed and documented