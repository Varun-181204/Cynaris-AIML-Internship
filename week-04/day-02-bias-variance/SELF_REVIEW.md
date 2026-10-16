# W4D2 Self-Review

## Completed
- Implemented reproducible regression experiments.
- Used pipelines to keep scaling within cross-validation folds.
- Compared Ridge grid search, Ridge randomized search, and Lasso grid search.
- Analysed decision-tree depth using training and cross-validation RMSE.
- Generated CSV, JSON, and plot evidence.
- Verified the implementation with four passing tests.

## Key findings
Lasso achieved the lowest cross-validation RMSE of the tested regularised
models. The unrestricted decision tree nearly eliminated training error but
performed worse under cross-validation, indicating high variance/overfitting.

## Improvements
- Test min_samples_leaf and other tree regularisation controls.
- Visualise Ridge/Lasso coefficient shrinkage.
- Add further regression tests for search configuration and edge cases.
