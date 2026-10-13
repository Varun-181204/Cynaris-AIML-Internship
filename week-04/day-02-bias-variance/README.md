# W4D2: Bias-Variance Tradeoff and Regularisation

## Objective
Study the bias-variance tradeoff, compare Ridge and Lasso regularisation,
evaluate grid and randomized hyperparameter searches, and save reproducible
results and visual evidence.

## Implementation
- Reproducible synthetic regression dataset.
- Train/test split and five-fold cross-validation.
- StandardScaler and regularised models inside scikit-learn pipelines.
- Ridge GridSearchCV, Ridge RandomizedSearchCV, and Lasso GridSearchCV.
- Decision-tree depth experiments to illustrate underfitting and overfitting.
- CSV, JSON, and PNG outputs.
- Automated unit tests.

## Results
- Lasso Grid Search: CV RMSE 18.1513, test RMSE 17.0734, test R² 0.9883.
- Ridge Grid Search: CV RMSE 18.1627, test RMSE 17.1211, test R² 0.9882.
- Ridge Randomized Search: CV RMSE 18.1627, test RMSE 17.1211, test R² 0.9882.
- Decision-tree depth 5 had the lowest cross-validation RMSE among the tested depths.
- The unrestricted tree achieved almost zero training RMSE but worse validation
  performance, illustrating overfitting.

## Verification
All four tests passed. Run the script and tests using the commands documented
in the repository's W4D2 instructions.

## Limitations and future improvements
- Test additional tree controls such as min_samples_leaf.
- Compare Ridge and Lasso coefficient shrinkage.
- Expand tests to cover model-selection and reproducibility edge cases.
