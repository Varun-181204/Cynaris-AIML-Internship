# CIA Full Stack Mentor Interactions — W4D2

## Interaction 1 — Design and Implementation Review

### Prompt
Reviewed the W4D2 Bias-Variance Tradeoff and Regularisation implementation, including the regression dataset, train/test split, cross-validation, preprocessing pipeline, Ridge/Lasso regularisation, hyperparameter searches, decision-tree depth experiments, output artifacts, and tests.

### Relevant Feedback
- Use reproducible random states for dataset generation, splitting, cross-validation, and randomized search.
- Keep preprocessing such as feature scaling inside a pipeline to prevent data leakage.
- Separate the test set from model selection and hyperparameter tuning.
- Use the appropriate negative RMSE scoring convention with scikit-learn search utilities.
- Make overfitting visible through training-versus-validation results and plots.
- Consider additional tree hyperparameters, coefficient visualisation, stronger leakage tests, clearer documentation, and experiment tracking.

### Action Taken
- Used fixed random states for reproducible experiments.
- Kept StandardScaler inside the Ridge/Lasso pipelines.
- Split the dataset before training and used cross-validation for model selection.
- Used `neg_root_mean_squared_error` for hyperparameter searches.
- Generated CSV/JSON results and plots, added unit tests, and documented the experiment.

## Interaction 2 — Results Interpretation and Improvement Review

### Prompt
Review the W4D2 results using the reported decision-tree depths and RMSE values. Explain the bias-variance pattern, how alpha affects Ridge and Lasso, the difference between GridSearchCV and RandomizedSearchCV, and three useful improvements. Do not claim that code was executed unless execution evidence is provided.

### Relevant Feedback
- Shallow trees show high bias because both training and cross-validation errors are high.
- Increasing tree depth reduces training error but increases the train/CV error gap, indicating overfitting.
- Ridge and Lasso penalise coefficient magnitudes; increasing alpha strengthens regularisation and can eventually cause underfitting.
- GridSearchCV evaluates every supplied parameter combination, while RandomizedSearchCV samples a fixed number of candidates and can explore broad parameter spaces more efficiently.
- Recommended improvements:
  1. Tune hyperparameters, including regularisation alpha and relevant tree parameters.
  2. Scale features for Ridge/Lasso inside a pipeline.
  3. Compare models consistently using cross-validation mean and standard deviation.
- Optional improvements include experiment tracking, training/validation curves, and coefficient or feature-importance visualisations.

### Action Taken
- Recorded the mentor's interpretation and improvement recommendations.
- Confirmed that the existing implementation uses pipelines for scaling, reproducible cross-validation, and both grid and randomized hyperparameter search.
- The current tree-depth results show depth 5 has the lowest CV RMSE among the tested depths (124.48). Depth 8 has CV RMSE 133.11, and unrestricted depth has CV RMSE 131.24.
- Ridge/Lasso results have test RMSE around 17.1 and CV RMSE around 18.2. These results are from a different model experiment than the tree-depth analysis, so they should not be presented as a strictly controlled comparison.
- The existing four unit tests passed. No additional code changes are claimed from this review.

## Evidence
- Implementation: `src/bias_variance.py`
- Tests: `tests/test_bias_variance.py`
- Results and plots: `outputs/`
- Task documentation: `README.md`
