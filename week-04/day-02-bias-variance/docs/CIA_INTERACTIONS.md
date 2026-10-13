# W4D2 CIA Interactions — Bias-Variance Tradeoff & Regularisation

## CIA Interaction 1

### Mentor Focus
Design-level review of the bias-variance and regularisation implementation, including reproducibility, data splitting, cross-validation, model pipelines, hyperparameter search, decision-tree depth analysis, testing, and evidence.

### Relevant Feedback
- Use reproducible random seeds throughout the experiment.
- Keep preprocessing inside pipelines to reduce data leakage risk.
- Improve the visual explanation of bias and variance.
- Consider additional decision-tree regularisation settings.
- Strengthen tests and documentation.
- Consider comparing Ridge and Lasso coefficients to explain regularisation.

### Action Taken
- Confirmed that the implementation uses fixed random states for dataset generation, splitting, cross-validation, randomized search, and the decision tree.
- Kept StandardScaler inside the model pipelines.
- Generated training-versus-cross-validation RMSE analysis and comparison plots.
- Added automated tests and documented potential future improvements.
- Treated optional suggestions as improvements rather than claiming they were already implemented.

## CIA Interaction 2

### Mentor Focus
Interpret the experimental results, explain the regularisation parameter alpha, compare grid search with randomized search, and identify three practical improvements.

### Prompt
Review the W4D2 results using the reported decision-tree depths and RMSE values. Explain the bias-variance pattern, how alpha affects Ridge and Lasso, the difference between GridSearchCV and RandomizedSearchCV, and three useful improvements. Do not claim that code was executed unless execution evidence is provided.

### Relevant Feedback
The exact CIA response must be recorded here from the actual second interaction.

### Action Taken
Pending confirmation of the actual second CIA response.

## CIA Evidence Summary

- Two CIA mentor interactions are required for this task.
- Interaction 1 covered design and implementation review.
- Interaction 2 covers experimental interpretation and improvement recommendations.
- The documented results and actions should match the actual CIA responses and repository evidence.
