# W2D4 CIA Interactions

## CIA Interaction 1 — Technical Review

### Prompt
Review my W2D4 Train/Test Split & Cross-Validation implementation against the internship requirements. Check my train/test split, stratification, 5-fold StratifiedKFold cross-validation, StandardScaler usage, Pipeline implementation, data leakage prevention, evaluation methodology, reproducibility, and output evidence. Identify any technical issues or improvements relevant to the W2D4 task.

### Key Feedback
- Train/test data must remain mutually exclusive.
- Stratification should preserve class proportions.
- 5-fold StratifiedKFold is appropriate for classification.
- StandardScaler should be fitted only on training data.
- Using StandardScaler inside a Pipeline helps prevent leakage during cross-validation.
- Random states should be fixed for reproducibility.
- Cross-validation scores should be recorded and summarized.
- Test-set evaluation should be performed only after cross-validation/training.

### Action Taken
The W2D4 implementation uses an 80/20 stratified train/test split, 5-fold StratifiedKFold cross-validation, and StandardScaler inside a Scikit-learn Pipeline. Mean CV accuracy and standard deviation are calculated, and the final model is evaluated on the unseen test set.

---

## CIA Interaction 2 — Documentation & Deliverables Review

### Prompt
Act as a Full Stack Mentor and review my W2D4 documentation and deliverables. Check my README, self-review checklist, cross-validation results CSV, accuracy visualization, viva answers, and Git workflow against the W2D4 requirements. Identify anything missing, unclear, or technically incorrect, and suggest only improvements relevant to this internship task.

### Key Feedback
- README should clearly explain train/test splitting and cross-validation.
- The cross-validation results should include individual fold scores and a summary.
- Output visualization should clearly communicate the CV results.
- Data leakage prevention should be documented.
- The self-review checklist should match the actual implementation.
- Viva answers should explain the key design decisions.
- Git commits should clearly describe the implemented work.

### Action Taken
The W2D4 README documents the train/test split, stratification, cross-validation, Pipeline-based scaling, leakage prevention, results, output evidence, viva answers, and self-review checklist. The CV fold scores, mean accuracy, standard deviation, and test accuracy are saved and documented.

### CIA Evidence Summary
Two Full Stack Mentor CIA interactions were completed for W2D4. The relevant feedback was reviewed against the actual W2D4 requirements and documented without adopting unrelated generic recommendations.