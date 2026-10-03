# W2D5 CIA Interactions

## CIA Interaction 1 — Technical Review

### Prompt
Review my W2D5 End-to-End Preprocessing Pipeline against the official requirements. Check my Titanic preprocessing implementation for EDA, missing-value handling, categorical encoding, numerical scaling, ColumnTransformer usage, data leakage risks, output evidence, and code quality. Identify any issues or improvements I should make before final submission.

### Key Feedback
- Missing-value handling should be performed before scaling.
- Numeric and categorical features should be processed through separate pipelines.
- ColumnTransformer is appropriate for applying different preprocessing steps to different feature groups.
- OneHotEncoder with `handle_unknown="ignore"` helps handle unseen categories safely.
- Preprocessing transformations should be fitted only on training data when the pipeline is used for model evaluation.
- Data leakage should be explicitly considered and prevented.
- Output evidence should clearly demonstrate the preprocessing results.
- The preprocessing design should remain reproducible and readable.

### Action Taken
The W2D5 implementation uses separate numeric and categorical pipelines inside a ColumnTransformer. Numeric missing values are handled with median imputation before StandardScaler, while categorical missing values are handled with most-frequent imputation before OneHotEncoder. The implementation verifies that no missing values remain and exports the ML-ready dataset along with preprocessing evidence.

---

## CIA Interaction 2 — Documentation & Deliverables Review

### Prompt
Act as a Full Stack Mentor and review my W2D5 documentation and deliverables. Check whether my README, self-review checklist, viva answers, output evidence, preprocessing summary, missing-value report, ML-ready Titanic dataset, and Git deliverables satisfy the W2D5 requirements. Identify anything missing, unclear, or technically incorrect and suggest only improvements relevant to this internship task.

### Key Feedback
- README should clearly explain the preprocessing workflow and design decisions.
- Self-review checklist should accurately reflect the implementation.
- Viva answers should explain the main preprocessing decisions.
- Missing-value evidence should show the data-quality issues before preprocessing.
- The preprocessing summary should document the transformations applied.
- The ML-ready dataset should contain no remaining missing values.
- Output files should provide clear evidence that the pipeline executed successfully.
- Git commits should use clear, descriptive messages.

### Action Taken
The W2D5 README documents the Titanic dataset, preprocessing workflow, missing-value handling, encoding, scaling, results, output evidence, viva answers, and self-review checklist. The implementation generates a missing-value report, preprocessing summary, survival visualization, and ML-ready Titanic CSV. Two descriptive Git commits were also created and pushed.

### CIA Evidence Summary
Two Full Stack Mentor CIA interactions were completed for W2D5. The relevant feedback was reviewed against the actual W2D5 implementation and requirements. Unrelated generic recommendations from the CIA response were not adopted because they did not match the implemented project.