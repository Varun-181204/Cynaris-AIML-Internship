# W2D1 CIA Interactions

## CIA Interaction 1 — Technical Review

### Prompt
Review my W2D1 Feature Engineering & Encoding implementation against the internship requirements. Check my use of LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler, feature distribution plots, and SelectKBest. Check for correctness, data leakage risks, code quality, and whether the implementation satisfies the task. Identify any issues or improvements needed.

### Key Feedback
- Use LabelEncoder primarily for the target variable.
- Use OneHotEncoder for nominal categorical features.
- Use OrdinalEncoder only when categories have a meaningful order.
- Fit encoders, scalers, and feature selectors only on training data to prevent leakage.
- Document the purpose and trade-offs of the encoding methods.
- Compare distributions before and after scaling.
- Use an appropriate scoring function with SelectKBest.
- Document the reason for the selected value of `k`.
- Keep the implementation reproducible and readable.

### Action Taken
The W2D1 implementation uses all three requested encoders, all three requested scalers, distribution plots, and SelectKBest. Feature leakage prevention and encoder trade-offs are documented in the README. Since the Iris dataset has only four numeric predictors, the implementation ranks the four available features rather than selecting five unavailable features.

---

## CIA Interaction 2 — Documentation Review

### Prompt
Act as a Full Stack Mentor and review my W2D1 documentation and deliverables. Check whether my README, self-review checklist, output evidence, feature selection results, encoder trade-offs, feature leakage explanation, and viva answers are sufficient for submission. Identify anything missing, unclear, or technically incorrect and suggest specific improvements.

### Key Feedback
- Ensure the README clearly explains the implementation and design decisions.
- Keep the self-review checklist aligned with the actual work completed.
- Include output evidence for encoding, scaling, and feature selection.
- Document encoder trade-offs and feature leakage prevention.
- Explain the rationale behind feature selection.
- Keep viva answers concise but technically justified.
- Maintain reproducibility and clear execution instructions.

### Action Taken
The W2D1 README and self-review checklist document the required encoding, scaling, feature selection, leakage prevention, output evidence, and viva answers. Separate scaler distribution plots and feature selection results are included in the outputs folder.

### CIA Evidence Summary
Two Full Stack Mentor CIA interactions were completed for W2D1. The feedback was reviewed against the actual Cynaris W2D1 requirements, and relevant improvements were incorporated into the implementation and documentation.