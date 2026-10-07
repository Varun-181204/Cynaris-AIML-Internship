# W2D3 CIA Interactions

## CIA Interaction 1 — Technical Review

### Prompt
Review my W2D3 Handling Imbalanced Data — SMOTE implementation against the internship requirements. Check whether my class imbalance setup, train/test split, stratification, StandardScaler usage, and SMOTE implementation are technically correct. Pay particular attention to whether SMOTE is applied only to the training data and whether there are any data leakage risks. Also review code quality, reproducibility, and output evidence. Identify any specific issues or improvements needed.

### Key Feedback
- Train/test splitting should happen before SMOTE.
- Stratification should preserve class proportions between training and testing data.
- SMOTE should be applied only to the training data.
- Test data must remain untouched by synthetic oversampling.
- Preprocessing such as scaling should be fitted using training data only.
- Fixed random states improve reproducibility.
- Class distribution before and after SMOTE should be documented.
- Data leakage risks should be clearly explained.

### Action Taken
The W2D3 implementation performs a stratified train/test split before applying SMOTE. SMOTE is applied only to the training data, while the test set remains untouched. The implementation records class distributions before and after SMOTE and generates a visualization as output evidence.

---

## CIA Interaction 2 — Documentation & Deliverables Review

### Prompt
Act as a Full Stack Mentor and review my W2D3 documentation and deliverables. Check my README, self-review checklist, class distribution CSV, SMOTE visualization, explanation of why SMOTE is used, data leakage explanation, viva answers, and Git deliverables against the W2D3 requirements. Identify anything missing, unclear, or technically incorrect and suggest only the improvements relevant to this internship task.

### Key Feedback
- README should clearly explain why SMOTE is used.
- Class imbalance before and after SMOTE should be documented.
- Data leakage prevention should be explicitly explained.
- Output evidence should clearly show the effect of SMOTE.
- Viva answers should explain the main design decisions.
- Self-review should reflect the actual implementation.
- Git history should contain clear descriptive commits.

### Action Taken
The W2D3 README documents the purpose of SMOTE, train/test splitting before oversampling, data leakage prevention, before/after class distributions, output evidence, viva answers, and the self-review checklist. Two descriptive commits were created and the changes were pushed to the Week 2 branch.

### CIA Evidence Summary
Two Full Stack Mentor CIA interactions were completed for W2D3. The relevant feedback was reviewed against the actual W2D3 requirements and documented without adopting unrelated generic recommendations.