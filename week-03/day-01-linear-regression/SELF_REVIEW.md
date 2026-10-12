# W3D1 Self-Review — Linear Regression

## Implementation Checklist

- [x] Loaded a real-world regression dataset
- [x] Split data into training and testing sets
- [x] Trained Linear Regression
- [x] Printed Linear Regression coefficients
- [x] Printed Linear Regression intercept
- [x] Calculated MSE
- [x] Calculated RMSE
- [x] Calculated MAE
- [x] Calculated R²
- [x] Generated predicted-vs-actual plot
- [x] Generated residual plot
- [x] Added Ridge Regression
- [x] Added Lasso Regression
- [x] Compared all three models
- [x] Saved model comparison results to CSV
- [x] Used reproducible random_state=42
- [x] Created required output directory before saving files

## Model Results

The three regression models achieved the following test-set R² scores:

- Linear Regression: R² = 0.5758
- Ridge: R² = 0.5759
- Lasso: R² = 0.5773

Lasso achieved the highest R² among the three evaluated models.

## CIA Full-Stack Mentor Review

Two CIA Full-Stack Mentor Mode interactions were completed for W3D1.

### Interaction 1

CIA reviewed the Linear Regression implementation for:

- Correctness
- Requirement coverage
- Potential bugs
- scikit-learn compatibility
- Output and file handling
- Code quality

The review was based on a partial code submission. Some findings therefore referred to code that was not included in the submitted excerpt.

The `target_names` point was independently verified against the installed scikit-learn version and confirmed to be supported.

### Interaction 2

CIA performed a final pre-commit style review covering:

- Linear Regression, Ridge, and Lasso
- Regression metrics
- Output generation
- CSV persistence
- Output directory handling
- scikit-learn compatibility
- Remaining implementation issues

The review again used the partial code provided to CIA. The findings were compared against the completed implementation before committing.

## CIA Completion

- [x] CIA Interaction 1 completed
- [x] CIA Interaction 2 completed
- [x] CIA code review completed
- [x] CIA feedback reviewed
- [x] Feedback checked against the completed implementation

Detailed prompts and response summaries are recorded in:

`CIA_INTERACTIONS.md`

## Final Verification

The completed W3D1 implementation was executed successfully.

Generated outputs:

- `outputs/model_comparison.csv`
- `outputs/predicted_vs_actual.png`
- `outputs/residuals.png`

The implementation is complete and ready for the W3D1 submission workflow.