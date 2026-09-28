# W1D3 Self-Review — Data Loading, Cleaning & Inspection

## Practical Requirements

- [x] Loaded the India Census 2011 dataset.
- [x] Inspected dataset shape.
- [x] Displayed sample records.
- [x] Checked missing values.
- [x] Checked duplicate rows.
- [x] Inspected data types.
- [x] Removed duplicate rows.
- [x] Handled missing numeric values.
- [x] Handled missing text values.
- [x] Exported cleaned data.
- [x] Generated output evidence.

## Code Quality

- [x] Used functions with single responsibilities.
- [x] Used `pathlib.Path`.
- [x] Added type hints.
- [x] Added docstrings.
- [x] Added explicit file validation.
- [x] Ran Black successfully.
- [x] Verified the program executes successfully.
- [x] Removed avoidable Pandas warnings.

## Documentation

- [x] README created.
- [x] Project structure documented.
- [x] Dataset information documented.
- [x] Execution instructions documented.
- [x] Results documented.
- [x] CIA mentor interactions documented.

## Git Requirements

- [x] Minimum 2 commits completed.
- [x] Changes pushed to GitHub.
- [x] Existing Week 1 Pull Request updated.
- [x] PR description includes what changed, why, and how to test.

## Viva Preparation

### 1. What did you build?

A Pandas-based data loading and cleaning workflow for the India Census 2011 district dataset. It loads the data, inspects its quality, removes duplicates, handles missing values, and exports the cleaned dataset.

### 2. What was the hardest part?

Handling the Pandas 3 string dtype behavior required explicitly selecting `str` columns instead of relying on the deprecated `object` selection.

### 3. If you had one more day, what would you improve?

I would add automated tests for the loading and cleaning functions, stronger column-level validation, and additional data-quality checks.

## Final Review

- **Implementation:** Complete
- **Output Evidence:** Complete
- **Documentation:** Complete
- **AI/CIA Evidence:** Complete
- **Git Workflow:** Complete