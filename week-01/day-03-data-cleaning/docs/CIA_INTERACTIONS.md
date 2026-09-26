# W1D3 CIA Mentor Interactions

## Interaction 1 — Code Review

### Prompt

Review my W1D3 Pandas data-loading and cleaning workflow for code quality, maintainability, and internship standards. Check whether the functions have clear responsibilities, use pathlib correctly, handle errors explicitly, and follow Python formatting standards.

### Summary of Feedback

- Keep dataset loading separate from inspection and cleaning.
- Use `pathlib.Path` instead of hardcoded file paths.
- Add type hints and docstrings.
- Validate that the dataset exists before loading.
- Use Black for formatting.
- Avoid unnecessary warnings during execution.

### Changes Applied

- Used separate functions for loading, inspection, cleaning, and exporting.
- Added dataset path validation.
- Added type hints and docstrings.
- Applied Black formatting.
- Updated Pandas string-column selection to avoid the Pandas 3 warning.

---

## Interaction 2 — Data Cleaning Review

### Prompt

Review the data-cleaning approach for the India Census 2011 dataset. Check duplicate handling, missing-value handling, and whether the output should be validated after cleaning.

### Summary of Feedback

- Check duplicate rows before removing them.
- Measure missing values before and after cleaning.
- Use median values for missing numeric data when appropriate.
- Use a clear placeholder for missing text values.
- Verify the final dataset shape and output file.

### Changes Applied

- Added duplicate-row detection.
- Added duplicate removal.
- Added missing-value counts before and after cleaning.
- Added median-based numeric missing-value handling.
- Added `Unknown` handling for missing text values.
- Added final cleaned dataset shape validation.
- Exported the cleaned dataset to CSV.