# W1D2 Self-Review — Pandas for Data Manipulation

## Practical Requirements

* [x] Loaded a real Indian dataset into a Pandas DataFrame.
* [x] Printed dataset shape.
* [x] Printed dataset data types.
* [x] Displayed the first 10 rows.
* [x] Implemented filtering.
* [x] Implemented groupby.
* [x] Implemented merge.
* [x] Implemented pivot_table.
* [x] Exported cleaned data to CSV.
* [x] Exported cleaned data to Parquet.
* [x] Compared CSV and Parquet file sizes.

## Code Quality

* [x] Used functions with single responsibilities.
* [x] Used `pathlib.Path` for file paths.
* [x] Added type hints to function signatures.
* [x] Added docstrings to functions.
* [x] Added comments explaining non-obvious operations.
* [x] Added explicit dataset path validation.
* [x] Ran Black formatting check successfully.
* [x] Verified the program executes successfully.

## Documentation

* [x] README created.
* [x] Project structure documented.
* [x] Dataset information documented.
* [x] Execution instructions documented.
* [x] Output file sizes documented.
* [x] CIA Mentor interactions documented.
* [x] Output screenshots added.

## CIA Mentor Interactions

* [x] CIA Interaction 1 completed and documented.
* [x] CIA Interaction 2 completed and documented.

See `docs/CIA_INTERACTIONS.md` for the prompts and response summaries.

## Git Requirements

* [ ] Minimum 2 commits completed.
* [ ] Changes pushed to GitHub.
* [ ] Pull Request created or updated.
* [ ] PR description includes what changed, why, and how to test.

## Viva Preparation

### 1. `loc` vs `iloc`

* `loc` selects data using labels.
* `iloc` selects data using integer positions.

### 2. `merge` vs `join` vs `concat`

* `merge` combines DataFrames using matching keys.
* `join` is mainly used for combining DataFrames based on indexes or specified keys.
* `concat` combines DataFrames along rows or columns.

### 3. `groupby` internals

`groupby` follows the split-apply-combine pattern:

1. Split the data into groups.
2. Apply an aggregation or transformation.
3. Combine the results into the output.

## Final Review

* **Implementation:** Complete
* **Output Evidence:** Complete
* **Documentation:** Complete
* **AI/CIA Evidence:** Complete
* **Git Workflow:** Pending
