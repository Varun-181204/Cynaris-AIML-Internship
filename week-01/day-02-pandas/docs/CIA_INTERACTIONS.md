# W1D2 CIA Interactions — Pandas for Data Manipulation

## Interaction 1

**Prompt:**

Explain the difference between Pandas `groupby()`, `merge()`, and `pivot_table()` with simple examples. Also mention one common mistake to avoid when using each.

**CIA Response Summary:**

CIA explained that:

* `groupby()` follows the split-apply-combine approach and is mainly used for aggregation and transformation by categorical groups.
* `merge()` performs SQL-style joins between DataFrames using matching keys.
* `pivot_table()` reshapes data into a summarized matrix using index, columns, values, and an aggregation function.
* Common issues include incorrect aggregation of non-numeric columns, mismatched join-key data types, and missing values in pivot tables.

**What I learned:**

The three operations serve different purposes: `groupby()` summarizes data by groups, `merge()` combines related datasets, and `pivot_table()` creates a structured summary by reshaping and aggregating data.

---

## Interaction 2

**Prompt:**

Review my W1D2 Pandas implementation from a production-oriented perspective. What improvements should I consider for data validation, memory efficiency, error handling, file paths, and exporting CSV/Parquet? Keep the recommendations relevant to a small AI/ML internship project.

**CIA Response Summary:**

CIA recommended:

* Validating dataset schemas, data types, and value ranges.
* Using `pathlib.Path` instead of hard-coded file paths.
* Considering memory optimization through numeric downcasting and categorical data types.
* Using chunked reading for very large datasets.
* Adding explicit error handling and custom validation exceptions where appropriate.
* Using Parquet for efficient downstream data processing and CSV for human-readable inspection.
* Adding logging for important processing information.

**What I learned:**

Production-oriented Pandas code should be reproducible, validated, memory-conscious, and explicit about errors and file paths. The recommendations also showed that optimization techniques such as chunking should be applied based on dataset size and actual resource requirements rather than automatically.

## AI Usage Reflection

The CIA interactions helped validate the W1D2 implementation and identify production-oriented improvements. I used the suggestions as guidance and did not directly copy the generated implementation into the project.
