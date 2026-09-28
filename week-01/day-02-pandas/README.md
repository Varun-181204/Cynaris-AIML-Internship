# W1D2 — Pandas for Data Manipulation

## Objective

Practice core Pandas data manipulation operations using a real Indian district-level dataset from the India Census 2011.

## Dataset

**Dataset:** India Census 2011 District-Level Data

* Rows: 640
* Columns: 118
* Format: CSV
* Source: India Census 2011 district dataset

The dataset contains district-level demographic and socioeconomic information such as population, literacy, gender, workers, households, and power-related statistics.

## Tasks Completed

### 1. Load and Inspect Dataset

Loaded the CSV dataset into a Pandas DataFrame and inspected:

* Dataset shape
* Column data types
* First 10 rows

### 2. Filter

Filtered districts where the population is greater than 1,000,000.

### 3. GroupBy

Grouped districts by state and calculated the total population for each state.

### 4. Merge

Created state-level population and district-count DataFrames and merged them using the common `State name` column.

### 5. Pivot Table

Created a pivot table containing the sum and mean population for each state.

### 6. Export

Exported the processed dataset into:

* CSV
* Parquet

Measured file sizes:

| Format  |          Size |
| ------- | ------------: |
| CSV     | 447,908 bytes |
| Parquet | 510,941 bytes |

The file-size comparison reflects the generated files for this dataset and configuration.

## Project Structure

```text
day-02-pandas/
├── data/
│   └── india_districts_census_2011.csv
├── docs/
├── outputs/
│   ├── cleaned_census_data.csv
│   ├── cleaned_census_data.parquet
│   └── results.txt
└── src/
    └── pandas_manipulation.py
```

## How to Run

From the repository root:

```powershell
python .\week-01\day-02-pandas\src\pandas_manipulation.py
```

To save the output as evidence:

```powershell
python .\week-01\day-02-pandas\src\pandas_manipulation.py > .\week-01\day-02-pandas\outputs\results.txt
```

## Dependencies

The project uses Python with:

* NumPy
* Pandas
* PyArrow

Dependencies are recorded in the root `requirements.txt`.

## Code Quality

* Functions have single responsibilities.
* Type hints are used for function signatures.
* `pathlib.Path` is used for file paths.
* Dataset paths are resolved relative to the project structure.
* Comments document the purpose of Pandas operations.
* Black formatting has been verified successfully.
* No hardcoded user-specific absolute paths are used.
