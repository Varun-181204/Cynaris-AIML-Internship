# W1D3 — Data Loading, Cleaning & Inspection

## Objective

Load, inspect, clean, and export the India Census 2011 district-level dataset using Pandas.

## Dataset

- Source: India Census 2011 district-level dataset
- Rows: 640
- Columns: 118
- Format: CSV

## Operations Performed

1. Loaded the dataset using Pandas.
2. Inspected dataset shape.
3. Displayed the first five rows.
4. Checked missing values.
5. Checked duplicate rows.
6. Inspected data types.
7. Removed duplicate rows.
8. Filled missing numeric values using column medians.
9. Filled missing text values with `Unknown`.
10. Exported the cleaned dataset to CSV.

## Results

- Original shape: `(640, 118)`
- Duplicate rows found: `0`
- Missing values before cleaning: `0`
- Missing values after cleaning: `0`
- Final cleaned shape: `(640, 118)`

Because the source dataset contained no duplicate rows or missing values, the cleaning process did not change the number of rows.

## Project Structure

```text
day-03-data-cleaning/
├── data/
│   └── india_districts_census_2011.csv
├── docs/
├── outputs/
│   ├── cleaned_census_data.csv
│   └── results.txt
└── src/
    └── data_cleaning.py