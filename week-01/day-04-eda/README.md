# W1D4 — Exploratory Data Analysis

## Objective

Perform Exploratory Data Analysis (EDA) on the India Census 2011 district-level dataset using Pandas, Matplotlib, and Seaborn.

## Dataset

- Source: India Census 2011 district-level dataset
- Rows: 640
- Columns: 118
- Numeric columns: 116
- Text columns: 2

## EDA Performed

1. Generated descriptive statistics using `df.describe()`.
2. Inspected dataset structure using `df.info()`.
3. Checked missing values using `df.isnull().sum()`.
4. Examined numeric feature distributions.
5. Generated a correlation heatmap.
6. Identified the top 10 states by district count.
7. Documented five observations from the analysis.
8. Created a 200-word EDA narrative.

## Key Findings

- The dataset contains 640 district records.
- There are 116 numeric and 2 text columns.
- No missing values were found.
- No duplicate rows were found.
- District population ranges from approximately 8,004 to 11.06 million.
- Several demographic variables show strong correlations because they represent related population components.
- Uttar Pradesh has the highest number of district records in the dataset.

## Output Files

- `outputs/results.txt`
- `outputs/numeric_distributions.png`
- `outputs/correlation_heatmap.png`
- `outputs/top_10_state_counts.png`
- `docs/EDA_NARRATIVE.md`

## Project Structure

```text
day-04-eda/
├── data/
│   └── india_districts_census_2011.csv
├── docs/
│   └── EDA_NARRATIVE.md
├── outputs/
│   ├── correlation_heatmap.png
│   ├── numeric_distributions.png
│   ├── results.txt
│   └── top_10_state_counts.png
├── src/
│   └── eda_analysis.py
└── README.md