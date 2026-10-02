\# W2D5: End-to-End Preprocessing Pipeline



\## Objective



Build an end-to-end preprocessing pipeline using the Titanic dataset:



EDA → Missing Value Handling → Encoding → Scaling → ML-Ready Features



\## Dataset



The Titanic dataset contains 891 passenger records and 15 columns.



The preprocessing pipeline uses these features:



\- `pclass`

\- `sex`

\- `age`

\- `sibsp`

\- `parch`

\- `fare`

\- `embarked`



Target:



\- `survived`



Redundant target-related columns such as `alive` were excluded to avoid target leakage.



\## Implementation



\### 1. EDA



A survival distribution visualization was generated to understand the target distribution.



Output:



\- `survival\_distribution.png`



\### 2. Missing Value Handling



Missing values were identified before preprocessing.



| Feature | Missing Values |

|---|---:|

| age | 177 |

| embarked | 2 |



Numeric missing values are filled using the median.



Categorical missing values are filled using the most frequent category.



\### 3. Encoding



Categorical features are encoded using `OneHotEncoder`.



`handle\_unknown="ignore"` is used so unseen categories do not cause errors during transformation.



\### 4. Scaling



Numeric features are standardized using `StandardScaler`.



The scaler is included inside the preprocessing pipeline so preprocessing can be consistently applied to ML data.



\### 5. ColumnTransformer



`ColumnTransformer` applies separate preprocessing to numeric and categorical features.



This keeps the preprocessing workflow organized and reproducible.



\## Results



| Metric | Result |

|---|---:|

| Original rows | 891 |

| Original columns | 15 |

| Processed rows | 891 |

| Processed features | 10 |

| Remaining missing values | 0 |



The final ML-ready dataset contains 10 processed input features plus the `survived` target column.



\## Output Evidence



Generated files:



\- `missing\_values\_before.csv`

\- `preprocessing\_summary.csv`

\- `survival\_distribution.png`

\- `titanic\_ml\_ready.csv`



\## Viva Questions



\### 1. Explain what you built today and why you made your key design decisions.



I built an end-to-end Titanic preprocessing pipeline covering EDA, missing value handling, categorical encoding, numerical scaling, and ML-ready dataset export. Median imputation was selected for numeric missing values because it is less sensitive to outliers than the mean. Most-frequent imputation was used for categorical values. OneHotEncoder was used for categorical features, and StandardScaler was used for numerical features.



\### 2. What was the hardest part? How did you solve it?



The main challenge was handling different preprocessing requirements for numeric and categorical columns. I solved this using separate Scikit-learn pipelines combined with `ColumnTransformer`.



\### 3. If you had one more day, what would you improve?



I would add train/test splitting and cross-validation directly into the end-to-end pipeline, compare different scaling strategies, and evaluate a machine learning model using the processed features.



\## Self-Review Checklist



\- \[x] Titanic dataset loaded.

\- \[x] EDA completed.

\- \[x] Missing values identified.

\- \[x] Numeric missing values handled.

\- \[x] Categorical missing values handled.

\- \[x] Categorical features encoded.

\- \[x] Numeric features scaled.

\- \[x] ColumnTransformer used.

\- \[x] ML-ready dataset exported.

\- \[x] Output evidence generated.

\- \[x] No missing values remain after preprocessing.

\- \[x] Code tested successfully.

\- \[x] Viva questions answered.

\- \[x] Documentation completed.



\## Git



Branch:



`feat/aiml-W2-Varun-K-S`



Minimum two descriptive commits will be used for W2D5.

