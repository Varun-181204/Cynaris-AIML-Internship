# W2D1 - Feature Engineering & Encoding

## Objective

Apply categorical encoding, feature scaling, and feature selection techniques
using the Iris dataset.

## Dataset

- Dataset: Iris
- Rows: 150
- Numeric features: 4
- Target: species

## 1. Categorical Encoding

### LabelEncoder

Converts categorical target labels into integer values.

**Use:** Suitable for encoding a target variable.

**Trade-off:** Integer values may incorrectly imply an order if used directly
for nominal input features.

### OneHotEncoder

Converts categories into separate binary columns.

**Use:** Suitable for nominal categorical input features.

**Trade-off:** Can increase the number of features when many categories exist.

### OrdinalEncoder

Converts categories into integer values based on an order.

**Use:** Suitable when categories have meaningful ordinal relationships.

**Trade-off:** Using it for unordered categories can introduce a false
ordering.

## 2. Feature Scaling

### StandardScaler

Transforms features so they have approximately zero mean and unit variance.

### MinMaxScaler

Transforms values to a specified range, commonly 0 to 1.

### RobustScaler

Uses the median and interquartile range, making it more resistant to outliers.

### Distribution Evidence

- `outputs/distributions_before_scaling.png`
- `outputs/distributions_after_scaling.png`

## 3. SelectKBest Feature Selection

SelectKBest with ANOVA F-test was used to rank numeric features.

The dataset contains only 4 numeric input features, so selecting 5 features
is not possible. Therefore, all 4 available features were selected and ranked.

| Feature | F-Score | Why it matters |
|---|---:|---|
| petal_length | 1179.03 | Strong separation between Iris species. |
| petal_width | 959.32 | Strongly associated with differences between species. |
| sepal_length | 119.26 | Provides useful information for distinguishing species. |
| sepal_width | 47.36 | Provides additional class-separation information, although its F-score is lower than the other features. |

The feature scores are saved in:

`outputs/feature_selection_scores.csv`

## 4. Feature Leakage

Feature leakage occurs when information that would not be available at
prediction time is used during model training.

### Prevention

- Split training and testing data before fitting preprocessing operations.
- Fit encoders and scalers only on the training data.
- Apply the fitted transformations to validation/test data.
- Do not use target-derived information as an input feature.
- Use pipelines to keep preprocessing and model training separated correctly.

## 5. Output Evidence

The following files provide evidence of the implementation:

- `distributions_before_scaling.png`
- `distributions_after_scaling.png`
- `feature_selection_scores.csv`

## Viva Preparation

### Q1. When do you use OneHotEncoder vs OrdinalEncoder?

Use OneHotEncoder for nominal categories without a meaningful order.
Use OrdinalEncoder when the categories have a meaningful ranking.

### Q2. Why does StandardScaler not work well with outliers?

StandardScaler uses the mean and standard deviation. Extreme values can
strongly affect both, causing the scaled values to be influenced by outliers.
RobustScaler is less sensitive because it uses the median and interquartile
range.

### Q3. What is feature leakage and how can it be prevented?

Feature leakage occurs when information unavailable at prediction time
influences model training. It can be prevented by splitting data before
fitting preprocessing steps and ensuring transformations are fitted only
on training data.
