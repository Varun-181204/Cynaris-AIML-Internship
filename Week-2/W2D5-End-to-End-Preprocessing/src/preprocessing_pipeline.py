import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "titanic.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load the Titanic dataset.
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# Keep relevant columns and remove redundant target-related columns.
features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]
target = "survived"

X = df[features]
y = df[target]


# Record missing values before preprocessing.
missing_before = X.isnull().sum()
missing_before.to_csv(
    os.path.join(OUTPUT_DIR, "missing_values_before.csv"),
    header=["missing_count"],
)

print("\nMissing values before preprocessing:")
print(missing_before)


# Define numeric and categorical features.
numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

categorical_features = [
    "sex",
    "embarked",
]


# Numeric pipeline:
# 1. Fill missing numeric values with the median.
# 2. Standardize numeric features.
numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)


# Categorical pipeline:
# 1. Fill missing categorical values with the most frequent value.
# 2. Convert categories into one-hot encoded features.
categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


# Combine numeric and categorical preprocessing.
preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# Fit and transform the feature data.
X_processed = preprocessor.fit_transform(X)


# Get the generated feature names.
feature_names = preprocessor.get_feature_names_out()

processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names,
)

# Add the target column.
processed_df[target] = y.values


# Export ML-ready features.
output_path = os.path.join(
    OUTPUT_DIR,
    "titanic_ml_ready.csv",
)

processed_df.to_csv(output_path, index=False)


# Verify that the processed dataset contains no missing values.
remaining_missing = processed_df.isnull().sum().sum()

print("\nProcessed dataset shape:", processed_df.shape)
print("Remaining missing values:", remaining_missing)
print("ML-ready dataset saved to:", output_path)


# Create an EDA visualization of survival counts.
survival_counts = df[target].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(
    ["Did Not Survive", "Survived"],
    survival_counts.values,
)
plt.xlabel("Outcome")
plt.ylabel("Number of Passengers")
plt.title("Titanic Survival Distribution")
plt.tight_layout()

eda_output = os.path.join(
    OUTPUT_DIR,
    "survival_distribution.png",
)

plt.savefig(eda_output)
plt.close()

print("EDA visualization saved to:", eda_output)


# Save a summary of the preprocessing result.
summary = pd.DataFrame(
    {
        "metric": [
            "original_rows",
            "original_columns",
            "processed_rows",
            "processed_features",
            "remaining_missing_values",
        ],
        "value": [
            df.shape[0],
            df.shape[1],
            processed_df.shape[0],
            processed_df.shape[1] - 1,
            remaining_missing,
        ],
    }
)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "preprocessing_summary.csv"),
    index=False,
)

print("Preprocessing completed successfully.")