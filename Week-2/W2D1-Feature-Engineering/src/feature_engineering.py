import os

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "iris.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------
# 1. Categorical Encoding
# ---------------------------------------------------------

categorical_data = df[["species"]].copy()

# LabelEncoder
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(categorical_data["species"])

# OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
one_hot_encoded = one_hot_encoder.fit_transform(categorical_data)

# OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
ordinal_encoded = ordinal_encoder.fit_transform(categorical_data)

print("\n--- Encoding Results ---")
print("LabelEncoder:", label_encoded[:10])
print("OneHotEncoder:")
print(one_hot_encoded[:5])
print("OrdinalEncoder:", ordinal_encoded[:10])

print("\nEncoding trade-offs:")
print("- LabelEncoder: Converts categories into integer labels; suitable for target labels.")
print("- OneHotEncoder: Creates binary columns; avoids implying an order between categories.")
print("- OrdinalEncoder: Converts categories into ordered integers; useful when categories have meaningful order.")

# ---------------------------------------------------------
# 2. Feature Scaling
# ---------------------------------------------------------

numeric_features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

X_numeric = df[numeric_features]

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

X_standard = standard_scaler.fit_transform(X_numeric)
X_minmax = minmax_scaler.fit_transform(X_numeric)
X_robust = robust_scaler.fit_transform(X_numeric)

standard_df = pd.DataFrame(X_standard, columns=numeric_features)
minmax_df = pd.DataFrame(X_minmax, columns=numeric_features)
robust_df = pd.DataFrame(X_robust, columns=numeric_features)

print("\n--- Scaling Results ---")
print("\nStandardScaler:")
print(standard_df.head())

print("\nMinMaxScaler:")
print(minmax_df.head())

print("\nRobustScaler:")
print(robust_df.head())

# Distribution plots: before scaling
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Feature Distributions Before Scaling")

for ax, feature in zip(axes.ravel(), numeric_features):
    ax.hist(X_numeric[feature], bins=15, edgecolor="black")
    ax.set_title(feature)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distributions_before_scaling.png"))
plt.close()

# Distribution plots: after scaling
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Feature Distributions After Scaling")

for ax, feature in zip(axes.ravel(), numeric_features):
    ax.hist(standard_df[feature], bins=15, edgecolor="black")
    ax.set_title(f"{feature} - StandardScaler")
    ax.set_xlabel("Scaled Value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distributions_after_scaling.png"))
plt.close()

print("\nScaling distribution plots saved successfully.")

# ---------------------------------------------------------
# 3. Feature Selection - SelectKBest
# ---------------------------------------------------------

# Encode the target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["species"])

# SelectKBest cannot select 5 features because Iris has only 4
# numeric input features. Therefore, we select all 4 available
# features and rank them by ANOVA F-score.
k = min(5, X_numeric.shape[1])

selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X_numeric, y)

feature_scores = pd.DataFrame({
    "Feature": numeric_features,
    "F_Score": selector.scores_,
    "P_Value": selector.pvalues_,
})

feature_scores = feature_scores.sort_values(
    by="F_Score",
    ascending=False
)

print("\n--- SelectKBest Feature Selection ---")
print(f"Requested top 5 features; dataset contains {X_numeric.shape[1]} numeric features.")
print(f"Selected top {k} available features:")
print(feature_scores.to_string(index=False))

feature_scores.to_csv(
    os.path.join(OUTPUT_DIR, "feature_selection_scores.csv"),
    index=False
)

print("\nFeature selection results saved successfully.")