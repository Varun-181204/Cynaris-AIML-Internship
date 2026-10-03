import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "iris.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load the Iris dataset.
df = pd.read_csv(DATA_PATH)

feature_columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

X = df[feature_columns]
y = df["species"]


# Split the data into training and testing sets.
# Stratification preserves the class distribution.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Dataset shape:", X.shape)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Build a pipeline so scaling is performed separately
# inside each cross-validation training fold.
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(max_iter=1000),
        ),
    ]
)

# Use stratified 5-fold cross-validation.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
)

print("\nCross-validation accuracy scores:")
for fold, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold}: {score:.4f}")

print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
print(f"CV standard deviation: {cv_scores.std():.4f}")


# Train the final pipeline on the complete training set.
pipeline.fit(X_train, y_train)

# Evaluate only on the unseen test set.
y_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {test_accuracy:.4f}")


# Save CV results.
cv_results = pd.DataFrame(
    {
        "fold": range(1, len(cv_scores) + 1),
        "accuracy": cv_scores,
    }
)

cv_results.loc[len(cv_results)] = [
    "mean",
    cv_scores.mean(),
]

cv_results.to_csv(
    os.path.join(OUTPUT_DIR, "cross_validation_results.csv"),
    index=False,
)


# Plot cross-validation scores.
plt.figure(figsize=(8, 5))
plt.bar(
    [f"Fold {i}" for i in range(1, len(cv_scores) + 1)],
    cv_scores,
)
plt.axhline(
    cv_scores.mean(),
    linestyle="--",
    label="Mean CV Accuracy",
)
plt.ylim(0, 1.1)
plt.xlabel("Cross-Validation Fold")
plt.ylabel("Accuracy")
plt.title("5-Fold Cross-Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "cross_validation_accuracy.png")
)
plt.close()

print("\nCross-validation completed successfully.")
print("Output files saved in:", OUTPUT_DIR)