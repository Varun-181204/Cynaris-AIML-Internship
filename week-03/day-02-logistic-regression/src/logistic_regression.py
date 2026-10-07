import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = "week-03/day-02-logistic-regression/outputs"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# PART 1: BINARY CLASSIFICATION
# BREAST CANCER DATASET
# ============================================================

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names,
)

y = pd.Series(
    data.target,
    name="target",
)

print("Dataset shape:", X.shape)

print("\nClasses:", data.target_names)

print("\nClass distribution:")
print(y.value_counts())


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# BINARY LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

model.fit(
    X_train_scaled,
    y_train,
)


# ============================================================
# COEFFICIENTS AND INTERCEPT
# ============================================================

print("\nLogistic Regression Coefficients:")

coefficients = pd.DataFrame(
    {
        "Feature": X.columns,
        "Coefficient": model.coef_[0],
    }
)

print(coefficients)

print("\nIntercept:", model.intercept_[0])


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

precision = precision_score(
    y_test,
    y_pred,
)

recall = recall_score(
    y_test,
    y_pred,
)

f1 = f1_score(
    y_test,
    y_pred,
)

roc_auc = roc_auc_score(
    y_test,
    y_probability,
)


print("\nClassification Metrics:")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
)

print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names,
    )
)


# ============================================================
# SAVE BINARY METRICS
# ============================================================

metrics = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
        ],
    }
)

metrics.to_csv(
    f"{output_dir}/classification_metrics.csv",
    index=False,
)


# ============================================================
# CONFUSION MATRIX HEATMAP
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)

im = ax.imshow(cm)

ax.set_title(
    "Logistic Regression - Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_xticks([0, 1])

ax.set_yticks([0, 1])

ax.set_xticklabels(
    data.target_names
)

ax.set_yticklabels(
    data.target_names
)


for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )


fig.colorbar(
    im,
    ax=ax,
)

plt.tight_layout()

plt.savefig(
    f"{output_dir}/confusion_matrix.png",
    dpi=300,
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability,
)


plt.figure(
    figsize=(7, 5)
)

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Logistic Regression - ROC Curve"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{output_dir}/roc_curve.png",
    dpi=300,
)

plt.close()


# ============================================================
# PART 2: MULTI-CLASS CLASSIFICATION
# IRIS DATASET
# ============================================================

print("\n" + "=" * 60)

print("MULTI-CLASS CLASSIFICATION: IRIS")

print("=" * 60)


iris = load_iris()


X_iris = pd.DataFrame(
    iris.data,
    columns=iris.feature_names,
)

y_iris = pd.Series(
    iris.target,
    name="target",
)


print("\nIris dataset shape:", X_iris.shape)

print(
    "\nIris classes:",
    iris.target_names,
)


# Train-test split
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris,
    y_iris,
    test_size=0.2,
    random_state=42,
    stratify=y_iris,
)


# Feature scaling
iris_scaler = StandardScaler()

X_iris_train_scaled = iris_scaler.fit_transform(
    X_iris_train
)

X_iris_test_scaled = iris_scaler.transform(
    X_iris_test
)


# ============================================================
# ONE-VS-REST CLASSIFICATION
# ============================================================

ovr_base_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

ovr_model = OneVsRestClassifier(
    ovr_base_model
)

ovr_model.fit(
    X_iris_train_scaled,
    y_iris_train,
)

ovr_pred = ovr_model.predict(
    X_iris_test_scaled
)


# ============================================================
# SOFTMAX / MULTINOMIAL CLASSIFICATION
# ============================================================

softmax_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

softmax_model.fit(
    X_iris_train_scaled,
    y_iris_train,
)

softmax_pred = softmax_model.predict(
    X_iris_test_scaled
)


# ============================================================
# ONE-VS-REST METRICS
# ============================================================

ovr_accuracy = accuracy_score(
    y_iris_test,
    ovr_pred,
)

ovr_precision = precision_score(
    y_iris_test,
    ovr_pred,
    average="weighted",
)

ovr_recall = recall_score(
    y_iris_test,
    ovr_pred,
    average="weighted",
)

ovr_f1 = f1_score(
    y_iris_test,
    ovr_pred,
    average="weighted",
)


# ============================================================
# SOFTMAX METRICS
# ============================================================

softmax_accuracy = accuracy_score(
    y_iris_test,
    softmax_pred,
)

softmax_precision = precision_score(
    y_iris_test,
    softmax_pred,
    average="weighted",
)

softmax_recall = recall_score(
    y_iris_test,
    softmax_pred,
    average="weighted",
)

softmax_f1 = f1_score(
    y_iris_test,
    softmax_pred,
    average="weighted",
)


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    {
        "Model": [
            "One-vs-Rest",
            "Softmax",
        ],
        "Accuracy": [
            ovr_accuracy,
            softmax_accuracy,
        ],
        "Precision": [
            ovr_precision,
            softmax_precision,
        ],
        "Recall": [
            ovr_recall,
            softmax_recall,
        ],
        "F1-Score": [
            ovr_f1,
            softmax_f1,
        ],
    }
)


print("\nMulti-Class Model Comparison:")

print(
    comparison.to_string(
        index=False
    )
)


# Save comparison table
comparison.to_csv(
    f"{output_dir}/multiclass_comparison.csv",
    index=False,
)


# ============================================================
# MULTI-CLASS CLASSIFICATION REPORTS
# ============================================================

print("\nOne-vs-Rest Classification Report:")

print(
    classification_report(
        y_iris_test,
        ovr_pred,
        target_names=iris.target_names,
    )
)


print("\nSoftmax Classification Report:")

print(
    classification_report(
        y_iris_test,
        softmax_pred,
        target_names=iris.target_names,
    )
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nOutputs saved successfully:")

print(
    "- outputs/classification_metrics.csv"
)

print(
    "- outputs/confusion_matrix.png"
)

print(
    "- outputs/roc_curve.png"
)

print(
    "- outputs/multiclass_comparison.csv"
)

print("\nAll W3D2 outputs generated successfully.")