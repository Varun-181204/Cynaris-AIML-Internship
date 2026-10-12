import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


OUTPUT_DIR = "week-03/day-04-svm-knn/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_model(model_name, model, X_test, y_test):
    """Evaluate a trained model and return its classification metrics."""
    predictions = model.predict(X_test)

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1-Score": f1_score(y_test, predictions),
    }


def save_confusion_matrix(model_name, model, X_test, y_test, filename):
    """Save a confusion matrix visualization."""
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)

    print(f"\n{model_name} Confusion Matrix:")
    print(matrix)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


# Load the Breast Cancer classification dataset.
data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Classes:", data.target_names)

# Create a stratified 80/20 train-test split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# SVM pipeline with feature scaling.
svm_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ]
)

# KNN pipeline with feature scaling.
knn_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)

# Train the baseline SVM and KNN models.
svm_pipeline.fit(X_train, y_train)
knn_pipeline.fit(X_train, y_train)

# Evaluate baseline models.
svm_result = evaluate_model(
    "SVM",
    svm_pipeline,
    X_test,
    y_test,
)

knn_result = evaluate_model(
    "KNN",
    knn_pipeline,
    X_test,
    y_test,
)

print("\nBaseline Model Comparison")
print(pd.DataFrame([svm_result, knn_result]).to_string(index=False))

# Tune SVM hyperparameters.
svm_param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", "auto"],
    "svm__kernel": ["rbf", "linear"],
}

svm_grid = GridSearchCV(
    svm_pipeline,
    svm_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

svm_grid.fit(X_train, y_train)

print("\nBest SVM Parameters:")
print(svm_grid.best_params_)

# Tune KNN hyperparameters.
knn_param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}

knn_grid = GridSearchCV(
    knn_pipeline,
    knn_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

knn_grid.fit(X_train, y_train)

print("\nBest KNN Parameters:")
print(knn_grid.best_params_)

# Evaluate tuned models.
tuned_svm_result = evaluate_model(
    "Tuned SVM",
    svm_grid.best_estimator_,
    X_test,
    y_test,
)

tuned_knn_result = evaluate_model(
    "Tuned KNN",
    knn_grid.best_estimator_,
    X_test,
    y_test,
)

results = pd.DataFrame(
    [
        svm_result,
        knn_result,
        tuned_svm_result,
        tuned_knn_result,
    ]
)

print("\nFinal Model Comparison")
print(results.to_string(index=False))

# Save comparison results.
results.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False,
)

# Save confusion matrices for tuned models.
save_confusion_matrix(
    "Tuned SVM",
    svm_grid.best_estimator_,
    X_test,
    y_test,
    "svm_confusion_matrix.png",
)

save_confusion_matrix(
    "Tuned KNN",
    knn_grid.best_estimator_,
    X_test,
    y_test,
    "knn_confusion_matrix.png",
)

# Print detailed classification reports.
print("\nTuned SVM Classification Report:")
print(
    classification_report(
        y_test,
        svm_grid.best_estimator_.predict(X_test),
        target_names=data.target_names,
    )
)

print("\nTuned KNN Classification Report:")
print(
    classification_report(
        y_test,
        knn_grid.best_estimator_.predict(X_test),
        target_names=data.target_names,
    )
)

print("\nOutput files saved successfully.")