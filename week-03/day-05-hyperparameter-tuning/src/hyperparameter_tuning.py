import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


OUTPUT_DIR = "week-03/day-05-hyperparameter-tuning/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


# Create an SVM pipeline with feature scaling.
svm_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svm", SVC()),
    ]
)


# Create a KNN pipeline with feature scaling.
knn_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ]
)


# Define the SVM parameter search space.
svm_param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", "auto"],
    "svm__kernel": ["rbf", "linear"],
}


# Define the KNN parameter search space.
knn_param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}


# Run GridSearchCV for SVM.
svm_grid = GridSearchCV(
    svm_pipeline,
    svm_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

svm_grid.fit(X_train, y_train)


# Run RandomizedSearchCV for KNN.
knn_random = RandomizedSearchCV(
    knn_pipeline,
    knn_param_grid,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

knn_random.fit(X_train, y_train)


# Print the best hyperparameters.
print("\nBest SVM Parameters:")
print(svm_grid.best_params_)

print("\nBest SVM Cross-Validation Accuracy:")
print(svm_grid.best_score_)

print("\nBest KNN Parameters:")
print(knn_random.best_params_)

print("\nBest KNN Cross-Validation Accuracy:")
print(knn_random.best_score_)


# Evaluate the best models on the held-out test set.
svm_test_accuracy = svm_grid.best_estimator_.score(X_test, y_test)
knn_test_accuracy = knn_random.best_estimator_.score(X_test, y_test)

print("\nTest Set Accuracy:")
print(f"Tuned SVM: {svm_test_accuracy:.4f}")
print(f"Tuned KNN: {knn_test_accuracy:.4f}")


# Save model comparison results.
results = pd.DataFrame(
    [
        {
            "Model": "GridSearchCV SVM",
            "Search_Method": "GridSearchCV",
            "Best_Parameters": str(svm_grid.best_params_),
            "Best_CV_Accuracy": svm_grid.best_score_,
            "Test_Accuracy": svm_test_accuracy,
        },
        {
            "Model": "RandomizedSearchCV KNN",
            "Search_Method": "RandomizedSearchCV",
            "Best_Parameters": str(knn_random.best_params_),
            "Best_CV_Accuracy": knn_random.best_score_,
            "Test_Accuracy": knn_test_accuracy,
        },
    ]
)

results.to_csv(
    os.path.join(OUTPUT_DIR, "tuning_comparison.csv"),
    index=False,
)

svm_results = pd.DataFrame(svm_grid.cv_results_)
svm_results.to_csv(
    os.path.join(OUTPUT_DIR, "svm_grid_search_results.csv"),
    index=False,
)

knn_results = pd.DataFrame(knn_random.cv_results_)
knn_results.to_csv(
    os.path.join(OUTPUT_DIR, "knn_random_search_results.csv"),
    index=False,
)

plot_data = results.set_index("Model")[
    ["Best_CV_Accuracy", "Test_Accuracy"]
]

ax = plot_data.plot(
    kind="bar",
    figsize=(8, 5),
)

ax.set_ylabel("Accuracy")
ax.set_title("GridSearchCV vs RandomizedSearchCV")
ax.set_ylim(0.9, 1.0)
ax.legend(["Best CV Accuracy", "Test Accuracy"])

plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "tuning_comparison.png")
)
plt.close()

print("\nOutput files saved successfully.")


# Save search results for further analysis.
svm_results = pd.DataFrame(svm_grid.cv_results_)
svm_results.to_csv(
    os.path.join(OUTPUT_DIR, "svm_grid_search_results.csv"),
    index=False,
)

knn_results = pd.DataFrame(knn_random.cv_results_)
knn_results.to_csv(
    os.path.join(OUTPUT_DIR, "knn_random_search_results.csv"),
    index=False,
)


# Create a simple comparison chart.
plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["Test_Accuracy"],
)

plt.ylabel("Test Accuracy")
plt.title("GridSearchCV vs RandomizedSearchCV")
plt.ylim(0.9, 1.0)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "tuning_comparison.png")
)

plt.close()


print("\nOutput files saved successfully.")