import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


OUTPUT_DIR = "week-03/day-03-decision-trees/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load the Breast Cancer dataset
data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Classes: {list(data.target_names)}")


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Train a baseline Decision Tree
decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(X_train, y_train)

dt_predictions = decision_tree.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Tree Results")
print(f"Accuracy: {dt_accuracy:.4f}")
print(f"Tree depth: {decision_tree.get_depth()}")
print(f"Number of leaves: {decision_tree.get_n_leaves()}")


# Train a tuned Decision Tree to reduce overfitting
tuned_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)

tuned_tree.fit(X_train, y_train)

tuned_predictions = tuned_tree.predict(X_test)

tuned_accuracy = accuracy_score(y_test, tuned_predictions)

print("\nTuned Decision Tree Results")
print(f"Accuracy: {tuned_accuracy:.4f}")
print(f"Tree depth: {tuned_tree.get_depth()}")
print(f"Number of leaves: {tuned_tree.get_n_leaves()}")


# Train Random Forest
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Results")
print(f"Accuracy: {rf_accuracy:.4f}")


# Compare the models
comparison = pd.DataFrame(
    {
        "Model": [
            "Decision Tree",
            "Tuned Decision Tree",
            "Random Forest",
        ],
        "Accuracy": [
            dt_accuracy,
            tuned_accuracy,
            rf_accuracy,
        ],
    }
)

print("\nModel Comparison")
print(comparison.to_string(index=False))

comparison.to_csv(
    f"{OUTPUT_DIR}/model_comparison.csv",
    index=False,
)


# Save Decision Tree visualization
plt.figure(figsize=(20, 12))

plot_tree(
    tuned_tree,
    feature_names=X.columns,
    class_names=data.target_names,
    filled=True,
    max_depth=3,
    fontsize=7,
)

plt.title("Tuned Decision Tree")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/decision_tree.png",
    dpi=150,
)
plt.close()


# Save confusion matrix for Random Forest
cm = confusion_matrix(y_test, rf_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names,
)

disp.plot()

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/random_forest_confusion_matrix.png",
    dpi=150,
)
plt.close()


# Save Random Forest feature importance
feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": random_forest.feature_importances_,
    }
).sort_values(
    by="Importance",
    ascending=False,
)

feature_importance.to_csv(
    f"{OUTPUT_DIR}/feature_importance.csv",
    index=False,
)

print("\nTop 10 Random Forest Features")
print(feature_importance.head(10).to_string(index=False))

print("\nOutput files saved successfully.")