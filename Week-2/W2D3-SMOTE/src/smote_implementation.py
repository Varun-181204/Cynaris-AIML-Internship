import os

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outputs",
)


# Create an intentionally imbalanced binary classification dataset.
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    weights=[0.90, 0.10],
    random_state=42,
)

feature_names = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
]

data = pd.DataFrame(X, columns=feature_names)
data["target"] = y

# Split before applying SMOTE to prevent data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    data[feature_names],
    data["target"],
    test_size=0.2,
    random_state=42,
    stratify=data["target"],
)

# Scale features before applying SMOTE.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Class distribution before SMOTE:")
print(y_train.value_counts().sort_index())

# Apply SMOTE only to the training data.
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train,
)

print("\nClass distribution after SMOTE:")
print(y_train_smote.value_counts().sort_index())

# Save class distribution evidence.
distribution = pd.DataFrame(
    {
        "Before_SMOTE": y_train.value_counts().sort_index(),
        "After_SMOTE": y_train_smote.value_counts().sort_index(),
    }
)

distribution.to_csv(
    os.path.join(OUTPUT_DIR, "class_distribution.csv")
)

# Plot class distributions before and after SMOTE.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

y_train.value_counts().sort_index().plot(
    kind="bar",
    ax=axes[0],
)
axes[0].set_title("Before SMOTE")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")

y_train_smote.value_counts().sort_index().plot(
    kind="bar",
    ax=axes[1],
)
axes[1].set_title("After SMOTE")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "smote_class_distribution.png")
)
plt.close()

print("\nSMOTE processing completed successfully.")
print("Output files saved in:", OUTPUT_DIR)