"""Evaluate a binary classifier using metrics, stratified CV, and learning curves."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42


def build_dataset():
    """Create a reproducible, imbalanced binary classification dataset."""
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        weights=[0.70, 0.30],
        class_sep=1.0,
        random_state=RANDOM_STATE,
    )
    return X, y


def build_model():
    """Create a pipeline that fits scaling only on training folds."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])


def calculate_metrics(y_true, y_pred, y_score):
    """Calculate binary classification metrics for the positive class."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def run_cross_validation(model, X, y):
    """Evaluate the complete pipeline using stratified five-fold CV."""
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["precision", "recall", "f1", "roc_auc"],
        return_train_score=True,
        n_jobs=-1,
    )
    return scores


def save_learning_curve(model, X, y, output_dir):
    """Plot training and validation ROC-AUC as training size increases."""
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    sizes, train_scores, validation_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    plt.figure(figsize=(8, 5))
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    validation_mean = validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    plt.plot(sizes, train_mean, "o-", label="Training ROC-AUC")
    plt.fill_between(
        sizes, train_mean - train_std, train_mean + train_std, alpha=0.15
    )
    plt.plot(sizes, validation_mean, "o-", label="Validation ROC-AUC")
    plt.fill_between(
        sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15,
    )
    plt.xlabel("Number of training examples")
    plt.ylabel("ROC-AUC")
    plt.title("Learning Curve: Logistic Regression")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=150)
    plt.close()

    return sizes, train_scores, validation_scores


def run_experiment(output_dir=OUTPUT_DIR):
    """Train, evaluate, cross-validate, and save reproducible evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, y_pred, y_score)

    # Cross-validation uses only the training split; the test set stays untouched.
    cv_scores = run_cross_validation(model, X_train, y_train)

    with (output_dir / "evaluation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "test_score", "cv_mean", "cv_std"])
        scoring_names = {
            "precision": "precision",
            "recall": "recall",
            "f1_score": "f1",
            "roc_auc": "roc_auc",
        }
        for metric, test_score in test_metrics.items():
            if metric == "accuracy":
                writer.writerow([metric, f"{test_score:.4f}", "N/A", "N/A"])
            else:
                cv_name = scoring_names[metric]
                writer.writerow([
                    metric,
                    f"{test_score:.4f}",
                    f"{cv_scores['test_' + cv_name].mean():.4f}",
                    f"{cv_scores['test_' + cv_name].std():.4f}",
                ])

    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, y_pred, zero_division=0),
        encoding="utf-8",
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Negative", "Positive"], cmap="Blues"
    )
    plt.title("Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_score)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.title("Test Set ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()

    sizes, train_scores, validation_scores = save_learning_curve(
        model, X_train, y_train, output_dir
    )

    print("MODEL EVALUATION RESULTS")
    print("=" * 45)
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    for name, value in test_metrics.items():
        print(f"{name.replace('_', ' ').title():12}: {value:.4f}")

    print("\nSTRATIFIED 5-FOLD CROSS-VALIDATION (training split)")
    for metric in ["precision", "recall", "f1", "roc_auc"]:
        values = cv_scores[f"test_{metric}"]
        print(f"{metric:12}: {values.mean():.4f} +/- {values.std():.4f}")

    print("\nLEARNING CURVE (ROC-AUC)")
    for size, train_row, validation_row in zip(
        sizes, train_scores, validation_scores
    ):
        print(
            f"Training size {size:4}: "
            f"train={train_row.mean():.4f}, "
            f"validation={validation_row.mean():.4f}"
        )

    print(f"\nEvidence saved to: {output_dir.resolve()}")
    return test_metrics


if __name__ == "__main__":
    run_experiment()

