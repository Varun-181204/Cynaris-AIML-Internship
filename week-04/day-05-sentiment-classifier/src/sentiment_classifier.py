
"""Train and evaluate Logistic Regression and Random Forest sentiment classifiers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42


def build_dataset() -> tuple[list[str], np.ndarray]:
    """Build a deterministic synthetic binary sentiment dataset."""
    positive = [
        "I love this product", "excellent quality and great value",
        "absolutely wonderful experience", "very happy with this purchase",
        "the service was friendly and helpful", "fantastic results and easy to use",
        "highly recommend this excellent item", "beautiful design and reliable performance",
        "great experience would buy again", "the best product I have used",
        "superb quality and fast delivery", "this made my day better",
        "really impressive and enjoyable", "works perfectly and feels premium",
        "delightful experience from start to finish", "very satisfied with the result",
        "helpful support solved everything", "brilliant product and amazing service",
        "easy setup and excellent performance", "a wonderful and worthwhile purchase",
    ]
    negative = [
        "I hate this product", "terrible quality and poor value",
        "absolutely awful experience", "very unhappy with this purchase",
        "the service was rude and unhelpful", "horrible results and hard to use",
        "do not recommend this disappointing item", "ugly design and unreliable performance",
        "bad experience would not buy again", "the worst product I have used",
        "awful quality and slow delivery", "this made my day worse",
        "really frustrating and unpleasant", "works poorly and feels cheap",
        "disappointing experience from start to finish", "very dissatisfied with the result",
        "unhelpful support solved nothing", "broken product and terrible service",
        "difficult setup and poor performance", "a wasteful and regrettable purchase",
    ]
    templates = [
        "Review: {}.",
        "Customer feedback says {}.",
        "Overall, {}.",
        "My experience: {}.",
        "I can say that {}.",
    ]

    texts, labels = [], []
    for phrase in positive:
        for template in templates:
            texts.append(template.format(phrase))
            labels.append(1)
    for phrase in negative:
        for template in templates:
            texts.append(template.format(phrase))
            labels.append(0)

    return texts, np.asarray(labels, dtype=int)


def build_models() -> dict[str, Pipeline]:
    """Return two models using the same TF-IDF configuration."""
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("classifier", LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE
            )),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),
    }


def evaluate_models(output_dir: Path = OUTPUT_DIR) -> dict:
    """Train models, evaluate them, and save evidence files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    texts, labels = build_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    models = build_models()
    results: dict[str, dict[str, float]] = {}
    reports = []

    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(12, 5))
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    for (name, model), ax_cm in zip(models.items(), axes_cm):
        # TF-IDF is fitted only on training data through the pipeline.
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(
                y_test, predictions, zero_division=0
            )),
            "recall": float(recall_score(
                y_test, predictions, zero_division=0
            )),
            "f1": float(f1_score(
                y_test, predictions, zero_division=0
            )),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        results[name] = metrics

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["negative", "positive"],
            zero_division=0,
        )
        reports.append(f"=== {name} ===\n{report}")

        print(f"\n=== {name} ===")
        print(report)
        print("Metrics:", json.dumps(metrics, indent=2))

        ConfusionMatrixDisplay(
            confusion_matrix(y_test, predictions, labels=[0, 1]),
            display_labels=["Negative", "Positive"],
        ).plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title(name)

        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            name=f"{name} (AUC={metrics['roc_auc']:.3f})",
            ax=ax_roc,
        )

    fig_cm.tight_layout()
    fig_cm.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig_cm)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax_roc.set_title("Sentiment Classifier ROC Curves")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(output_dir / "roc_auc_curves.png", dpi=150)
    plt.close(fig_roc)

    (output_dir / "classification_report.txt").write_text(
        "\n\n".join(reports) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    # Keep CSV columns and values aligned with metrics.json.
    columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    with (output_dir / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow({"model": name, **metrics})

    return results


def predict_sentiment(text: str, model: Pipeline) -> str:
    """Predict sentiment for non-empty text using a fitted pipeline."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return "positive" if int(model.predict([text])[0]) == 1 else "negative"


if __name__ == "__main__":
    evaluate_models()
