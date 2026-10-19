
"""W4D6 Responsible AI audit: AIF360 fairness, Reweighing, and SHAP."""

from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
UNPRIVILEGED = [{"sex": 0.0}]
PRIVILEGED = [{"sex": 1.0}]


def build_adult_dataset():
    """Load the UCI Adult Income dataset using AIF360."""
    return AdultDataset(
        protected_attribute_names=["sex"],
        privileged_classes=[["Male"]],
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "native-country",
        ],
        features_to_keep=[
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "sex",
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "native-country",
        ],
        na_values=["?", " ?"],
        custom_preprocessing=None,
        metadata={"label_maps": [{1.0: ">50K", 0.0: "<=50K"}]},
    )


def build_scaled_classifier():
    """Create a scaled logistic-regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])


def audit_adult_income():
    """Compare baseline fairness metrics with AIF360 Reweighing."""
    print("Loading the UCI Adult Income dataset through AIF360...")
    adult = build_adult_dataset()

    indices = np.arange(adult.features.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=adult.labels.ravel(),
    )
    train = adult.subset(train_idx)
    test = adult.subset(test_idx)

    # Baseline model: unmodified training instance weights.
    baseline = build_scaled_classifier()
    baseline.fit(train.features, train.labels.ravel())
    baseline_pred = baseline.predict(test.features)

    # Reweighing changes training instance weights, not the test set.
    reweigh = Reweighing(
        unprivileged_groups=UNPRIVILEGED,
        privileged_groups=PRIVILEGED,
    )
    train_reweighted = reweigh.fit_transform(train)

    mitigated = build_scaled_classifier()
    mitigated.fit(
        train_reweighted.features,
        train_reweighted.labels.ravel(),
        classifier__sample_weight=train_reweighted.instance_weights,
    )
    mitigated_pred = mitigated.predict(test.features)

    def calculate_group_metrics(predictions, model_name):
        predicted_dataset = test.copy(deepcopy=True)
        predicted_dataset.labels = predictions.reshape(-1, 1).astype(float)

        rate_metric = BinaryLabelDatasetMetric(
            predicted_dataset,
            unprivileged_groups=UNPRIVILEGED,
            privileged_groups=PRIVILEGED,
        )
        classified_metric = ClassificationMetric(
            test,
            predicted_dataset,
            unprivileged_groups=UNPRIVILEGED,
            privileged_groups=PRIVILEGED,
        )

        return {
            "model": model_name,
            "accuracy": float(
                accuracy_score(test.labels.ravel(), predictions)
            ),
            "demographic_parity_disparate_impact": float(
                rate_metric.disparate_impact()
            ),
            "demographic_parity_difference": float(
                classified_metric.statistical_parity_difference()
            ),
            "average_odds_difference": float(
                classified_metric.average_odds_difference()
            ),
            "equal_opportunity_difference": float(
                classified_metric.equal_opportunity_difference()
            ),
        }

    results = pd.DataFrame([
        calculate_group_metrics(baseline_pred, "Baseline"),
        calculate_group_metrics(mitigated_pred, "Reweighing"),
    ])

    results.to_csv(OUTPUTS / "fairness_metrics.csv", index=False)
    (OUTPUTS / "fairness_metrics.json").write_text(
        results.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    # Plot 1: disparate impact. The 0.80 line is a screening reference.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    results.set_index("model")[
        ["demographic_parity_disparate_impact"]
    ].plot(kind="bar", ax=axes[0], legend=False, ylim=(0, 1.5))

    axes[0].axhline(
        0.8,
        linestyle="--",
        label="0.80 screening reference",
    )
    axes[0].set_title("Disparate impact")
    axes[0].set_ylabel("Unprivileged / privileged positive rate")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend()

    # Plot 2: signed differences. Values nearer zero indicate smaller gaps.
    results.set_index("model")[
        [
            "demographic_parity_difference",
            "average_odds_difference",
        ]
    ].plot(kind="bar", ax=axes[1])

    axes[1].axhline(0, linestyle="--")
    axes[1].set_title("Fairness differences")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUTS / "fairness_comparison.png", dpi=160)
    plt.close(fig)

    print("\nFairness comparison:")
    print(results.to_string(index=False))
    print("\nSaved fairness metrics and comparison plot.")
    print(
        "The 0.80 reference is a screening heuristic, "
        "not a legal finding or proof of fairness."
    )

    return results


def explain_w4d1_model():
    """Generate SHAP explanations for the W4D1 synthetic model."""
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        weights=[0.70, 0.30],
        class_sep=1.0,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])
    model.fit(X_train, y_train)

    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = [
        f"feature_{i:02d}" for i in range(X.shape[1])
    ]
    background = X_train_scaled[:100]
    sample = X_test_scaled[:200]

    explainer = shap.LinearExplainer(classifier, background)
    shap_values = explainer(sample)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values.values,
        sample,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUTS / "shap_summary.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    result = {
        "model": "StandardScaler + LogisticRegression",
        "dataset": (
            "Synthetic make_classification dataset used by W4D1"
        ),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy_on_this_split": float(
            accuracy_score(y_test, model.predict(X_test))
        ),
        "features": int(X.shape[1]),
        "shap_background_samples": int(len(background)),
        "shap_explained_samples": int(len(sample)),
        "interpretation": (
            "SHAP values explain the fitted model on synthetic data. "
            "They do not establish causation or real-world validity."
        ),
    }

    (OUTPUTS / "shap_summary_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nW4D1 SHAP demonstration:")
    print(json.dumps(result, indent=2))
    print("Saved SHAP summary plot and metrics.")

    return result


def main():
    audit_adult_income()
    explain_w4d1_model()
    print("\nW4D6 audit script completed.")


if __name__ == "__main__":
    main()

