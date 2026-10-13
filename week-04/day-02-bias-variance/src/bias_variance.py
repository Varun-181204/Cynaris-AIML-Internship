"""W4D2: Bias-variance tradeoff, regularisation, and hyperparameter search."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42


def make_dataset():
    """Create a reproducible regression dataset."""
    X, y = make_regression(
        n_samples=600,
        n_features=15,
        n_informative=8,
        noise=18.0,
        random_state=RANDOM_STATE,
    )
    return X, y


def build_pipeline(estimator):
    """Scale features before fitting a regression estimator."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def rmse(y_true, y_pred):
    """Calculate root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_experiment(output_dir=None):
    """Run regularisation searches and a decision-tree bias-variance study."""
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    results = []
    search_details = {}

    # Grid Search evaluates every alpha in the specified list.
    ridge_grid = GridSearchCV(
        build_pipeline(Ridge()),
        param_grid={"model__alpha": alphas},
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    ridge_grid.fit(X_train, y_train)
    ridge_pred = ridge_grid.predict(X_test)
    results.append({
        "method": "Ridge Grid Search",
        "best_alpha": float(ridge_grid.best_params_["model__alpha"]),
        "cv_rmse": float(-ridge_grid.best_score_),
        "test_rmse": rmse(y_test, ridge_pred),
        "test_r2": float(r2_score(y_test, ridge_pred)),
    })
    search_details["ridge_grid"] = ridge_grid

    # Randomised Search samples a fixed number of alpha candidates.
    ridge_random = RandomizedSearchCV(
        build_pipeline(Ridge()),
        param_distributions={"model__alpha": np.logspace(-3, 3, 100)},
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )
    ridge_random.fit(X_train, y_train)
    random_pred = ridge_random.predict(X_test)
    results.append({
        "method": "Ridge Randomised Search",
        "best_alpha": float(ridge_random.best_params_["model__alpha"]),
        "cv_rmse": float(-ridge_random.best_score_),
        "test_rmse": rmse(y_test, random_pred),
        "test_r2": float(r2_score(y_test, random_pred)),
    })
    search_details["ridge_random"] = ridge_random

    # Lasso encourages sparse coefficients; scaling helps regularisation.
    lasso_grid = GridSearchCV(
        build_pipeline(Lasso(max_iter=20000)),
        param_grid={"model__alpha": alphas},
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    lasso_grid.fit(X_train, y_train)
    lasso_pred = lasso_grid.predict(X_test)
    results.append({
        "method": "Lasso Grid Search",
        "best_alpha": float(lasso_grid.best_params_["model__alpha"]),
        "cv_rmse": float(-lasso_grid.best_score_),
        "test_rmse": rmse(y_test, lasso_pred),
        "test_r2": float(r2_score(y_test, lasso_pred)),
    })
    search_details["lasso_grid"] = lasso_grid

    results_df = pd.DataFrame(results).sort_values("cv_rmse")
    results_df.to_csv(output_dir / "search_comparison.csv", index=False)

    # Compare training and cross-validation error across tree depths.
    # A large train/CV gap suggests high variance; both errors high suggest bias.
    depths = [1, 2, 3, 5, 8, None]
    bias_rows = []

    for depth in depths:
        estimator = DecisionTreeRegressor(
            max_depth=depth, random_state=RANDOM_STATE
        )
        scores = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            return_train_score=True,
            n_jobs=-1,
        )
        bias_rows.append({
            "max_depth": "None" if depth is None else str(depth),
            "train_rmse": float(-np.mean(scores["train_score"])),
            "cv_rmse": float(-np.mean(scores["test_score"])),
            "cv_rmse_std": float(np.std(-scores["test_score"])),
        })

    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(output_dir / "bias_variance_analysis.csv", index=False)

    # Plot train and CV errors to make the bias-variance pattern visible.
    x_positions = np.arange(len(bias_df))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_positions, bias_df["train_rmse"], marker="o", label="Training RMSE")
    ax.errorbar(
        x_positions,
        bias_df["cv_rmse"],
        yerr=bias_df["cv_rmse_std"],
        marker="o",
        capsize=4,
        label="Cross-validation RMSE",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bias_df["max_depth"])
    ax.set_xlabel("Decision tree max_depth (None = unrestricted)")
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Bias-Variance Analysis")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bias_variance_curve.png", dpi=150)
    plt.close(fig)

    # Compare the three search strategies/models using CV RMSE.
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = results_df.sort_values("cv_rmse", ascending=True)
    ax.bar(ordered["method"], ordered["cv_rmse"])
    ax.set_ylabel("Best cross-validation RMSE (lower is better)")
    ax.set_title("Regularisation and Hyperparameter Search Comparison")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "search_comparison.png", dpi=150)
    plt.close(fig)

    summary = {
        "random_state": RANDOM_STATE,
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "cross_validation_folds": 5,
        "best_method_by_cv_rmse": str(results_df.iloc[0]["method"]),
        "search_results": results_df.to_dict(orient="records"),
        "bias_variance_results": bias_rows,
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== Regularisation and Search Results ===")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== Bias-Variance Results ===")
    print(bias_df.to_string(index=False))
    print(f"\nEvidence saved to: {output_dir.resolve()}")

    return summary


if __name__ == "__main__":
    run_experiment()
