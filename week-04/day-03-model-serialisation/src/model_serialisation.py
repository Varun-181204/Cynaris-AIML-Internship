"""Train, serialise, reload, and verify a scikit-learn regression model."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_model() -> Pipeline:
    """Create a reproducible preprocessing and regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0)),
    ])


def train_model() -> tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic data, split it, and fit the model."""
    features, target = make_regression(
        n_samples=600,
        n_features=10,
        n_informative=7,
        noise=15.0,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = create_model()
    model.fit(x_train, y_train)
    return model, x_train, x_test, y_train, y_test


def load_serialized_models(
    joblib_path: Path,
    pickle_path: Path,
) -> tuple[Any, Any]:
    """Load previously saved artifacts; only use trusted files."""
    try:
        if not joblib_path.is_file():
            raise FileNotFoundError(f"joblib artifact not found: {joblib_path}")
        if not pickle_path.is_file():
            raise FileNotFoundError(f"pickle artifact not found: {pickle_path}")

        reloaded_joblib = joblib.load(joblib_path)
        with pickle_path.open("rb") as file:
            reloaded_pickle = pickle.load(file)
        return reloaded_joblib, reloaded_pickle
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifacts: {exc}") from exc


def save_and_reload(
    model: Any,
    joblib_path: Path,
    pickle_path: Path,
) -> tuple[Any, Any]:
    """Save a model in both formats and reload the trusted artifacts."""
    try:
        joblib_path.parent.mkdir(parents=True, exist_ok=True)
        pickle_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, joblib_path)
        with pickle_path.open("wb") as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise RuntimeError(f"Failed to save model artifacts: {exc}") from exc

    return load_serialized_models(joblib_path, pickle_path)


def run_experiment(output_dir: str | Path) -> dict[str, Any]:
    """Train, save, reload, evaluate, and write evidence files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, _, x_test, _, y_test = train_model()
    joblib_path = output_path / "regression_model.joblib"
    pickle_path = output_path / "regression_model.pkl"

    loaded_joblib, loaded_pickle = save_and_reload(
        model, joblib_path, pickle_path
    )

    original_predictions = model.predict(x_test)
    joblib_predictions = loaded_joblib.predict(x_test)
    pickle_predictions = loaded_pickle.predict(x_test)

    joblib_match = bool(np.allclose(
        original_predictions, joblib_predictions, rtol=1e-10, atol=1e-10
    ))
    pickle_match = bool(np.allclose(
        original_predictions, pickle_predictions, rtol=1e-10, atol=1e-10
    ))

    if not joblib_match or not pickle_match:
        raise AssertionError(
            "Reloaded model predictions do not match the original model."
        )

    rmse = float(np.sqrt(mean_squared_error(y_test, original_predictions)))
    metrics = {
        "test_samples": int(len(y_test)),
        "mae": float(mean_absolute_error(y_test, original_predictions)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, original_predictions)),
    }

    with (output_path / "prediction_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "actual", "original_prediction",
            "joblib_prediction", "pickle_prediction",
        ])
        writer.writerows(zip(
            y_test, original_predictions, joblib_predictions, pickle_predictions
        ))

    summary = {
        "task": "Model serialisation with joblib and pickle",
        "model": "StandardScaler + Ridge(alpha=1.0)",
        "random_state": 42,
        "serialization": {
            "joblib": {
                "file": joblib_path.name,
                "size_bytes": joblib_path.stat().st_size,
                "predictions_match": joblib_match,
            },
            "pickle": {
                "file": pickle_path.name,
                "size_bytes": pickle_path.stat().st_size,
                "predictions_match": pickle_match,
            },
        },
        "metrics": metrics,
    }
    (output_path / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    task_dir = Path(__file__).resolve().parents[1]
    results = run_experiment(task_dir / "outputs")

    print("=== Model Serialisation Results ===")
    print(f"Test samples: {results['metrics']['test_samples']}")
    print(f"MAE:          {results['metrics']['mae']:.4f}")
    print(f"RMSE:         {results['metrics']['rmse']:.4f}")
    print(f"R²:           {results['metrics']['r2']:.4f}")
    for method, details in results["serialization"].items():
        print(
            f"{method}: predictions_match={details['predictions_match']}, "
            f"size={details['size_bytes']} bytes"
        )
    print(f"Evidence directory: {task_dir / 'outputs'}")
