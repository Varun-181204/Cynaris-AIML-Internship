"""Unit tests for model serialisation."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))

from src.model_serialisation import (
    load_serialized_models,
    run_experiment,
    save_and_reload,
    train_model,
)


class ModelSerialisationTests(unittest.TestCase):
    def test_training_is_reproducible(self):
        model_a, _, x_test_a, _, y_test_a = train_model()
        model_b, _, x_test_b, _, y_test_b = train_model()

        np.testing.assert_array_equal(x_test_a, x_test_b)
        np.testing.assert_array_equal(y_test_a, y_test_b)
        np.testing.assert_allclose(
            model_a.predict(x_test_a), model_b.predict(x_test_b)
        )

    def test_joblib_and_pickle_preserve_predictions(self):
        model, _, x_test, _, _ = train_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            loaded_joblib, loaded_pickle = save_and_reload(
                model, root / "nested" / "model.joblib",
                root / "nested" / "model.pkl",
            )
            expected = model.predict(x_test)
            np.testing.assert_allclose(loaded_joblib.predict(x_test), expected)
            np.testing.assert_allclose(loaded_pickle.predict(x_test), expected)

    def test_missing_artifacts_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(
                RuntimeError, "Failed to load model artifacts"
            ):
                load_serialized_models(
                    root / "missing.joblib", root / "missing.pkl"
                )

    def test_experiment_writes_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary = run_experiment(output_dir)

            self.assertTrue(summary["serialization"]["joblib"]["predictions_match"])
            self.assertTrue(summary["serialization"]["pickle"]["predictions_match"])
            self.assertGreater(summary["serialization"]["joblib"]["size_bytes"], 0)
            self.assertGreater(summary["serialization"]["pickle"]["size_bytes"], 0)
            self.assertGreater(summary["metrics"]["test_samples"], 0)

            for filename in (
                "regression_model.joblib",
                "regression_model.pkl",
                "prediction_comparison.csv",
                "experiment_summary.json",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
