"""Unit tests for the W4D2 bias-variance experiment."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from bias_variance import make_dataset, rmse, run_experiment


class TestBiasVariance(unittest.TestCase):
    def test_dataset_is_reproducible(self):
        X1, y1 = make_dataset()
        X2, y2 = make_dataset()
        self.assertTrue(np.array_equal(X1, X2))
        self.assertTrue(np.array_equal(y1, y2))
        self.assertEqual(X1.shape, (600, 15))

    def test_rmse_for_known_predictions(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 3.0, 3.0])
        self.assertAlmostEqual(rmse(actual, predicted), np.sqrt(1.0 / 3.0))

    def test_rmse_is_zero_for_perfect_predictions(self):
        values = np.array([2.0, 4.0, 6.0])
        self.assertAlmostEqual(rmse(values, values), 0.0)

    def test_experiment_returns_valid_results_and_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_experiment(temp_dir)
            output_dir = Path(temp_dir)

            self.assertEqual(summary["training_samples"], 480)
            self.assertEqual(summary["test_samples"], 120)
            self.assertEqual(summary["cross_validation_folds"], 5)
            self.assertEqual(len(summary["search_results"]), 3)
            self.assertEqual(len(summary["bias_variance_results"]), 6)

            for result in summary["search_results"]:
                self.assertGreater(result["best_alpha"], 0)
                self.assertTrue(np.isfinite(result["cv_rmse"]))
                self.assertTrue(np.isfinite(result["test_rmse"]))
                self.assertTrue(np.isfinite(result["test_r2"]))

            expected_files = [
                "search_comparison.csv",
                "bias_variance_analysis.csv",
                "bias_variance_curve.png",
                "search_comparison.png",
                "experiment_summary.json",
            ]
            for filename in expected_files:
                self.assertTrue((output_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
