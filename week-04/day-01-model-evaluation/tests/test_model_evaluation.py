"""Unit tests for the Week 4 Day 1 evaluation experiment."""

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from model_evaluation import (  # noqa: E402
    build_dataset,
    build_model,
    calculate_metrics,
    run_cross_validation,
)


class TestModelEvaluation(unittest.TestCase):
    def test_metrics_on_known_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_score = np.array([0.1, 0.6, 0.8, 0.9])

        metrics = calculate_metrics(y_true, y_pred, y_score)

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1_score"], 0.8)
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertLessEqual(metrics["roc_auc"], 1.0)

    def test_dataset_is_reproducible_and_binary(self):
        X1, y1 = build_dataset()
        X2, y2 = build_dataset()

        self.assertEqual(X1.shape, (1500, 20))
        self.assertTrue(np.array_equal(X1, X2))
        self.assertTrue(np.array_equal(y1, y2))
        self.assertEqual(set(np.unique(y1)), {0, 1})

    def test_lower_threshold_does_not_reduce_recall(self):
        from sklearn.metrics import recall_score

        X, y = build_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        model = build_model()
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]

        predictions_at_05 = (probabilities >= 0.5).astype(int)
        predictions_at_03 = (probabilities >= 0.3).astype(int)

        recall_at_05 = recall_score(y_test, predictions_at_05)
        recall_at_03 = recall_score(y_test, predictions_at_03)

        self.assertGreaterEqual(recall_at_03, recall_at_05)

    def test_pipeline_and_stratified_cross_validation(self):
        X, y = build_dataset()
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )

        model = build_model()
        scores = run_cross_validation(model, X_train, y_train)

        for metric in ["precision", "recall", "f1", "roc_auc"]:
            self.assertEqual(len(scores[f"test_{metric}"]), 5)
            self.assertTrue(np.all(np.isfinite(scores[f"test_{metric}"])))
            self.assertTrue(np.all(scores[f"test_{metric}"] >= 0.0))
            self.assertTrue(np.all(scores[f"test_{metric}"] <= 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

