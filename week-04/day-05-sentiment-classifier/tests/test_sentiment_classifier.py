"""Tests for the W4D5 sentiment classifier."""

import sys
import tempfile
import unittest
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from sentiment_classifier import (
    build_dataset,
    build_models,
    evaluate_models,
    predict_sentiment,
)


class SentimentClassifierTests(unittest.TestCase):
    def test_dataset_is_binary_and_balanced(self):
        texts, labels = build_dataset()
        self.assertEqual(len(texts), len(labels))
        self.assertEqual(set(labels.tolist()), {0, 1})
        self.assertEqual(int((labels == 0).sum()), int((labels == 1).sum()))

    def test_train_test_split_keeps_both_classes(self):
        texts, labels = build_dataset()
        _, _, y_train, y_test = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=labels
        )
        self.assertEqual(set(y_train.tolist()), {0, 1})
        self.assertEqual(set(y_test.tolist()), {0, 1})

    def test_both_models_can_fit_and_predict(self):
        texts, labels = build_dataset()
        x_train, x_test, y_train, _ = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=labels
        )
        for model in build_models().values():
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            probabilities = model.predict_proba(x_test)
            self.assertEqual(len(predictions), len(x_test))
            self.assertEqual(probabilities.shape, (len(x_test), 2))

    def test_prediction_rejects_empty_input(self):
        model = next(iter(build_models().values()))
        with self.assertRaises(ValueError):
            predict_sentiment("  ", model)

    def test_evaluation_generates_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = evaluate_models(Path(tmp))
            self.assertEqual(set(results), {"Logistic Regression", "Random Forest"})
            for filename in (
                "classification_report.txt",
                "metrics.json",
                "model_comparison.csv",
                "confusion_matrices.png",
                "roc_auc_curves.png",
            ):
                self.assertTrue((Path(tmp) / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()



