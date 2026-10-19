
"""Artifact tests for the W4D6 Responsible AI audit."""

import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
NOTEBOOKS = ROOT / "notebooks"
REPORTS = ROOT / "reports"


class TestResponsibleAIDeliverables(unittest.TestCase):
    def test_fairness_artifacts_exist(self):
        for name in (
            "fairness_metrics.csv",
            "fairness_metrics.json",
            "fairness_comparison.png",
        ):
            with self.subTest(file=name):
                self.assertTrue((OUTPUTS / name).is_file())

    def test_fairness_metrics_have_expected_models_and_columns(self):
        metrics = pd.read_csv(OUTPUTS / "fairness_metrics.csv")

        self.assertEqual(
            metrics["model"].tolist(),
            ["Baseline", "Reweighing"],
        )

        expected_columns = {
            "model",
            "accuracy",
            "demographic_parity_disparate_impact",
            "demographic_parity_difference",
            "average_odds_difference",
            "equal_opportunity_difference",
        }
        self.assertTrue(expected_columns.issubset(metrics.columns))
        self.assertTrue(metrics["accuracy"].between(0, 1).all())

    def test_fairness_json_is_valid(self):
        data = json.loads(
            (OUTPUTS / "fairness_metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(data), 2)

    def test_shap_artifacts_exist_and_are_valid(self):
        self.assertTrue((OUTPUTS / "shap_summary.png").is_file())

        data = json.loads(
            (OUTPUTS / "shap_summary_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(data["features"], 20)
        self.assertEqual(data["train_samples"], 1200)
        self.assertEqual(data["test_samples"], 300)

    def test_documentation_and_notebook_exist(self):
        required_files = [
            DOCS / "model_card.md",
            DOCS / "india_bias_case.md",
            NOTEBOOKS / "responsible_ai_audit.ipynb",
            REPORTS / "model_card.pdf",
        ]
        for path in required_files:
            with self.subTest(file=path.name):
                self.assertTrue(path.is_file(), f"Missing: {path}")


if __name__ == "__main__":
    unittest.main()

