"""Automated tests for the FastAPI model serving endpoint."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from main import app  # noqa: E402

client = TestClient(app)
VALID_FEATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_root_returns_service_information():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_reports_loaded_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction_returns_finite_number():
    response = client.post("/predict", json={"features": VALID_FEATURES})
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body["prediction"], float)
    assert body["prediction"] == body["prediction"]
    assert abs(body["prediction"]) != float("inf")
    assert body["features_received"] == 10
    assert body["model"] == "w4d3-ridge-regression"


def test_wrong_feature_count_is_rejected():
    response = client.post("/predict", json={"features": [0.1] * 9})
    assert response.status_code == 422


def test_non_numeric_feature_is_rejected():
    features = VALID_FEATURES.copy()
    features[0] = "not-a-number"
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422


def test_api_docs_are_available():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]
