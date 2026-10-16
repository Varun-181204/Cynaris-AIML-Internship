"""API tests covering normal requests, validation, and model failure."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import main  # noqa: E402

client = TestClient(main.app)
VALID_FEATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_root_returns_service_information():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_reports_loaded_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction_returns_finite_number():
    response = client.post("/predict", json={"features": VALID_FEATURES})
    assert response.status_code == 200
    body = response.json()
    assert body["features_received"] == 10
    assert body["prediction"] == pytest.approx(body["prediction"])
    assert abs(body["prediction"]) != float("inf")


def test_wrong_feature_count_is_rejected():
    response = client.post("/predict", json={"features": VALID_FEATURES[:9]})
    assert response.status_code == 422


def test_too_many_features_are_rejected():
    response = client.post("/predict", json={"features": VALID_FEATURES + [1.1]})
    assert response.status_code == 422


def test_non_numeric_feature_is_rejected():
    features = VALID_FEATURES.copy()
    features[0] = "not-a-number"
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422


@pytest.mark.parametrize("special", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_feature_is_rejected(special):
    values = [str(value) for value in VALID_FEATURES]
    values[0] = special
    # Python's JSON parser recognizes these tokens; the Pydantic validator
    # must reject them before the model is called.
    payload = '{"features":[' + ",".join(values) + ']}'
    response = client.post(
        "/predict",
        content=payload,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_missing_model_returns_503(monkeypatch):
    monkeypatch.setattr(main, "MODEL", None)
    response = client.post("/predict", json={"features": VALID_FEATURES})
    assert response.status_code == 503
    assert response.json()["detail"].startswith("Model not available")


def test_openapi_documents_prediction_endpoint():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]
