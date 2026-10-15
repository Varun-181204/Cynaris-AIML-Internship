"""FastAPI service for the W4D3 regression model."""

import math
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = (
    REPO_ROOT
    / "week-04"
    / "day-03-model-serialisation"
    / "outputs"
    / "regression_model.joblib"
)

app = FastAPI(
    title="Cynaris Model Serving API",
    description="Serves predictions from the W4D3 regression model.",
    version="1.0.0",
)

# Only load trusted model artifacts: joblib/pickle-based formats can execute code.
MODEL = None
MODEL_LOAD_ERROR = None

try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)
except Exception as exc:
    MODEL_LOAD_ERROR = str(exc)


class PredictionRequest(BaseModel):
    """A single prediction request with exactly ten features."""

    features: list[float] = Field(
        ...,
        min_length=10,
        max_length=10,
        description="Exactly 10 numeric feature values in training order.",
        examples=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
    )

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Every feature must be a finite number.")
        return values


class PredictionResponse(BaseModel):
    prediction: float
    model: str
    features_received: int


@app.get("/")
def root() -> dict[str, str]:
    """Describe the API and its documentation."""
    return {
        "service": "Cynaris Model Serving API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, object]:
    """Report whether the model was loaded successfully."""
    return {
        "status": "ok" if MODEL is not None else "error",
        "model_loaded": MODEL is not None,
        "model_artifact": MODEL_PATH.name,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Return one regression prediction for ten input features."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is unavailable: {MODEL_LOAD_ERROR}",
        )

    try:
        feature_array = np.asarray([request.features], dtype=float)
        result = float(MODEL.predict(feature_array)[0])

        if not math.isfinite(result):
            raise ValueError("The model returned a non-finite prediction.")

        return PredictionResponse(
            prediction=result,
            model="w4d3-ridge-regression",
            features_received=len(request.features),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the input and model configuration.",
        ) from exc
