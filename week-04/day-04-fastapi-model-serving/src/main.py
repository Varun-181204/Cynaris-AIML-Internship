"""FastAPI service for the W4D3 regression model."""

import logging
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cynaris.model_api")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "week-04"
    / "day-03-model-serialisation"
    / "outputs"
    / "regression_model.joblib"
)

MODEL_PATH = Path(
    os.environ.get("CYNARIS_MODEL_PATH", str(DEFAULT_MODEL_PATH))
).expanduser().resolve()

app = FastAPI(
    title="Cynaris Model Serving API",
    description="REST API for the W4D3 regression model.",
    version="1.0.0",
)

# Only load trusted model artifacts: joblib/pickle-based formats can execute code.
MODEL = None
MODEL_LOAD_ERROR = None


def load_model() -> None:
    """Load the configured trusted model artifact without crashing startup."""
    global MODEL, MODEL_LOAD_ERROR

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {MODEL_PATH}"
            )

        MODEL = joblib.load(MODEL_PATH)
        MODEL_LOAD_ERROR = None

        logger.info(
            "Regression model loaded successfully from %s",
            MODEL_PATH,
        )
    except Exception as exc:
        MODEL = None
        MODEL_LOAD_ERROR = str(exc)
        logger.exception("Unable to load the regression model")


load_model()


def sanitize_error_value(value: Any) -> Any:
    """Make validation error details safe for strict JSON serialization."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value

    if isinstance(value, dict):
        return {
            str(key): sanitize_error_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [sanitize_error_value(item) for item in value]

    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return safe 422 responses, including for NaN and Infinity inputs."""
    errors = []

    for error in exc.errors():
        errors.append(
            {
                "type": error.get("type", "validation_error"),
                "loc": list(error.get("loc", [])),
                "msg": error.get("msg", "Invalid request"),
                "input": sanitize_error_value(error.get("input")),
            }
        )

    return JSONResponse(
        status_code=422,
        content={"detail": errors},
    )


class PredictionRequest(BaseModel):
    """Ten finite numeric features in the original training order."""

    features: list[float] = Field(
        ...,
        min_length=10,
        max_length=10,
        description=(
            "Exactly 10 finite numeric features, in the same order "
            "used when training the W4D3 regression model."
        ),
        examples=[
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ],
    )

    @field_validator("features")
    @classmethod
    def require_finite_values(
        cls,
        values: list[float],
    ) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Every feature must be a finite number.")
        return values


class PredictionResponse(BaseModel):
    """Structured response containing a model prediction."""

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
            detail=(
                "Model not available. Check server logs "
                "and model configuration."
            ),
        )

    try:
        features = np.asarray([request.features], dtype=float)
        result = float(MODEL.predict(features)[0])

        if not math.isfinite(result):
            raise ValueError("Model produced a non-finite prediction.")

        logger.info(
            "Prediction completed for %d features",
            len(request.features),
        )

        return PredictionResponse(
            prediction=result,
            model="w4d3-ridge-regression",
            features_received=len(request.features),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs for details.",
        ) from exc