# W4D4 — FastAPI Model Serving Endpoint

## Overview
Serves the trusted W4D3 Ridge regression pipeline through a FastAPI REST API.

## Setup
Run from the repository root in the activated virtual environment:

```powershell
python -m pip install -r "week-04\day-04-fastapi-model-serving\requirements.txt"
```

The service uses the W4D3 model at
`week-04/day-03-model-serialisation/outputs/regression_model.joblib`.
Override this path with `CYNARIS_MODEL_PATH` if required.

## Start the server

```powershell
python -m uvicorn main:app --app-dir "week-04\day-04-fastapi-model-serving\src" --host 127.0.0.1 --port 8000
```

Interactive API documentation: `http://127.0.0.1:8000/docs`

## API contract

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Model-load status |
| POST | `/predict` | Generate a regression prediction |

Example request body for `/predict`:

```json
{
  "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}
```

Provide exactly ten finite numeric features in the order expected by the W4D3 training pipeline.

Successful response includes `prediction`, `model`, and `features_received`.

## Error responses
- `422`: malformed input, wrong feature count, non-numeric or non-finite values.
- `503`: model artifact unavailable or failed to load.
- `500`: unexpected prediction failure; inspect server logs for details.

## Tests and evidence

```powershell
python -m pytest "week-04\day-04-fastapi-model-serving\tests" -v
python "week-04\day-04-fastapi-model-serving\src\generate_evidence.py"
```

Evidence is written to `outputs/api_evidence.json`. Test output should be captured in `outputs/test_results.txt`.

## Security and limitations
Only load model artifacts from trusted sources. This local demonstration does not yet implement authentication, rate limiting, or production deployment hardening.
