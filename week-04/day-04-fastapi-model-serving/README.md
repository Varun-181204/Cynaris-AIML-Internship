# W4D4 — FastAPI Model Serving Endpoint

## Objective
Serve the regression pipeline created and serialized in W4D3 through a validated REST API.

## Design
- FastAPI exposes `/`, `/health`, and `/predict`.
- The service loads the trusted W4D3 joblib artifact.
- `/predict` accepts exactly ten numeric features in the same order as model training.
- Invalid input is rejected with HTTP 422.
- If the artifact cannot be loaded, health reports the failure and predictions return HTTP 503.
- Unexpected prediction errors return HTTP 500 without exposing internal exception details.

## Run locally
From the repository root, with the project virtual environment active:

```powershell
python -m pip install -r "week-04\day-04-fastapi-model-serving\requirements.txt"
python -m uvicorn main:app --app-dir "week-04\day-04-fastapi-model-serving\src" --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`.

## Test
```powershell
python -m pytest "week-04\day-04-fastapi-model-serving\tests" -v
python "week-04\day-04-fastapi-model-serving\src\generate_evidence.py"
```

## Evidence
- `outputs/test_results.txt`: automated test output.
- `outputs/api_evidence.json`: health, prediction, and invalid-input responses.

## Security and limitations
Load only trusted model artifacts. The API currently serves a single local regression model and expects ten features in training order. It does not yet include authentication, rate limiting, or production deployment configuration.

## Review status
CIA mentor interactions are to be recorded after the two required reviews are completed.
