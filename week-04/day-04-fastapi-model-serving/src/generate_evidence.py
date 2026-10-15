"""Generate repeatable JSON evidence for the API endpoints."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

evidence = {
    "health": {
        "status_code": (health := client.get("/health")).status_code,
        "response": health.json(),
    },
    "valid_prediction": {
        "status_code": (prediction := client.post(
            "/predict", json={"features": features}
        )).status_code,
        "response": prediction.json(),
    },
    "invalid_feature_count": {
        "status_code": (invalid := client.post(
            "/predict", json={"features": features[:9]}
        )).status_code,
        "response": invalid.json(),
    },
}

output_path = Path(__file__).resolve().parents[1] / "outputs" / "api_evidence.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

print(json.dumps(evidence, indent=2))
print(f"\nEvidence saved to: {output_path}")
