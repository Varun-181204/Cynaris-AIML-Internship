"""Generate JSON evidence for successful and rejected API requests."""

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from main import app  # noqa: E402

client = TestClient(app)
features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

health = client.get("/health")
prediction = client.post("/predict", json={"features": features})
wrong_count = client.post("/predict", json={"features": features[:9]})
nonnumeric = client.post(
    "/predict",
    json={"features": ["invalid", *features[1:]]},
)

evidence = {
    "health": {"status_code": health.status_code, "response": health.json()},
    "valid_prediction": {
        "status_code": prediction.status_code,
        "response": prediction.json(),
    },
    "wrong_feature_count": {
        "status_code": wrong_count.status_code,
        "response": wrong_count.json(),
    },
    "non_numeric_feature": {
        "status_code": nonnumeric.status_code,
        "response": nonnumeric.json(),
    },
}

output_path = SRC_DIR.parent / "outputs" / "api_evidence.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
print(json.dumps(evidence, indent=2))
print(f"Evidence saved to: {output_path}")
