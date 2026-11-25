import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "evaluation_dataset.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth.json"


def main():
    samples = json.loads(DATASET.read_text(encoding="utf-8"))
    references = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))

    reference_map = {item["id"]: item["reference"] for item in references}

    for sample in samples:
        sample["reference"] = reference_map[sample["id"]]

    DATASET.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Dataset prepared successfully.")
    print("Evaluation samples:", len(samples))
    print("Ground-truth references:", len(reference_map))


if __name__ == "__main__":
    main()
