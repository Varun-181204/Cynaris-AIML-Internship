# W4D3 — Model Serialisation with joblib and pickle

## Objective
Train a reproducible regression model, serialize it using joblib and pickle, reload both saved models, and verify that their predictions match the original model.

## Implementation
- Dataset: synthetic regression data generated with scikit-learn.
- Split: 80% training and 20% testing, using a fixed random seed.
- Model: StandardScaler and Ridge regression in a scikit-learn Pipeline.
- Formats: joblib and Python pickle.
- Validation: compare predictions from the original and reloaded models.
- Evidence: prediction comparison CSV and experiment summary JSON.

## Recorded execution results
- Unit tests: 4 passed.
- Test samples: 120.
- MAE: 13.6800.
- RMSE: 17.0787.
- R-squared: 0.9743.
- joblib file size: 1,329 bytes.
- pickle file size: 1,117 bytes.
- Both serialized models preserved the original model's predictions in the recorded experiment.

These values were verified in the latest local run, with all 4 unit tests passing.

## Reproduce
From the repository root, with the project virtual environment activated:

```powershell
python -m unittest discover -s "week-04\day-03-model-serialisation\tests" -p "test_*.py" -v
python "week-04\day-03-model-serialisation\src\model_serialisation.py"
```

## Generated evidence
- `outputs/regression_model.joblib`
- `outputs/regression_model.pkl`
- `outputs/prediction_comparison.csv`
- `outputs/experiment_summary.json`

## Serialization safety
Only load joblib or pickle files from trusted sources. Deserializing untrusted files can execute arbitrary code. Model artifacts should also be versioned and tested against compatible dependency versions before deployment.

## Limitations and future work
- The experiment uses synthetic data rather than a production dataset.
- File size is reported for this run and is not a general performance benchmark.
- Production use should include dependency pinning, artifact integrity checks, and validation after loading.

