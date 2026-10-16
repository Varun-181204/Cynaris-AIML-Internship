# W4D5 — Sentiment Classifier

## Objective
Train and compare Logistic Regression and Random Forest for binary sentiment classification. This task demonstrates a repeatable ML workflow from text preprocessing through evaluation and evidence generation.

## Implementation
- TF-IDF text features with unigrams and bigrams.
- Stratified 75/25 train/test split with a fixed random seed.
- Logistic Regression and Random Forest pipelines using the same split.
- Accuracy, precision, recall, F1-score, ROC AUC, and classification reports.
- Confusion matrices and ROC-AUC curves saved as image evidence.

## Dataset note
This task uses a small, deterministic synthetic review dataset embedded in the source code. It is suitable for demonstrating the workflow and automated tests, but it is not a benchmark and results should not be interpreted as real-world sentiment performance. The high evaluation scores may reflect the repetitive synthetic examples. A next iteration should use a documented public dataset and a held-out test set.

## Setup
From the repository root with the internship virtual environment activated, install the task dependencies if needed:

```powershell
python -m pip install -r .\week-04\day-05-sentiment-classifier\requirements.txt
```

## Run
From the repository root:

```powershell
python .\week-04\day-05-sentiment-classifier\src\sentiment_classifier.py
python -m unittest discover -s .\week-04\day-05-sentiment-classifier\tests -p "test_*.py" -v
```

## Evidence
Running the script generates:
- `outputs/classification_report.txt`
- `outputs/metrics.json`
- `outputs/model_comparison.csv`
- `outputs/confusion_matrices.png`
- `outputs/roc_auc_curves.png`

Test output is recorded in:
- `outputs/test_results.txt`

## Results interpretation
On the current synthetic test split:
- Logistic Regression: accuracy 1.00, precision 1.00, recall 1.00, F1-score 1.00, ROC AUC 1.00.
- Random Forest: accuracy 0.96, precision 1.00, recall 0.92, F1-score approximately 0.9583, ROC AUC 1.00.

These scores describe only this synthetic dataset and split. They do not demonstrate equivalent performance on real user reviews. Compare precision and recall alongside accuracy; accuracy alone can hide class-specific errors. ROC AUC measures how well model scores rank positive examples above negative examples across thresholds.

## Limitations and next steps
- The dataset is small and synthetic, so generalization is limited.
- Text preprocessing and model parameters are intentionally simple.
- Future work: use a public labeled sentiment dataset, tune hyperparameters, evaluate repeated splits, track experiments with MLflow, and expose the chosen model through a validated API if required.

## Safe model handling
Only load serialized model artifacts from trusted sources. This task does not load external serialized models.
