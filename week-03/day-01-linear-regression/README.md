# W3D1 — Linear Regression

## Objective

Train and evaluate regression models using scikit-learn and compare Linear Regression, Ridge, and Lasso.

## Dataset

California Housing dataset provided by scikit-learn.

- Samples: 20,640
- Features: 8
- Target: `MedHouseVal`
- Train/Test Split: 80/20
- Random State: 42

## Models

- Linear Regression
- Ridge Regression
- Lasso Regression

## Evaluation Metrics

- MSE
- RMSE
- MAE
- R²

## Results

| Model | MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 0.555892 | 0.745581 | 0.533200 | 0.575788 |
| Ridge | 0.555803 | 0.745522 | 0.533204 | 0.575855 |
| Lasso | 0.553894 | 0.744241 | 0.533286 | 0.577312 |

Lasso achieved the highest test-set R² among the three models.

## Outputs

The `outputs` directory contains:

- `model_comparison.csv`
- `predicted_vs_actual.png`
- `residuals.png`

## CIA Review

The implementation was reviewed using CIA Full-Stack Mentor Mode, with two CIA interactions completed before committing.