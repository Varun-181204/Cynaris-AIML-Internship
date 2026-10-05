from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

print("Dataset shape:", X.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(housing.target_names)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# Create models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001),
}


# Store results
results = []
trained_models = {}


# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append(
        {
            "Model": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }
    )

    trained_models[name] = model


# Linear Regression coefficients
linear_model = trained_models["Linear Regression"]

coefficients = pd.DataFrame(
    {
        "Feature": X.columns,
        "Coefficient": linear_model.coef_,
    }
)

print("\nLinear Regression Coefficients:")
print(coefficients)

print("\nLinear Regression Intercept:", linear_model.intercept_)


# Results table
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.to_string(index=False))


# Create output directory before saving files
output_dir = "week-03/day-01-linear-regression/outputs"
os.makedirs(output_dir, exist_ok=True)


# Save model comparison results
results_df.to_csv(
    f"{output_dir}/model_comparison.csv",
    index=False,
)

print("\nResults saved successfully:")
print("- outputs/model_comparison.csv")


# Predicted vs Actual plot
y_pred_linear = trained_models["Linear Regression"].predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.4)

min_value = min(y_test.min(), y_pred_linear.min())
max_value = max(y_test.max(), y_pred_linear.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Predicted vs Actual House Values")
plt.tight_layout()

plt.savefig(
    f"{output_dir}/predicted_vs_actual.png"
)

plt.close()


# Residual plot
residuals = y_test - y_pred_linear

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_linear, residuals, alpha=0.4)

plt.axhline(
    y=0,
    linestyle="--",
)

plt.xlabel("Predicted House Value")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()

plt.savefig(
    f"{output_dir}/residuals.png"
)

plt.close()


print("\nPlots saved successfully:")
print("- outputs/predicted_vs_actual.png")
print("- outputs/residuals.png")