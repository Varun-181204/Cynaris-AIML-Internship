"""Exploratory Data Analysis for Cynaris AI/ML Internship — W1D4."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_dataset() -> pd.DataFrame:
    """Load the India Census 2011 dataset."""
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = (
        project_root
        / "week-01"
        / "day-04-eda"
        / "data"
        / "india_districts_census_2011.csv"
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    return pd.read_csv(dataset_path)


def run_eda(dataset: pd.DataFrame) -> None:
    """Run descriptive EDA and save the analysis output."""
    print("=== DATASET INFORMATION ===")
    dataset.info()

    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(dataset.describe())

    print("\n=== MISSING VALUES ===")
    print(dataset.isnull().sum().sort_values(ascending=False).head(15))

    numeric_data = dataset.select_dtypes(include="number")

    print("\n=== NUMERIC COLUMNS ===")
    print(list(numeric_data.columns))

    print("\n=== TOP 10 STATES BY DISTRICT COUNT ===")
    print(dataset["State name"].value_counts().head(10))

    output_directory = (
        Path(__file__).resolve().parents[3] / "week-01" / "day-04-eda" / "outputs"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    numeric_data.hist(figsize=(16, 12))
    plt.tight_layout()
    plt.savefig(output_directory / "numeric_distributions.png")
    plt.close()

    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm")
    plt.title("Correlation Heatmap — India Census 2011")
    plt.tight_layout()
    plt.savefig(output_directory / "correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    dataset["State name"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 States by District Count")
    plt.xlabel("State")
    plt.ylabel("Number of Districts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_directory / "top_10_state_counts.png")
    plt.close()

    print("\nEDA plots saved successfully.")


if __name__ == "__main__":
    census_data = load_dataset()
    run_eda(census_data)
