"""Data visualization workflow for Cynaris AI/ML Internship — W1D5."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_dataset() -> pd.DataFrame:
    """Load the India Census 2011 district dataset."""
    project_root = Path(__file__).resolve().parents[3]

    dataset_path = (
        project_root
        / "week-01"
        / "day-05-visualization"
        / "data"
        / "india_districts_census_2011.csv"
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    return pd.read_csv(dataset_path)


def create_visualizations(dataset: pd.DataFrame) -> None:
    """Create and save Matplotlib and Seaborn visualizations."""
    output_directory = (
        Path(__file__).resolve().parents[3]
        / "week-01"
        / "day-05-visualization"
        / "outputs"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    numeric_data = dataset.select_dtypes(include="number")

    population_column = "Population"

    plt.figure(figsize=(10, 6))
    plt.hist(dataset[population_column], bins=30)
    plt.title("District Population Distribution")
    plt.xlabel("Population")
    plt.ylabel("Number of Districts")
    plt.tight_layout()
    plt.savefig(output_directory / "population_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(y=dataset[population_column])
    plt.title("District Population Box Plot")
    plt.ylabel("Population")
    plt.tight_layout()
    plt.savefig(output_directory / "population_boxplot.png")
    plt.close()

    correlation_columns = numeric_data.columns[:12]
    correlation_matrix = dataset[correlation_columns].corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap — Selected Census Features")
    plt.tight_layout()
    plt.savefig(output_directory / "correlation_heatmap.png")
    plt.close()

    state_counts = dataset["State name"].value_counts().head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=state_counts.index, y=state_counts.values)
    plt.title("Top 10 States by District Count")
    plt.xlabel("State")
    plt.ylabel("Number of Districts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_directory / "top_10_states.png")
    plt.close()

    print("Visualizations created successfully.")
    print(f"Output directory: {output_directory}")


if __name__ == "__main__":
    census_data = load_dataset()
    create_visualizations(census_data)
