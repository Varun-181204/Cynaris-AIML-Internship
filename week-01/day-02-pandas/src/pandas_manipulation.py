"""Pandas data manipulation for Cynaris AI/ML Internship — W1D2."""

from pathlib import Path

import pandas as pd


def load_dataset() -> pd.DataFrame:
    """Load the India Census 2011 district dataset."""
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = (
        project_root
        / "week-01"
        / "day-02-pandas"
        / "data"
        / "india_districts_census_2011.csv"
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    return pd.read_csv(dataset_path)


def inspect_dataset(dataset: pd.DataFrame) -> None:
    """Display basic information about the dataset."""
    print("Dataset shape:")
    print(dataset.shape)

    print("\nDataset data types:")
    print(dataset.dtypes)

    print("\nFirst 10 rows:")
    print(dataset.head(10))


def filter_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Filter districts with population greater than one million."""
    # Filter rows using a population condition.
    filtered_data = dataset[dataset["Population"] > 1_000_000]

    print("\nFiltered districts with population > 1,000,000:")
    print(filtered_data[["State name", "District name", "Population"]].head(10))

    return filtered_data


def group_by_state(dataset: pd.DataFrame) -> pd.DataFrame:
    """Calculate total population for each state."""
    # Group districts by state and calculate total population.
    state_population = (
        dataset.groupby("State name", as_index=False)["Population"]
        .sum()
        .sort_values("Population", ascending=False)
    )

    print("\nTotal population by state:")
    print(state_population.head(10))

    return state_population


def merge_state_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """Merge state population data with state district counts."""
    state_population = (
        dataset.groupby("State name", as_index=False)["Population"]
        .sum()
        .rename(columns={"Population": "Total Population"})
    )

    district_counts = (
        dataset.groupby("State name", as_index=False)["District name"]
        .count()
        .rename(columns={"District name": "District Count"})
    )

    # Merge two DataFrames using the common state name column.
    merged_data = pd.merge(
        state_population,
        district_counts,
        on="State name",
        how="inner",
    )

    print("\nMerged state population and district count:")
    print(merged_data.head(10))

    return merged_data


def create_population_pivot(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table for population statistics by state."""
    # Create a pivot table to summarize population statistics by state.
    population_pivot = pd.pivot_table(
        dataset,
        values="Population",
        index="State name",
        aggfunc=["sum", "mean"],
    )

    print("\nPopulation pivot table:")
    print(population_pivot.head(10))

    return population_pivot


def export_cleaned_data(dataset: pd.DataFrame) -> None:
    """Export the cleaned dataset to CSV and Parquet and compare file sizes."""
    project_root = Path(__file__).resolve().parents[3]
    output_directory = project_root / "week-01" / "day-02-pandas" / "outputs"

    output_directory.mkdir(parents=True, exist_ok=True)

    cleaned_data = dataset.dropna().copy()

    csv_path = output_directory / "cleaned_census_data.csv"
    parquet_path = output_directory / "cleaned_census_data.parquet"

    cleaned_data.to_csv(csv_path, index=False)
    cleaned_data.to_parquet(parquet_path, index=False)

    csv_size = csv_path.stat().st_size
    parquet_size = parquet_path.stat().st_size

    print("\nExported files:")
    print(f"CSV: {csv_path.name} - {csv_size:,} bytes")
    print(f"Parquet: {parquet_path.name} - {parquet_size:,} bytes")


if __name__ == "__main__":
    census_data = load_dataset()
    inspect_dataset(census_data)
    filter_dataset(census_data)
    group_by_state(census_data)
    merge_state_data(census_data)
    create_population_pivot(census_data)
    export_cleaned_data(census_data)
