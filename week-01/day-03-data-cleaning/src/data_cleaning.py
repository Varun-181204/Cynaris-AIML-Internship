"""Data loading, cleaning, and inspection for Cynaris AI/ML Internship — W1D3."""

from pathlib import Path

import pandas as pd


def load_dataset() -> pd.DataFrame:
    """Load the India Census 2011 district dataset."""
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = (
        project_root
        / "week-01"
        / "day-03-data-cleaning"
        / "data"
        / "india_districts_census_2011.csv"
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    return pd.read_csv(dataset_path)


def inspect_dataset(dataset: pd.DataFrame) -> None:
    """Display basic dataset inspection information."""
    print("Original dataset shape:")
    print(dataset.shape)

    print("\nFirst 5 rows:")
    print(dataset.head())

    print("\nMissing values by column:")
    print(dataset.isna().sum().sort_values(ascending=False).head(15))

    print("\nDuplicate rows:")
    print(dataset.duplicated().sum())

    print("\nData types:")
    print(dataset.dtypes)


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicate rows and handle missing values."""
    cleaned_data = dataset.copy()

    duplicate_count = cleaned_data.duplicated().sum()
    cleaned_data = cleaned_data.drop_duplicates()

    missing_before = int(cleaned_data.isna().sum().sum())

    # Fill numeric missing values with the median of each numeric column.
    numeric_columns = cleaned_data.select_dtypes(include="number").columns
    cleaned_data[numeric_columns] = cleaned_data[numeric_columns].fillna(
        cleaned_data[numeric_columns].median()
    )

    # Fill missing text values with a clear placeholder.
    text_columns = cleaned_data.select_dtypes(include="str").columns
    cleaned_data[text_columns] = cleaned_data[text_columns].fillna("Unknown")

    missing_after = int(cleaned_data.isna().sum().sum())

    print("\nCleaning summary:")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Missing values before cleaning: {missing_before}")
    print(f"Missing values after cleaning: {missing_after}")
    print(f"Cleaned dataset shape: {cleaned_data.shape}")

    return cleaned_data


def save_cleaned_dataset(dataset: pd.DataFrame) -> None:
    """Save the cleaned dataset as a CSV file."""
    project_root = Path(__file__).resolve().parents[3]
    output_directory = project_root / "week-01" / "day-03-data-cleaning" / "outputs"

    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / "cleaned_census_data.csv"
    dataset.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    census_data = load_dataset()
    inspect_dataset(census_data)
    cleaned_census_data = clean_dataset(census_data)
    save_cleaned_dataset(cleaned_census_data)
