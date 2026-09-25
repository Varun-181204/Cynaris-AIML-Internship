"""NumPy fundamentals for Cynaris AI/ML Internship — W1D1."""

import numpy as np
from pathlib import Path

import numpy as np
import pandas as pd


def demonstrate_array_shapes() -> None:
    """Create 1D, 2D, and 3D arrays and display their shapes."""
    array_1d = np.array([10, 20, 30, 40, 50])

    array_2d = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
        ]
    )

    array_3d = np.array(
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
        ]
    )

    print("1D array:", array_1d)
    print("1D shape:", array_1d.shape)

    print("\n2D array:")
    print(array_2d)
    print("2D shape:", array_2d.shape)

    print("\n3D array:")
    print(array_3d)
    print("3D shape:", array_3d.shape)


def demonstrate_vectorized_operations() -> None:
    """Demonstrate broadcasting and vectorized NumPy operations."""
    scores = np.array([10, 20, 30, 40, 50])

    increased_scores = scores + 5
    doubled_scores = scores * 2

    print("\nOriginal scores:", scores)
    print("Broadcasting (+5):", increased_scores)
    print("Vectorized (*2):", doubled_scores)


def demonstrate_matrix_multiplication() -> None:
    """Perform matrix multiplication using NumPy."""
    matrix_a = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    matrix_b = np.array(
        [
            [5, 6],
            [7, 8],
        ]
    )

    result = matrix_a @ matrix_b

    print("\nMatrix A:")
    print(matrix_a)

    print("\nMatrix B:")
    print(matrix_b)

    print("\nMatrix multiplication:")
    print(result)


def calculate_dataset_statistics() -> None:
    """Calculate statistics from the Iris CSV dataset."""
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = project_root / "week-01" / "day-01-numpy" / "data" / "iris.csv"

    dataset = pd.read_csv(dataset_path)

    numerical_data = dataset[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ].to_numpy()

    mean_values = np.mean(numerical_data, axis=0)
    standard_deviation = np.std(numerical_data, axis=0)
    correlation = np.corrcoef(numerical_data, rowvar=False)

    feature_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    print("\nDataset shape:", numerical_data.shape)

    print("\nMean:")
    for name, value in zip(feature_names, mean_values):
        print(f"{name}: {value:.4f}")

    print("\nStandard deviation:")
    for name, value in zip(feature_names, standard_deviation):
        print(f"{name}: {value:.4f}")

    print("\nCorrelation matrix:")
    print(np.round(correlation, 4))


if __name__ == "__main__":
    demonstrate_array_shapes()
    demonstrate_vectorized_operations()
    demonstrate_matrix_multiplication()
    calculate_dataset_statistics()
