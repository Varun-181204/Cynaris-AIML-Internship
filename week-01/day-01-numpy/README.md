# W1D1 — NumPy Fundamentals

## Objective

Complete the Week 1 Day 1 practical task for the Cynaris AI/ML Internship by implementing fundamental NumPy array operations and statistical calculations.

## Topics Covered

* 1D, 2D, and 3D NumPy arrays
* Array shapes
* Broadcasting
* Vectorized operations
* Matrix multiplication
* Mean and standard deviation
* Correlation
* Reading CSV data using Pandas

## Dataset

The practical task uses the Iris dataset containing 150 samples and four numerical features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The dataset is stored in:

```text
data/iris.csv
```

## Project Structure

```text
day-01-numpy/
├── data/
│   └── iris.csv
├── outputs/
│   └── results.txt
├── src/
│   └── numpy_fundamentals.py
└── README.md
```

## Run the Project

From the repository root:

```powershell
python .\week-01\day-01-numpy\src\numpy_fundamentals.py
```

## Output Evidence

The generated results are stored in:

```text
outputs/results.txt
```

The output includes array shapes, broadcasting results, vectorized operations, matrix multiplication, dataset statistics, and the correlation matrix.

## Code Quality

* Python type hints are used for function signatures.
* `pathlib.Path` is used for the dataset path.
* NumPy vectorized operations are used instead of Python loops for array calculations.
* Code is formatted using Black.
* No secrets or environment-specific credentials are included.
