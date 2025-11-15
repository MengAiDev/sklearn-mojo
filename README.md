# sklearn-mojo

This is a pure Mojo implementation of scikit-learn-like functionality. This package provides machine learning algorithms with a familiar scikit-learn API, but implemented entirely in Mojo for maximum performance.

## Features

- Linear Regression with scikit-learn-like API
- Matrix operations utilities
- Fit, predict, and score methods
- Support for intercept fitting
- Pure Mojo implementation for maximum performance

## Quick Start

### Using the Package

```mojo
from sklearn_mojo import fit_linear_regression, predict_linear_regression, score_linear_regression

# Train a model
var (coef, intercept) = fit_linear_regression(X, y, True)

# Make predictions
var predictions = predict_linear_regression(X_test, coef, intercept)

# Evaluate model
var score = score_linear_regression(X_test, y_test, coef, intercept)
```

## Package Structure

```
sklearn_mojo/
├── __init__.mojo           # Package exports
├── matrix_utils.mojo       # Matrix operations
└── linear_regression.mojo  # Linear regression implementation
```

## API Reference

### Core Functions

- `fit_linear_regression(X, y, fit_intercept=True)`: Fits a linear regression model
- `predict_linear_regression(X, coef, intercept)`: Makes predictions using fitted parameters  
- `score_linear_regression(X, y, coef, intercept)`: Calculates R² score

### Matrix Utilities

- `print_list(lst)`: Pretty prints a list
- `add_ones_column(X)`: Adds a column of ones for intercept
- `matmul(A, B)`: Matrix multiplication
- `transpose(X)`: Matrix transpose
- `solve_system(A, b)`: Solves linear system Ax = b

## Implementation Details

The implementation uses the normal equation method to solve linear regression:
- Coefficients are calculated using (X^T * X)^(-1) * X^T * y
- Matrix operations are implemented from scratch in pure Mojo
- Supports fitting with or without intercept
- Includes regularization for numerical stability

## Documentation

For detailed documentation, see the `docs/` directory:
- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api_reference.md)