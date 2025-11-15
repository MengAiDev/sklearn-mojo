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

## Documentation

For detailed documentation, see the `docs/` directory:
- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api_reference.md)
