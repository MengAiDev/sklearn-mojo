# sklearn-mojo

This is a pure Mojo implementation of scikit-learn-like functionality. This package provides machine learning algorithms with a familiar scikit-learn API, but implemented entirely in Mojo for maximum performance.

## Features

- **Linear Regression**: Continuous value prediction with normal equation solver
- **Logistic Regression**: Binary classification with gradient descent optimizer
- Matrix operations utilities
- Fit, predict, and score methods for both algorithms
- Support for intercept fitting
- Probability predictions for classification tasks
- Multiple evaluation metrics (accuracy, log-loss, R²)
- Pure Mojo implementation for maximum performance

## Quick Start

### Using the Package

#### Linear Regression (Continuous Values)

```mojo
from sklearn_mojo import fit_linear_regression, predict_linear_regression, score_linear_regression

# Train a linear regression model
var (coef, intercept) = fit_linear_regression(X, y, True)

# Make predictions
var predictions = predict_linear_regression(X_test, coef, intercept)

# Evaluate model
var score = score_linear_regression(X_test, y_test, coef, intercept)
```

#### Logistic Regression (Binary Classification)

```mojo
from sklearn_mojo import fit_logistic_regression, predict_logistic_regression, predict_proba_logistic_regression, score_logistic_regression

# Train a logistic regression model
var (coef, intercept) = fit_logistic_regression(X_binary, y_binary, True, 0.1, 1000)

# Make binary predictions
var predictions = predict_logistic_regression(X_test, coef, intercept)

# Get probabilities for threshold tuning
var probabilities = predict_proba_logistic_regression(X_test, coef, intercept)

# Evaluate model
var accuracy = score_logistic_regression(X_test, y_test, coef, intercept)
```

## Package Structure

```
sklearn_mojo/
├── __init__.mojo              # Package exports
├── matrix_utils.mojo          # Matrix operations & utilities
├── linear_regression.mojo     # Linear regression implementation
└── logistic_regression.mojo   # Logistic regression implementation
```

## Documentation

For detailed documentation, see the `docs/` directory:
- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api_reference.md)
