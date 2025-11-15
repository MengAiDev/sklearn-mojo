# API Reference

Complete reference for all functions and types in sklearn-mojo package.

## Core Linear Regression Functions

### `fit_linear_regression`

Fits a linear regression model to training data using the normal equation method.

```mojo
fn fit_linear_regression(X: List[List[Float64]], y: List[Float64], fit_intercept: Bool = True) -> (List[Float64], Float64)
```

**Parameters:**
- `X`: Training feature matrix (n_samples × n_features)
- `y`: Training target values (length n_samples)
- `fit_intercept`: Whether to fit an intercept term (default: True)

**Returns:**
- `coefficients`: Model coefficients (length n_features or n_features + 1 if intercept)
- `intercept`: Intercept value (0.0 if fit_intercept is False)

**Example:**
```mojo
var X: List[List[Float64]] = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]]
var y: List[Float64] = [8.1, 13.9, 9.8]
var (coef, intercept) = fit_linear_regression(X, y, True)
```

**Implementation Details:**
- Uses normal equation: (X^T * X)^(-1) * X^T * y
- Adds small regularization (1e-8) to diagonal for numerical stability
- Automatically handles matrix operations internally

### `predict_linear_regression`

Makes predictions using a fitted linear regression model.

```mojo
fn predict_linear_regression(X: List[List[Float64]], coef: List[Float64], intercept: Float64) -> List[Float64]
```

**Parameters:**
- `X`: Feature matrix for predictions (n_samples × n_features)
- `coef`: Model coefficients from `fit_linear_regression`
- `intercept`: Intercept value from `fit_linear_regression`

**Returns:**
- `predictions`: Predicted values (length n_samples)

**Example:**
```mojo
var X_test: List[List[Float64]] = [[1.0, 1.0], [2.0, 2.0]]
var predictions = predict_linear_regression(X_test, coef, intercept)
```

**Computation:**
- prediction = intercept + sum(coef[i] * X[i][j] for j in features)

### `score_linear_regression`

Calculates the coefficient of determination (R²) score.

```mojo
fn score_linear_regression(X: List[List[Float64]], y: List[Float64], coef: List[Float64], intercept: Float64) -> Float64
```

**Parameters:**
- `X`: Feature matrix
- `y`: True target values
- `coef`: Model coefficients
- `intercept`: Model intercept

**Returns:**
- `score`: R² score (1.0 is perfect, 0.0 is baseline)

**Formula:**
```
R² = 1 - (SS_res / SS_tot)
where:
SS_res = Σ(y_true - y_pred)²
SS_tot = Σ(y_true - y_mean)²
```

**Edge Cases:**
- Returns 1.0 if SS_tot = 0 and SS_res = 0 (constant target)
- Returns 0.0 if SS_tot = 0 but SS_res > 0

## Matrix Utility Functions

### `print_list`

Pretty prints a list of Float64 values.

```mojo
fn print_list(lst: List[Float64])
```

**Parameters:**
- `lst`: List of Float64 values to print

**Output Format:**
```
[1.0, 2.5, 3.14]
```

### `add_ones_column`

Adds a column of ones to the left side of matrix (for intercept term).

```mojo
fn add_ones_column(X: List[List[Float64]]) -> List[List[Float64]]
```

**Parameters:**
- `X`: Input matrix (n_samples × n_features)

**Returns:**
- Enhanced matrix with ones column (n_samples × (n_features + 1))

**Example:**
```mojo
var X: List[List[Float64]] = [[1.0, 2.0], [3.0, 4.0]]
var X_with_intercept = add_ones_column(X)
// Result: [[1.0, 1.0, 2.0], [1.0, 3.0, 4.0]]
```

### `matmul`

Performs matrix multiplication.

```mojo
fn matmul(A: List[List[Float64]], B: List[List[Float64]]) -> List[List[Float64]]
```

**Parameters:**
- `A`: Left matrix (m × n)
- `B`: Right matrix (n × p)

**Returns:**
- Result matrix (m × p)

**Requirements:**
- Number of columns in A must equal number of rows in B
- Returns empty matrix if dimensions don't match

**Complexity:** O(m × n × p)

### `transpose`

Computes the transpose of a matrix.

```mojo
fn transpose(X: List[List[Float64]]) -> List[List[Float64]]
```

**Parameters:**
- `X`: Input matrix (m × n)

**Returns:**
- Transposed matrix (n × m)

**Edge Cases:**
- Returns empty matrix if input is empty
- Single row/column matrices are handled correctly

### `solve_system`

Solves a system of linear equations using Gaussian elimination.

```mojo
fn solve_system(A: List[List[Float64]], b: List[Float64]) -> List[Float64]
```

**Parameters:**
- `A`: Coefficient matrix (n × n)
- `b`: Right-hand side vector (length n)

**Returns:**
- Solution vector x such that A × x = b

**Algorithm:**
- Gaussian elimination with partial pivoting
- Forward elimination followed by back substitution

**Numerical Stability:**
- Includes small regularization for ill-conditioned matrices
- Handles singular matrices gracefully

## Package Structure

### `sklearn_mojo.__init__`

The package initialization file exports all public functions:

```mojo
from .matrix_utils import *
from .linear_regression import *
```

**Exported Functions:**
- `fit_linear_regression`
- `predict_linear_regression` 
- `score_linear_regression`
- `print_list`
- `add_ones_column`
- `matmul`
- `transpose`
- `solve_system`

### File Organization

```
sklearn_mojo/
├── __init__.mojo              # Package exports
├── matrix_utils.mojo          # Matrix operations
└── linear_regression.mojo     # ML algorithms
```

## Data Types

### Matrix Format

All matrices use `List[List[Float64]]` format:
- Outer list: rows
- Inner list: columns
- Example: `[[1.0, 2.0], [3.0, 4.0]]` represents:
  ```
  [1.0  2.0]
  [3.0  4.0]
  ```

### Vector Format

Vectors use `List[Float64]` format:
- Example: `[1.0, 2.0, 3.0]`

### Boolean Parameters

- `fit_intercept`: True to include intercept term, False otherwise

## Error Handling

### Matrix Dimension Mismatches

```mojo
// matmul will print error and return empty matrix
var result = matmul(A, B)  // if A.cols != B.rows
```

### Empty Input Handling

```mojo
// transpose returns empty matrix for empty input
var empty_transposed = transpose([])  // returns []
```

### Numerical Stability

- Small regularization (1e-8) added to diagonal in normal equation
- Partial pivoting in Gaussian elimination
- Graceful handling of singular matrices

## Performance Considerations

### Time Complexity

- `fit_linear_regression`: O(n²m) where n=features, m=samples
- `predict_linear_regression`: O(nm)
- `matmul`: O(mnp) for matrices (m×n) × (n×p)
- `solve_system`: O(n³) for n×n system

### Memory Usage

- All operations create new matrices (copy semantics)
- Consider memory usage for large datasets
- Use batch processing for very large datasets

### Optimization Tips

1. **Data Preprocessing**: Normalize features for better numerical stability
2. **Batch Processing**: Split large datasets into smaller batches
3. **Memory Management**: Avoid unnecessary copying when possible
4. **Feature Selection**: Reduce dimensionality for better performance

## Examples

### Complete Workflow

```mojo
from sklearn_mojo import fit_linear_regression, predict_linear_regression, score_linear_regression

// 1. Prepare data
var X: List[List[Float64]] = [[1.0, 2.0], [2.0, 3.0]]
var y: List[Float64] = [8.1, 13.9]

// 2. Train model
var (coef, intercept) = fit_linear_regression(X, y, True)

// 3. Make predictions
var X_test: List[List[Float64]] = [[1.5, 2.5]]
var predictions = predict_linear_regression(X_test, coef, intercept)

// 4. Evaluate model
var score = score_linear_regression(X, y, coef, intercept)
```

### Matrix Operations

```mojo
from sklearn_mojo import add_ones_column, matmul, transpose

var X: List[List[Float64]] = [[1.0, 2.0]]
var X_with_intercept = add_ones_column(X)
var Xt = transpose(X)
var XtX = matmul(Xt, X)
```

## Version Information

- **Current Version**: 1.0.0
- **Compatibility**: Mojo 0.6.0+
- **API Stability**: Stable - functions won't change signature in minor versions

## Contributing

When adding new functions to the package:

1. Follow the established naming conventions
2. Include comprehensive parameter documentation
3. Add error handling for edge cases
4. Update this API reference
5. Add examples to the usage guide