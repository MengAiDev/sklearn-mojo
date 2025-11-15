# Usage Guide

This guide demonstrates how to use the sklearn-mojo package for linear regression tasks.

## Basic Usage

### Import the Package

```mojo
from sklearn_mojo import fit_linear_regression, predict_linear_regression, score_linear_regression
```

### Import Matrix Utilities (Optional)

```mojo
from sklearn_mojo import print_list, add_ones_column, matmul, transpose
```

## Linear Regression Example

### Step 1: Prepare Your Data

```mojo
# Training data (features)
var X: List[List[Float64]] = []

# Sample dataset: predict house prices based on size and rooms
var house1: List[Float64] = []
house1.append(1200.0)  # size in sq ft
house1.append(3.0)     # number of rooms
X.append(house1.copy())

var house2: List[Float64] = []
house2.append(1600.0)
house2.append(4.0)
X.append(house2.copy())

var house3: List[Float64] = []
house3.append(800.0)
house3.append(2.0)
X.append(house3.copy())

# Target values (prices)
var y: List[Float64] = []
y.append(350000.0)  # price for house1
y.append(480000.0)  # price for house2
y.append(220000.0)  # price for house3
```

### Step 2: Train the Model

```mojo
# Fit linear regression with intercept
var (coefficients, intercept) = fit_linear_regression(X, y, True)

print("Model coefficients:", end=" ")
print_list(coefficients)
print("Intercept:", intercept)
```

### Step 3: Make Predictions

```mojo
# New houses to predict
var X_test: List[List[Float64]] = []

var new_house1: List[Float64] = []
new_house1.append(1400.0)  # size
new_house1.append(3.5)     # rooms
X_test.append(new_house1.copy())

var new_house2: List[Float64] = []
new_house2.append(2000.0)
new_house2.append(5.0)
X_test.append(new_house2.copy())

# Make predictions
var predictions = predict_linear_regression(X_test, coefficients, intercept)

print("Predictions:")
for i in range(len(predictions)):
    print("House", i + 1, ": $", predictions[i])
```

### Step 4: Evaluate the Model

```mojo
# Calculate R² score on training data
var score = score_linear_regression(X, y, coefficients, intercept)
print("R² Score:", score)
```

## Advanced Usage

### Without Intercept

```mojo
# Fit model without intercept term
var (coef_no_intercept, _) = fit_linear_regression(X, y, False)
```

### Matrix Operations

```mojo
# Add intercept column manually
var X_with_intercept = add_ones_column(X)

# Perform matrix operations
var X_transposed = transpose(X)
var XtX = matmul(X_transposed, X)
```

## Complete Example

Here's a complete working example:

```mojo
from builtin import *
from sklearn_mojo import fit_linear_regression, predict_linear_regression, score_linear_regression, print_list

fn main():
    print("=== House Price Prediction Example ===")
    
    # Sample data: size (sq ft), rooms -> price
    var X: List[List[Float64]] = []
    
    var data: List[Float64] = []
    data.append(1200.0)
    data.append(3.0)
    X.append(data.copy())
    
    data = []
    data.append(1600.0)
    data.append(4.0)
    X.append(data.copy())
    
    data = []
    data.append(800.0)
    data.append(2.0)
    X.append(data.copy())
    
    var y: List[Float64] = []
    y.append(350000.0)
    y.append(480000.0)
    y.append(220000.0)
    
    # Train model
    var (coef, intercept) = fit_linear_regression(X, y, True)
    
    print("Model trained!")
    print("Coefficients:", end=" ")
    print_list(coef)
    print("Intercept:", intercept)
    
    # Test predictions
    var X_test: List[List[Float64]] = []
    var test_data: List[Float64] = []
    test_data.append(1400.0)
    test_data.append(3.5)
    X_test.append(test_data.copy())
    
    var predictions = predict_linear_regression(X_test, coef, intercept)
    
    print("Prediction for 1400 sq ft, 3.5 rooms:", predictions[0])
    
    # Evaluate model
    var score = score_linear_regression(X, y, coef, intercept)
    print("R² Score:", score)
```

## Performance Tips

### 1. Data Preprocessing

```mojo
# Always normalize your data for better numerical stability
# This is especially important for large datasets

# Example normalization function
fn normalize_data(X: List[List[Float64]]) -> List[List[Float64]]:
    var normalized: List[List[Float64]] = []
    # Implementation depends on your specific needs
    return normalized.copy()
```

### 2. Memory Efficiency

```mojo
# Use .copy() sparingly - only when necessary
var result = some_function(X)  # This copies the result
# Don't do: result = X.copy().copy()  # Unnecessary copying
```

### 3. Batch Processing

```mojo
# For large datasets, process in batches
fn batch_predict(X_batches: List[List[List[Float64]]], coef: List[Float64], intercept: Float64) -> List[List[Float64]]:
    var all_predictions: List[List[Float64]] = []
    
    for batch in X_batches:
        var batch_pred = predict_linear_regression(batch, coef, intercept)
        all_predictions.append(batch_pred.copy())
    
    return all_predictions.copy()
```

## Common Use Cases

### 1. Sales Prediction

```mojo
# Features: advertising spend, season, region
# Target: sales revenue
var X: List[List[Float64]] = []
var y: List[Float64] = []

# ... populate with your sales data ...

var (coef, intercept) = fit_linear_regression(X, y, True)
var forecast = predict_linear_regression(future_X, coef, intercept)
```

### 2. Quality Control

```mojo
# Features: measurements from manufacturing process
# Target: quality score
var X: List[List[Float64]] = []
var y: List[Float64] = []

var (coef, intercept) = fit_linear_regression(X, y, True)
var quality_prediction = predict_linear_regression(new_measurements, coef, intercept)
```

### 3. Financial Analysis

```mojo
# Features: economic indicators
# Target: stock price or market index
var X: List[List[Float64]] = []
var y: List[Float64] = []

var (coef, intercept) = fit_linear_regression(X, y, True)
var market_prediction = predict_linear_regression(current_indicators, coef, intercept)
```

## Error Handling

The package handles common numerical issues:

```mojo
# Matrix dimensions are validated
# Singular matrix detection and regularization
# Numerical stability improvements

# Check if results are valid
var (coef, intercept) = fit_linear_regression(X, y, True)
if len(coef) > 0:
    # Model was successfully fitted
    var predictions = predict_linear_regression(X_test, coef, intercept)
else:
    print("Model fitting failed - check your data")
```

## Next Steps

- Read the [API Reference](api_reference.md) for detailed function documentation
- Explore the source code to understand the implementation
- Experiment with different datasets and parameters
- Consider extending the package with additional algorithms