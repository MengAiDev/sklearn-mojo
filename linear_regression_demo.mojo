from builtin import *
from sklearn_mojo import fit_linear_regression, predict_linear_regression, score_linear_regression, print_list

fn main():
    print("🚀 sklearn-mojo: Pure Mojo Implementation of scikit-learn Linear Regression")
    print("=========================================================================")
    
    var X: List[List[Float64]] = []
    var row1: List[Float64] = []
    row1.append(1.0)
    row1.append(2.0)
    X.append(row1.copy())
    
    var row2: List[Float64] = []
    row2.append(2.0)
    row2.append(3.0)
    X.append(row2.copy())
    
    var row3: List[Float64] = []
    row3.append(3.0)
    row3.append(1.0)
    X.append(row3.copy())
    
    var row4: List[Float64] = []
    row4.append(4.0)
    row4.append(2.0)
    X.append(row4.copy())
    
    var row5: List[Float64] = []
    row5.append(5.0)
    row5.append(3.0)
    X.append(row5.copy())
    
    var y: List[Float64] = []
    y.append(8.1)
    y.append(13.9)
    y.append(9.8)
    y.append(15.2)
    y.append(20.1)
    
    print("Training data:")
    print("  X shape:", len(X), "x", len(X[0]) if len(X) > 0 else 0)
    print("  y length:", len(y))
    
    var (coef, intercept) = fit_linear_regression(X, y, True)
    
    print("\nModel parameters:")
    print("  Coefficients: ", end="")
    print_list(coef)
    print("  Intercept:", intercept)
    
    var y_pred = predict_linear_regression(X, coef, intercept)
    print("\nPredictions vs Actual:")
    print("  Predictions: ", end="")
    print_list(y_pred)
    print("  Actual:      ", end="")
    print_list(y)
    
    var score_val = score_linear_regression(X, y, coef, intercept)
    print("\nModel performance:")
    print("  R² Score:", score_val)
    
    print("\nTesting with new data:")
    var X_test: List[List[Float64]] = []
    var test_row1: List[Float64] = []
    test_row1.append(1.0)
    test_row1.append(1.0)
    X_test.append(test_row1.copy())
    
    var test_row2: List[Float64] = []
    test_row2.append(2.0)
    test_row2.append(2.0)
    X_test.append(test_row2.copy())
    
    var test_row3: List[Float64] = []
    test_row3.append(3.0)
    test_row3.append(3.0)
    X_test.append(test_row3.copy())
    
    var y_test_pred = predict_linear_regression(X_test, coef, intercept)
    print("  Test inputs:")
    for i in range(len(X_test)):
        print("    ", end="")
        print_list(X_test[i])
    print("  Test predictions:")
    for i in range(len(y_test_pred)):
        print("    ", y_test_pred[i])
    
    print("\n✅ Linear Regression implementation completed successfully!")
