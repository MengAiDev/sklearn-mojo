from builtin import *
from sklearn_mojo import fit_logistic_regression, predict_logistic_regression, predict_proba_logistic_regression, score_logistic_regression, log_loss_score_logistic_regression, print_list

fn main():
    print("🚀 sklearn-mojo: Pure Mojo Implementation of scikit-learn Logistic Regression")
    print("================================================================================")
    
    var X: List[List[Float64]] = []
    
    var row1: List[Float64] = []
    row1.append(2.0)
    row1.append(1.0)
    X.append(row1.copy())
    
    var row2: List[Float64] = []
    row2.append(3.0)
    row2.append(2.0)
    X.append(row2.copy())
    
    var row3: List[Float64] = []
    row3.append(1.0)
    row3.append(1.0)
    X.append(row3.copy())
    
    var row4: List[Float64] = []
    row4.append(4.0)
    row4.append(3.0)
    X.append(row4.copy())
    
    var row5: List[Float64] = []
    row5.append(5.0)
    row5.append(2.0)
    X.append(row5.copy())
    
    var row6: List[Float64] = []
    row6.append(2.0)
    row6.append(4.0)
    X.append(row6.copy())
    
    var row7: List[Float64] = []
    row7.append(3.0)
    row7.append(1.0)
    X.append(row7.copy())
    
    var row8: List[Float64] = []
    row8.append(1.0)
    row8.append(3.0)
    X.append(row8.copy())
    
    var y: List[Float64] = []
    y.append(0.0)
    y.append(1.0)
    y.append(0.0)
    y.append(1.0)
    y.append(1.0)
    y.append(1.0)
    y.append(1.0)
    y.append(0.0)
    
    print("Training data:")
    print("  X shape:", len(X), "x", len(X[0]) if len(X) > 0 else 0)
    print("  y length:", len(y))
    print("  Features (X):")
    for i in range(len(X)):
        print("    ", end="")
        print_list(X[i])
    print("  Labels (y): ", end="")
    print_list(y)
    
    var (coef, intercept) = fit_logistic_regression(X, y, True, 0.1, 1000, 1e-6)
    
    print("\nModel parameters:")
    print("  Coefficients: ", end="")
    print_list(coef)
    print("  Intercept:", intercept)
    
    var y_pred = predict_logistic_regression(X, coef, intercept)
    var y_proba = predict_proba_logistic_regression(X, coef, intercept)
    
    print("\nPredictions vs Actual:")
    print("  Actual:    ", end="")
    print_list(y)
    print("  Predicted: ", end="")
    print_list(y_pred)
    print("  Probabilities: ", end="")
    print_list(y_proba)
    
    var accuracy = score_logistic_regression(X, y, coef, intercept)
    var log_loss_val = log_loss_score_logistic_regression(X, y, coef, intercept)
    
    print("\nModel performance:")
    print("  Accuracy:", accuracy)
    print("  Log Loss:", log_loss_val)
    
    print("\nConfusion Matrix Analysis:")
    var true_positives = 0
    var false_positives = 0
    var true_negatives = 0
    var false_negatives = 0
    
    for i in range(len(y)):
        if y[i] == 1.0 and y_pred[i] == 1.0:
            true_positives += 1
        elif y[i] == 0.0 and y_pred[i] == 1.0:
            false_positives += 1
        elif y[i] == 0.0 and y_pred[i] == 0.0:
            true_negatives += 1
        elif y[i] == 1.0 and y_pred[i] == 0.0:
            false_negatives += 1
    
    print("  True Positives:", true_positives)
    print("  False Positives:", false_positives)
    print("  True Negatives:", true_negatives)
    print("  False Negatives:", false_negatives)
    
    if (true_positives + false_positives) > 0:
        var precision = Float64(true_positives) / Float64(true_positives + false_positives)
        print("  Precision:", precision)
    
    if (true_positives + false_negatives) > 0:
        var recall = Float64(true_positives) / Float64(true_positives + false_negatives)
        print("  Recall:", recall)
    
    print("\nTesting with new data:")
    var X_test: List[List[Float64]] = []
    var test_row1: List[Float64] = []
    test_row1.append(1.5)
    test_row1.append(1.5)
    X_test.append(test_row1.copy())
    
    var test_row2: List[Float64] = []
    test_row2.append(3.5)
    test_row2.append(2.5)
    X_test.append(test_row2.copy())
    
    var test_row3: List[Float64] = []
    test_row3.append(0.5)
    test_row3.append(0.5)
    X_test.append(test_row3.copy())
    
    var y_test_pred = predict_logistic_regression(X_test, coef, intercept)
    var y_test_proba = predict_proba_logistic_regression(X_test, coef, intercept)
    
    print("  Test inputs:")
    for i in range(len(X_test)):
        print("    ", end="")
        print_list(X_test[i])
    
    print("  Test predictions:")
    for i in range(len(y_test_pred)):
        print("    Prediction:", y_test_pred[i], " (Probability:", y_test_proba[i], ")")
    
    print("\n✅ Logistic Regression implementation completed successfully!")