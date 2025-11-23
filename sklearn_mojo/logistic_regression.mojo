from builtin import *
import math
from .matrix_utils import add_ones_column, matmul, transpose, solve_system, sigmoid, sigmoid_derivative, log_loss

fn fit_logistic_regression(X: List[List[Float64]], y: List[Float64], fit_intercept: Bool = True, 
                         learning_rate: Float64 = 0.01, max_iterations: Int = 1000, 
                         tolerance: Float64 = 1e-6) -> (List[Float64], Float64):
    var X_prepared: List[List[Float64]]
    if fit_intercept:
        X_prepared = add_ones_column(X.copy())
    else:
        X_prepared = X.copy()
    
    var n_samples = len(X_prepared)
    var n_features = len(X_prepared[0])
    
    var weights: List[Float64] = []
    for _ in range(n_features):
        weights.append(0.0)
    
    var previous_cost: Float64 = 0.0
    
    for _ in range(max_iterations):
        var predictions: List[Float64] = []
        for i in range(n_samples):
            var linear_pred = 0.0
            for j in range(n_features):
                linear_pred += X_prepared[i][j] * weights[j]
            predictions.append(sigmoid(linear_pred))
        
        var cost = log_loss(y.copy(), predictions.copy())
        
        if abs(previous_cost - cost) < tolerance:
            break
        
        previous_cost = cost
        
        var gradients: List[Float64] = []
        for _ in range(n_features):
            gradients.append(0.0)
        
        for i in range(n_samples):
            var error = predictions[i] - y[i]
            for j in range(n_features):
                gradients[j] += error * X_prepared[i][j]
        
        for i in range(n_features):
            gradients[i] /= Float64(n_samples)
            weights[i] -= learning_rate * gradients[i]
    
    var coef: List[Float64] = []
    var _intercept: Float64 = 0.0
    
    if fit_intercept:
        intercept = weights[0]
        for i in range(1, len(weights)):
            coef.append(weights[i])
    else:
        intercept = 0.0
        for i in range(len(weights)):
            coef.append(weights[i])
    
    return (coef.copy(), intercept)

fn predict_logistic_regression(X: List[List[Float64]], coef: List[Float64], 
                             intercept: Float64, probability_threshold: Float64 = 0.5) -> List[Float64]:
    var predictions: List[Float64] = []
    for i in range(len(X)):
        var linear_pred = intercept
        for j in range(len(X[i])):
            linear_pred += X[i][j] * coef[j]
        var prob = sigmoid(linear_pred)
        var pred = 0.0
        if prob >= probability_threshold:
            pred = 1.0
        predictions.append(pred)
    return predictions.copy()

fn predict_proba_logistic_regression(X: List[List[Float64]], coef: List[Float64], 
                                   intercept: Float64) -> List[Float64]:
    var probabilities: List[Float64] = []
    for i in range(len(X)):
        var linear_pred = intercept
        for j in range(len(X[i])):
            linear_pred += X[i][j] * coef[j]
        probabilities.append(sigmoid(linear_pred))
    return probabilities.copy()

fn score_logistic_regression(X: List[List[Float64]], y: List[Float64], coef: List[Float64], 
                           intercept: Float64) -> Float64:
    var y_pred = predict_logistic_regression(X.copy(), coef, intercept)
    
    var correct_predictions: Int = 0
    for i in range(len(y)):
        if abs(y[i] - y_pred[i]) < 1e-10:
            correct_predictions += 1
    
    return Float64(correct_predictions) / Float64(len(y))

fn log_loss_score_logistic_regression(X: List[List[Float64]], y: List[Float64], coef: List[Float64], 
                                    intercept: Float64) -> Float64:
    var y_proba = predict_proba_logistic_regression(X.copy(), coef, intercept)
    return log_loss(y.copy(), y_proba.copy())