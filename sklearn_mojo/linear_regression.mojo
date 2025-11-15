from builtin import *
from .matrix_utils import add_ones_column, matmul, transpose, solve_system

fn fit_linear_regression(X: List[List[Float64]], y: List[Float64], fit_intercept: Bool = True) -> (List[Float64], Float64):
    var X_prepared: List[List[Float64]]
    if fit_intercept:
        X_prepared = add_ones_column(X.copy())
    else:
        X_prepared = X.copy()
    
    var y_matrix: List[List[Float64]] = []
    for i in range(len(y)):
        var row: List[Float64] = []
        row.append(y[i])
        y_matrix.append(row.copy())
    
    var Xt = transpose(X_prepared.copy())
    var XtX = matmul(Xt.copy(), X_prepared.copy())
    
    var n = len(XtX)
    for i in range(n):
        XtX[i][i] += 1e-8
    
    var Xty = matmul(Xt.copy(), y_matrix.copy())
    
    var b: List[Float64] = []
    for i in range(len(Xty)):
        b.append(Xty[i][0])
    
    var weights = solve_system(XtX.copy(), b.copy())
    
    var coef: List[Float64] = []
    var intercept: Float64 = 0.0
    
    if fit_intercept:
        intercept = weights[0]
        for i in range(1, len(weights)):
            coef.append(weights[i])
    else:
        intercept = 0.0
        for i in range(len(weights)):
            coef.append(weights[i])
    
    return (coef.copy(), intercept)

fn predict_linear_regression(X: List[List[Float64]], coef: List[Float64], intercept: Float64) -> List[Float64]:
    var y_pred: List[Float64] = []
    for i in range(len(X)):
        var prediction = intercept
        for j in range(len(X[i])):
            prediction += X[i][j] * coef[j]
        y_pred.append(prediction)
    return y_pred.copy()

fn score_linear_regression(X: List[List[Float64]], y: List[Float64], coef: List[Float64], intercept: Float64) -> Float64:
    var y_pred = predict_linear_regression(X.copy(), coef, intercept)
    var y_mean: Float64 = 0.0
    for i in range(len(y)):
        y_mean += y[i]
    y_mean /= Float64(len(y))
    
    var ss_res: Float64 = 0.0
    var ss_tot: Float64 = 0.0
    for i in range(len(y)):
        var diff = y[i] - y_pred[i]
        ss_res += diff * diff
        var diff2 = y[i] - y_mean
        ss_tot += diff2 * diff2
    
    if ss_tot == 0.0:
        if ss_res == 0.0:
            return 1.0
        else:
            return 0.0
    return 1.0 - (ss_res / ss_tot)