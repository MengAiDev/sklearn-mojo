from builtin import *
import math

fn print_list(lst: List[Float64]):
    print("[", end="")
    for i in range(len(lst)):
        print(lst[i], end="")
        if i < len(lst) - 1:
            print(", ", end="")
    print("]")

fn add_ones_column(X: List[List[Float64]]) -> List[List[Float64]]:
    var result: List[List[Float64]] = []
    for i in range(len(X)):
        var new_row: List[Float64] = []
        new_row.append(1.0)
        for j in range(len(X[i])):
            new_row.append(X[i][j])
        result.append(new_row.copy())
    return result.copy()

fn matmul(A: List[List[Float64]], B: List[List[Float64]]) -> List[List[Float64]]:
    var rows_A = len(A)
    var cols_A = len(A[0])
    var rows_B = len(B)
    var cols_B = len(B[0])
    
    if cols_A != rows_B:
        print("Matrix dimensions don't match for multiplication")
        var empty: List[List[Float64]] = []
        return empty.copy()
    
    var result: List[List[Float64]] = []
    for i in range(rows_A):
        var row: List[Float64] = []
        for j in range(cols_B):
            var sum: Float64 = 0.0
            for k in range(cols_A):
                sum += A[i][k] * B[k][j]
            row.append(sum)
        result.append(row.copy())
    return result.copy()

fn transpose(X: List[List[Float64]]) -> List[List[Float64]]:
    if len(X) == 0:
        var empty: List[List[Float64]] = []
        return empty.copy()
    
    var rows = len(X)
    var cols = len(X[0])
    var result: List[List[Float64]] = []
    
    for j in range(cols):
        var row: List[Float64] = []
        for i in range(rows):
            row.append(X[i][j])
        result.append(row.copy())
    return result.copy()

fn solve_system(A: List[List[Float64]], b: List[Float64]) -> List[Float64]:
    var n = len(A)
    var augmented: List[List[Float64]] = []
    
    for i in range(n):
        var row: List[Float64] = []
        for j in range(n):
            row.append(A[i][j])
        row.append(b[i])
        augmented.append(row.copy())
    
    for i in range(n):
        var pivot = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[pivot][i]):
                pivot = k
        
        if pivot != i:
            var temp = augmented[i].copy()
            augmented[i] = augmented[pivot].copy()
            augmented[pivot] = temp.copy()
        
        var pivot_val = augmented[i][i]
        for j in range(len(augmented[i])):
            augmented[i][j] /= pivot_val
        
        for k in range(i + 1, n):
            var factor = augmented[k][i]
            for j in range(len(augmented[i])):
                augmented[k][j] -= factor * augmented[i][j]
    
    var x: List[Float64] = []
    for i in range(n):
        x.append(0.0)
    
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][n]
        for j in range(i + 1, n):
            x[i] -= augmented[i][j] * x[j]
    
    return x.copy()

fn sigmoid(x: Float64) -> Float64:
    if x < -60.0:
        return 0.0
    elif x > 60.0:
        return 1.0
    else:
        return 1.0 / (1.0 + math.exp(-x))

fn sigmoid_derivative(x: Float64) -> Float64:
    var s = sigmoid(x)
    return s * (1.0 - s)

fn log_loss(y_true: List[Float64], y_pred: List[Float64], epsilon: Float64 = 1e-15) -> Float64:
    var loss: Float64 = 0.0
    for i in range(len(y_true)):
        var y_t = y_true[i]
        var y_p = y_pred[i]
        
        y_p = max(y_p, epsilon)
        y_p = min(y_p, 1.0 - epsilon)
        
        if y_t == 1.0:
            loss += -math.log(y_p)
        else:
            loss += -math.log(1.0 - y_p)
    
    return loss / Float64(len(y_true))