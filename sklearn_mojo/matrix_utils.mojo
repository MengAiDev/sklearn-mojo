from builtin import *

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