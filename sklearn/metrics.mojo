# metrics.mojo
from base import Labels

fn accuracy_score(y_true: Labels, y_pred: Labels) -> F64:
    if y_true.shape != y_pred.shape:
        panic("Shape mismatch")
    let correct = (y_true == y_pred).sum().to_i64()
    return correct.to_f64() / y_true.size().to_f64()
