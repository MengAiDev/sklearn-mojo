# linear_model.mojo
from base import Array, Labels
from utils import apply_sigmoid

struct LogisticRegression:
    var weights: Array
    var intercept: F64 = 0.0
    var max_iter: I64
    var learning_rate: F64
    var C: F64
    var is_fitted: Bool = False

    fn __init__(inout self, max_iter: I64 = 100, learning_rate: F64 = 0.01, C: F64 = 1.0):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.C = C

    fn fit(inout self, X: Array, y: Labels):
        let n_samples = X.shape[0].to_i64()
        let n_features = X.shape[1].to_i64()
        let y_f64 = y.to_f64()

        self.weights = Tensor[F64].zeros([n_features])
        self.intercept = 0.0

        for _ in range(self.max_iter):
            let linear = X @ self.weights + self.intercept
            let y_pred = apply_sigmoid(linear)

            let errors = y_pred - y_f64
            let dw = (X.T @ errors) / n_samples.to_f64() + self.weights / (self.C * n_samples.to_f64())
            let db = errors.sum() / n_samples.to_f64()

            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

        self.is_fitted = True

    fn predict(self, X: Array) -> Labels:
        if not self.is_fitted:
            panic("Model not fitted")
        let linear = X @ self.weights + self.intercept
        let probs = apply_sigmoid(linear)
        return probs.map(fn(p: F64) -> I64: return 1 if p >= 0.5 else 0).to_i64()
