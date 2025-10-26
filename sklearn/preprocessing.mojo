# preprocessing.mojo
from base import Array
from math import sqrt

struct StandardScaler:
    var mean: Array
    var scale: Array
    var is_fitted: Bool = False

    fn fit(inout self, X: Array):
        self.mean = X.mean(axis=0)
        let var = ((X - self.mean) ** 2).mean(axis=0)
        self.scale = var.map(fn(v: F64) -> F64: return sqrt(v) if v > 1e-8 else 1.0)
        self.is_fitted = True

    fn transform(self, X: Array) -> Array:
        if not self.is_fitted:
            panic("StandardScaler not fitted")
        return (X - self.mean) / self.scale

    fn fit_transform(inout self, X: Array) -> Array:
        self.fit(X)
        return self.transform(X)
