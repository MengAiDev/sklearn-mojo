# utils.mojo
from base import Array, Labels
from math import exp

# sigmoid
fn sigmoid(x: F64) -> F64:
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        let z = exp(x)
        return z / (1.0 + z)

# Apply for the whole Tensor
fn apply_sigmoid(t: Array) -> Array:
    return t.map(sigmoid)
