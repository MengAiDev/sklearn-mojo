# model_selection.mojo
from base import Array, Labels
from random import seed, rand

fn train_test_split(
    X: Array, y: Labels, test_size: F64 = 0.25, random_state: I64 = 0
) -> (Array, Array, Labels, Labels):
    if test_size <= 0 or test_size >= 1:
        panic("test_size must be in (0, 1)")

    let n = X.shape[0].to_i64()
    let n_test = (n * test_size).to_i64()
    let n_train = n - n_test

    seed(random_state)
    var indices = Tensor[I64].arange(n)

    # Fisher-Yates shuffle
    for i in range(n - 1, 0, -1):
        let j = (rand() % (i + 1)).to_i64()
        let tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp

    let train_idx = indices.slice[0:n_train]
    let test_idx = indices.slice[n_train:]

    return (
        X.index_select(0, train_idx),
        X.index_select(0, test_idx),
        y.index_select(0, train_idx),
        y.index_select(0, test_idx)
    )
