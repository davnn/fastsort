from functools import partial

import numpy as np
import pytest

from fastsort import sort, argsort

rng = np.random.default_rng(seed=0)

dtypes = [
    np.uint16,
    np.uint32,
    np.uint64,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]
shapes = [
    (100,),
    (100, 100),
    (100, 100, 100),
    (100, 10, 1),
    (1, 10, 100),
    (100, 1, 10),
    (10, 1),
    (1, 10),
    (1,),
]
random = [
    partial(rng.choice, a=100*100*100, replace=False),
]
test_data = [
    rand(size=size).astype(dtype)
    for rand in random
    for size in shapes
    for dtype in dtypes
]
test_data = [ # extend the test data with all possible dimension variants
    (data, dim) for data in test_data for dim in (None, -1, *range(data.ndim))
]

@pytest.mark.parametrize("data, axis", test_data)
def test_sort_shape_and_type(data, axis) -> None:
    sort_data = sort(data, axis=axis)
    assert sort_data.dtype == data.dtype

    if axis is not None:
        assert sort_data.shape == data.shape
    else:
        assert sort_data.shape == (data.size,)

@pytest.mark.parametrize("data, axis", test_data)
def test_argsort_shape_and_type(data, axis) -> None:
    sort_data = argsort(data, axis=axis)
    assert sort_data.dtype == np.int64

    if axis is not None:
        assert sort_data.shape == data.shape
    else:
        assert sort_data.shape == (data.size,)

@pytest.mark.parametrize("data, axis", test_data)
def test_sort_increasing(data, axis) -> None:
    sort_data = sort(data, axis=axis)
    sort_diff = np.diff(sort_data, axis=axis if axis is not None else 0)
    assert np.all(sort_diff >= 0.0)

@pytest.mark.parametrize("data, axis", test_data)
def test_argsort_increasing(data, axis) -> None:
    sort_idx = argsort(data, axis=axis)
    sort_data = (np.take_along_axis(data, sort_idx, axis=axis) 
        if axis is not None else data.take(sort_idx))
    sort_diff = np.diff(sort_data, axis=axis if axis is not None else 0)
    assert np.all(sort_diff >= 0.0)
