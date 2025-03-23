"""fastsort: A high-performance sorting library for NumPy arrays.

This package provides an optimized sorting library for NumPy arrays, providing high-performance implementations
of `sort` and `argsort` for various numerical data types. It leverages specialized low-level Rust-based sort
implementations to achieve state-of-the-art sorting performance for a wide variety of data distributions.

Functions:
    - sort(arr: np.ndarray, axis: int | None = None) -> np.ndarray
        Efficiently sorts the input NumPy array along the specified axis.

    - argsort(arr: np.ndarray, axis: int | None = None) -> np.ndarray
        Returns the indices that would sort the input NumPy array along the specified axis.

Features:
    - Optimized sorting for float32, float64, int16, int32, int64, uint16, uint32, and uint64 data types.
    - Supports multi-dimensional sorting along a specified axis.
    - Provides a high-performance alternative to NumPy's sorting functions.

Exceptions:
    - Raises `TypeError` for unsupported input types.
    - Raises `ValueError` if the input array has zero dimensions.

Example Usage:
    >>> import numpy as np
    >>> from fastsort import sort, argsort
    >>> arr = np.array([3, 1, 2], dtype=np.int32)
    >>> sorted_arr = sort(arr)
    >>> sorted_arr
    array([1, 2, 3], dtype=int32)

    >>> indices = argsort(arr)
    >>> indices
    array([1, 2, 0], dtype=int32)

The package is designed for users who require efficient sorting operations for large-scale numerical data.
"""

from typing import Any, overload

import numpy as np
import numpy.typing as npt

from fastsort._core import argsort as _argsort
from fastsort._core import sortf32 as _sf32
from fastsort._core import sortf64 as _sf64
from fastsort._core import sorti16 as _si16
from fastsort._core import sorti32 as _si32
from fastsort._core import sorti64 as _si64
from fastsort._core import sortu16 as _su16
from fastsort._core import sortu32 as _su32
from fastsort._core import sortu64 as _su64

__all__ = [
    "argsort",
    "sort",
]


# Overload function signatures for sort based on the input array type
@overload
def sort(arr: npt.NDArray[np.float64], axis: int | None = None) -> npt.NDArray[np.float64]: ...


@overload
def sort(arr: npt.NDArray[np.float32], axis: int | None = None) -> npt.NDArray[np.float32]: ...


@overload
def sort(arr: npt.NDArray[np.uint64], axis: int | None = None) -> npt.NDArray[np.uint64]: ...


@overload
def sort(arr: npt.NDArray[np.uint32], axis: int | None = None) -> npt.NDArray[np.uint32]: ...


@overload
def sort(arr: npt.NDArray[np.uint16], axis: int | None = None) -> npt.NDArray[np.uint16]: ...


@overload
def sort(arr: npt.NDArray[np.int64], axis: int | None = None) -> npt.NDArray[np.int64]: ...


@overload
def sort(arr: npt.NDArray[np.int32], axis: int | None = None) -> npt.NDArray[np.int32]: ...


@overload
def sort(arr: npt.NDArray[np.int16], axis: int | None = None) -> npt.NDArray[np.int16]: ...


def sort(arr: npt.NDArray[Any], axis: int | None = -1):
    """Sorts the input array. If the array has more than one dimension and a specific axis is provided,
    sorting is performed along that axis. Otherwise, sorting is performed on the entire array.

    Args:
        arr: A NumPy array to be sorted.
        axis: The axis along which to sort. If None or not provided, the array is flattened before sorting.

    Returns:
        A NumPy array with the sorted elements.

    Raises:
        TypeError: If the input is an unsupported type.
        ValueError: If the input array has zero dimensions.

    """
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.float64:
            return _sf64(arr, axis)
        if arr.dtype == np.float32:
            return _sf32(arr, axis)
        if arr.dtype == np.int64:
            return _si64(arr, axis)
        if arr.dtype == np.int32:
            return _si32(arr, axis)
        if arr.dtype == np.int16:
            return _si16(arr, axis)
        if arr.dtype == np.uint64:
            return _su64(arr, axis)
        if arr.dtype == np.uint32:
            return _su32(arr, axis)
        if arr.dtype == np.uint16:
            return _su16(arr, axis)
        if arr.ndim == 0:
            msg = "Found ndim=0, argsort requires input that is at least one dimensional."
            raise ValueError(msg)
        msg = f"Found invalid data type for sort, no sort exists for data type {arr.dtype}."
        raise TypeError(msg)
    msg = f"Found invalid type for sort, no sort exists for type {type(arr)}."
    raise TypeError(msg)


# Overload function signatures for argsort based on the input array type
@overload
def argsort(arr: npt.NDArray[np.float64], axis: int | None = None) -> npt.NDArray[np.int64]: ...


@overload
def argsort(arr: npt.NDArray[np.float32], axis: int | None = None) -> npt.NDArray[np.int64]: ...


@overload
def argsort(arr: npt.NDArray[np.int64], axis: int | None = None) -> npt.NDArray[np.int64]: ...


@overload
def argsort(arr: npt.NDArray[np.int32], axis: int | None = None) -> npt.NDArray[np.int64]: ...


@overload
def argsort(arr: npt.NDArray[np.int16], axis: int | None = None) -> npt.NDArray[np.int64]: ...


@overload
def argsort(arr: npt.NDArray[np.uint64], axis: int | None = None) -> npt.NDArray[np.int64]: ...


@overload
def argsort(arr: npt.NDArray[np.uint32], axis: int | None = None) -> npt.NDArray[np.int64]: ...


@overload
def argsort(arr: npt.NDArray[np.uint16], axis: int | None = None) -> npt.NDArray[np.int64]: ...


def argsort(arr: npt.NDArray[Any], axis: int | None = -1) -> npt.NDArray[Any]:
    """Returns the indices that would sort the input array. If the array has more than one dimension
    and a specific axis is provided, sorting is performed along that axis.

    Args:
        arr: A NumPy array to find the sorted indices.
        axis: The axis along which to sort. If None or not provided, the array is flattened before sorting.

    Returns:
        A NumPy array of indices that would sort the input array.

    Raises:
        TypeError: If the input is an unsupported type.
        ValueError: If the input array has zero dimensions.

    """
    if not isinstance(arr, np.ndarray):
        msg = f"Found invalid type for sort, no sort exists for type {type(arr)}."
        raise TypeError(msg)

    if arr.ndim == 0:
        msg = "Found ndim=0, argsort requires input that is at least one dimensional."
        raise ValueError(msg)

    # A TypeError is raised if an unsupported data type is used in ``_argsort``
    return _argsort(arr, axis)
