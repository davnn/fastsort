import numpy as np
import numpy.typing as npt
from typing import overload

from fastsort._core import (
    argsort as _argsort,
    sortf64 as _sf64,
    sortf32 as _sf32,
    sortu64 as _su64,
    sortu32 as _su32,
    sortu16 as _su16,
    sorti64 as _si64,
    sorti32 as _si32,
    sorti16 as _si16
)

__all__ = [
    "sort",
    "argsort"
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

def sort(arr: np.ndarray, axis: int | None = -1):
    """
    Sorts the input array. If the array has more than one dimension and a specific axis is provided,
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
        elif arr.dtype == np.float32:
            return _sf32(arr, axis)
        elif arr.dtype == np.int64:
            return _si64(arr, axis)
        elif arr.dtype == np.int32:
            return _si32(arr, axis)
        elif arr.dtype == np.int16:
            return _si16(arr, axis)
        elif arr.dtype == np.uint64:
            return _su64(arr, axis)
        elif arr.dtype == np.uint32:
            return _su32(arr, axis)
        elif arr.dtype == np.uint16:
            return _su16(arr, axis)
        elif arr.ndim == 0:
            msg = "Found ndim=0, argsort requires input that is at least one dimensional."
            raise ValueError(msg)
        else:
            msg = f"Found invalid data type for sort, no sort exists for data type {arr.dtype}."
            raise TypeError(msg)
    else:
        msg = f"Found invalid type for sort, no sort exists for type {type(arr)}."
        raise TypeError(msg)

# Overload function signatures for argsort based on the input array type
@overload
def argsort(arr: npt.NDArray[np.float64], axis: int | None = None) -> npt.NDArray[np.float64]: ...

@overload
def argsort(arr: npt.NDArray[np.float32], axis: int | None = None) -> npt.NDArray[np.float32]: ...

@overload
def argsort(arr: npt.NDArray[np.int64], axis: int | None = None) -> npt.NDArray[np.int64]: ...

@overload
def argsort(arr: npt.NDArray[np.int32], axis: int | None = None) -> npt.NDArray[np.int32]: ...

@overload
def argsort(arr: npt.NDArray[np.int16], axis: int | None = None) -> npt.NDArray[np.int16]: ...

@overload
def argsort(arr: npt.NDArray[np.uint64], axis: int | None = None) -> npt.NDArray[np.uint64]: ...

@overload
def argsort(arr: npt.NDArray[np.uint32], axis: int | None = None) -> npt.NDArray[np.uint32]: ...

@overload
def argsort(arr: npt.NDArray[np.uint16], axis: int | None = None) -> npt.NDArray[np.uint16]: ...

def argsort(arr: np.ndarray, axis: int | None = -1):
    """
    Returns the indices that would sort the input array. If the array has more than one dimension
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
