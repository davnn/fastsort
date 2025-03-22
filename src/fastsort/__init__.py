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
    # Check the dtype of the array and dispatch accordingly
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
    if arr.ndim == 0:
        msg = "Found ndim=0, argsort requires input that is at least one dimensional."
        raise ValueError(msg)

    return _argsort(arr, axis)
