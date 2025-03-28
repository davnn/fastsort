from typing import overload

import numpy as np
import numpy.typing as npt

def sortf64(arr: npt.NDArray[np.float64], axis: int | None = None) -> npt.NDArray[np.float64]: ...
def sortf32(arr: npt.NDArray[np.float32], axis: int | None = None) -> npt.NDArray[np.float32]: ...
def sortu64(arr: npt.NDArray[np.uint64], axis: int | None = None) -> npt.NDArray[np.uint64]: ...
def sortu32(arr: npt.NDArray[np.uint32], axis: int | None = None) -> npt.NDArray[np.uint32]: ...
def sortu16(arr: npt.NDArray[np.uint16], axis: int | None = None) -> npt.NDArray[np.uint16]: ...
def sorti64(arr: npt.NDArray[np.int64], axis: int | None = None) -> npt.NDArray[np.int64]: ...
def sorti32(arr: npt.NDArray[np.int32], axis: int | None = None) -> npt.NDArray[np.int32]: ...
def sorti16(arr: npt.NDArray[np.int16], axis: int | None = None) -> npt.NDArray[np.int16]: ...
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
