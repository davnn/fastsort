"""Benchmark fastsort against different numpy sort algorithms.
From https://github.com/liwt31/numpy-sort-benchmark/tree/master.
"""

import argparse
import datetime
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

from fastsort import argsort, sort
from helpers.log_config import logger

AREA_NUM: int = 10
BUBBLE_SIZE: int = 10
BenchmarkDataT = tuple[npt.NDArray[Any], npt.NDArray[Any]]
BenchmarkFnT = Callable[[int], BenchmarkDataT]


class BenchmarkReport(TypedDict):
    name: str
    kind: str
    time_mean: float
    time_std: float


class BenchmarkResult(TypedDict):
    seed: int
    loops: int
    size: int
    flatten: bool
    contiguous: bool
    argsort: bool
    report: list[BenchmarkReport]


class BenchSuite:
    """The benchmarking suite runs benchmarks over a number of registered benchmark functions.

    A benchmark function can be registered using ``__call__`` of the instance and should be a
    function of type ``BenchmarkFnT``. The rest of the benchmark configuration happens directly
    in the run ``run`` method.

    By default, the benchmark results are logged to stdout, but can be configured to be saved
    as json reports as well (for later visualization).
    """

    sort_kinds: tuple[str, ...] = ("fast", "heapsort", "quicksort", "stable")

    def __init__(self) -> None:
        self.funcs: dict[str, BenchmarkFnT] = {}

    def __call__(self, func: BenchmarkFnT) -> BenchmarkFnT:
        self.funcs[func.__name__] = func
        return func

    def run(
        self,
        seed: int,
        loops: int,
        size: int,
        *,
        flatten: bool,
        contiguous: bool,
        use_argsort: bool,
        write_report: bool,
        report_folder: str,
    ) -> None:
        np.random.seed(seed)
        logger.info(f"Array size: {size}. Loop num: {loops}")
        sort_fn = argsort if use_argsort else sort
        sort_fn_np = np.argsort if use_argsort else np.sort
        report: list[BenchmarkReport] = []

        for name, func in self.funcs.items():
            base_time: float | None = None
            logger.info(f"Testing {name} array:")

            for kind in self.sort_kinds:
                times: list[float] = []

                for _ in range(loops):
                    arr, _ = func(size * size)
                    arr = arr if flatten else arr.reshape(size, size)
                    arr = arr if contiguous else arr.transpose().copy(order="C")
                    axis: int = -1 if contiguous else 0  # 0 and -1 are the same for vec (flatten)

                    time1: float = time.time()
                    if kind == "fast":
                        sort_fn(arr, axis=axis)
                    else:
                        sort_fn_np(arr, kind=kind, axis=axis)  # type: ignore[reportCallIssue]
                    time2: float = time.time()

                    times.append(time2 - time1)

                times_ms = np.array(times, dtype=np.double) * 1e3  # from s to ms
                mean, std = times_ms.mean(), times_ms.std()
                base_time = mean if base_time is None else base_time
                report.append({"name": name, "kind": kind, "time_mean": mean, "time_std": std})

                logger.info(f"    {kind}: {mean:.3f}±{std:.3f} us per loop. Relative: {mean/base_time*100:.0f}%")

            logger.info("--- run complete ---")

        if write_report:
            timestamp = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
            benchmark_result: BenchmarkResult = {
                "seed": seed,
                "loops": loops,
                "size": size,
                "flatten": flatten,
                "contiguous": contiguous,
                "argsort": use_argsort,
                "report": report,
            }
            script_dir = Path(__file__).resolve().parent / report_folder
            script_dir.mkdir(exist_ok=True)
            with (script_dir / f"{timestamp}-report.json").open("w") as f:
                json.dump(benchmark_result, f, indent=4)


# Instantiate the benchmarking suite and add benchmarking functions to the suite
bench_suite = BenchSuite()


@bench_suite
def random(size: int) -> BenchmarkDataT:
    a = np.arange(size)
    np.random.shuffle(a)
    return a, np.arange(size)


@bench_suite
def ordered(size: int) -> BenchmarkDataT:
    return np.arange(size), np.arange(size)


@bench_suite
def reversed(size: int) -> BenchmarkDataT:
    return np.arange(size - 1, -1, -1), np.arange(size)


@bench_suite
def same_elem(size: int) -> BenchmarkDataT:
    return np.ones(size), np.ones(size)


@bench_suite
def sorted_block_size_10(size: int) -> BenchmarkDataT:
    return sorted_block(size, 10)


@bench_suite
def sorted_block_size_100(size: int) -> BenchmarkDataT:
    return sorted_block(size, 100)


@bench_suite
def sorted_block_size_1000(size: int) -> BenchmarkDataT:
    return sorted_block(size, 1000)


@bench_suite
def swapped_pair_50_percent(size: int) -> BenchmarkDataT:
    return swapped_pair(size, 0.5)


@bench_suite
def swapped_pair_10_percent(size: int) -> BenchmarkDataT:
    return swapped_pair(size, 0.1)


@bench_suite
def swapped_pair_1_percent(size: int) -> BenchmarkDataT:
    return swapped_pair(size, 0.01)


@bench_suite
def random_unsorted_area_50_percent(size: int) -> BenchmarkDataT:
    return random_unsorted_area(size, 0.5, AREA_NUM)


@bench_suite
def random_unsorted_area_10_percent(size: int) -> BenchmarkDataT:
    return random_unsorted_area(size, 0.1, AREA_NUM)


@bench_suite
def random_unsorted_area_1_percent(size: int) -> BenchmarkDataT:
    return random_unsorted_area(size, 0.01, AREA_NUM)


@bench_suite
def random_bubble_1_fold(size: int) -> BenchmarkDataT:
    return random_unsorted_area(size, 1, size // BUBBLE_SIZE)


@bench_suite
def random_bubble_5_fold(size: int) -> BenchmarkDataT:
    return random_unsorted_area(size, 5, size // BUBBLE_SIZE)


@bench_suite
def random_bubble_10_fold(size: int) -> BenchmarkDataT:
    return random_unsorted_area(size, 10, size // BUBBLE_SIZE)


def sorted_block(size: int, block_size: int) -> BenchmarkDataT:
    a = np.arange(size)
    b: list[int] = []
    if size < block_size:
        return a, a
    block_num: int = size // block_size
    for i in range(block_num):
        b.extend(a[i::block_num])
    return np.array(b), a


def swapped_pair(size: int, swap_frac: float) -> BenchmarkDataT:
    a = np.arange(size)
    b = a.copy()
    for _ in range(int(size * swap_frac)):
        x, y = np.random.randint(0, size, 2)
        b[x], b[y] = b[y], b[x]
    return b, a


def random_unsorted_area(size: int, frac: float, area_num: int) -> BenchmarkDataT:
    area_num = int(area_num)
    a = np.arange(size)
    b = a.copy()
    unsorted_len = int(size * frac / area_num)
    for _ in range(area_num):
        start = np.random.randint(size - unsorted_len)
        end = start + unsorted_len
        np.random.shuffle(b[start:end])
    return b, a


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--contiguous", action="store_true")
    parser.add_argument("--argsort", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--folder", type=str, default="reports")
    args = parser.parse_args()

    bench_suite.run(
        seed=args.random_seed,
        loops=args.loops,
        size=args.size,
        flatten=args.flatten,
        contiguous=args.contiguous,
        use_argsort=args.argsort,
        write_report=args.report,
        report_folder=args.folder,
    )
