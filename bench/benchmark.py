"""
Benchmark fastsort against different numpy sort algorithms.
From https://github.com/liwt31/numpy-sort-benchmark/tree/master
"""

import time
import json
import argparse
import datetime
from pathlib import Path

import numpy as np
from fastsort import sort, argsort

AREA_NUM = 10
BUBBLE_SIZE = 10

class BenchSuite:

    sort_kinds = ["quicksort", "heapsort", "stable", "fast"]

    def __init__(self):
        self.funcs = dict()

    def __call__(self, func):
        self.funcs[func.__name__] = func
        return func

    def run(
            self,
            seed,
            loops,
            size,
            flatten,
            contiguous,
            use_argsort,
            write_report,
            report_folder,
        ):
        np.random.seed(seed)
        print(f"Array size: {size}. Loop num: {loops}")
        sort_fn = argsort if use_argsort else sort
        sort_fn_np = np.argsort if use_argsort else np.sort
        report = []
        for name, func in self.funcs.items():
            base_time = None
            print(f"Testing {name} array:")
            for kind in self.sort_kinds:
                times = []
                for i in range(loops):
                    arr, answer = func(size * size)
                    arr = arr if flatten else arr.reshape(size, size)
                    arr = arr if contiguous else arr.transpose().copy(order="C")
                    axis = -1 if contiguous else 0 # 0 and -1 is the same for vec (flatten)
                    time1 = time.time()
                    sort_fn(arr, axis=axis) if kind == "fast" else sort_fn_np(arr, kind=kind, axis=axis)
                    time2 = time.time()
                    times.append(time2 - time1)
                times = np.array(times) * 1e3 # from s to ms
                mean, std = times.mean(), times.std()
                base_time = mean if base_time is None else base_time
                report.append({"name": name, "kind": kind, "time_mean": mean, "time_std": std})
                print(f"    {kind}: {mean:.3f}±{std:.3f} us per loop. Relative: {mean/base_time*100:.0f}%")
            print()

        if write_report:
            timestamp = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
            report = {
                "seed": seed,
                "loops": loops,
                "size": size,
                "flatten": flatten,
                "contiguous": contiguous,
                "argsort": use_argsort,
                "report": report
            }
            script_dir = Path(__file__).resolve().parent / report_folder
            script_dir.mkdir(exist_ok=True)
            with open(script_dir / f"{timestamp}-report.json", "w") as f:
                json.dump(report, f, indent=4)

bench_suite = BenchSuite()

@bench_suite
def random(size):
    a = np.arange(size)
    np.random.shuffle(a)
    return a, np.arange(size)


@bench_suite
def ordered(size):
    return np.arange(size), np.arange(size)


@bench_suite
def reversed(size):
    return np.arange(size-1, -1, -1), np.arange(size)


@bench_suite
def same_elem(size):
    return np.ones(size), np.ones(size)


def sorted_block(size, block_size):
    a = np.arange(size)
    b = []
    if size < block_size:
        return a, a
    block_num = size // block_size
    for i in range(block_num):
        b.extend(a[i::block_num])
    return np.array(b), a


@bench_suite
def sorted_block_size_10(size):
    return sorted_block(size, 10)


@bench_suite
def sorted_block_size_100(size):
    return sorted_block(size, 100)


@bench_suite
def sorted_block_size_1000(size):
    return sorted_block(size, 1000)


def swapped_pair(size, swap_frac):
    a = np.arange(size)
    b = a.copy()
    for i in range(int(size * swap_frac)):
        x, y = np.random.randint(0, size, 2)
        b[x], b[y] = b[y], b[x]
    return b, a


@bench_suite
def swapped_pair_50_percent(size):
    return swapped_pair(size, 0.5)


@bench_suite
def swapped_pair_10_percent(size):
    return swapped_pair(size, 0.1)


@bench_suite
def swapped_pair_1_percent(size):
    return swapped_pair(size, 0.01)


def random_unsorted_area(size, frac, area_num):
    area_num = int(area_num)
    a = np.arange(size)
    b = a.copy()
    unsorted_len = int(size * frac / area_num)
    for i in range(area_num):
        start = np.random.randint(size-unsorted_len)
        end = start + unsorted_len
        np.random.shuffle(b[start:end])
    return b, a

@bench_suite
def random_unsorted_area_50_percent(size):
    return random_unsorted_area(size, 0.5, AREA_NUM)


@bench_suite
def random_unsorted_area_10_percent(size):
    return random_unsorted_area(size, 0.1, AREA_NUM)


@bench_suite
def random_unsorted_area_1_percent(size):
    return random_unsorted_area(size, 0.01, AREA_NUM)

@bench_suite
def random_bubble_1_fold(size):
    return random_unsorted_area(size, 1, size / BUBBLE_SIZE)


@bench_suite
def random_bubble_5_fold(size):
    return random_unsorted_area(size, 5, size / BUBBLE_SIZE)


@bench_suite
def random_bubble_10_fold(size):
    return random_unsorted_area(size, 10, size / BUBBLE_SIZE)


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
