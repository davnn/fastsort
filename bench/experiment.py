"""
Run multiple benchmarks as an experiment for later evaluation.
"""

import argparse
from benchmark import bench_suite

sizes = [100, 250, 500, 1000]
flatten = [True, False]
contiguous = [True, False]
argsort = [True, False]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="reports")
    args = parser.parse_args()

    for s in sizes:
        for f in flatten:
            for a in argsort:
                for c in contiguous:
                     # There is no difference between contiguous=True and contiguous=False
                     # for one-dimensional arrays, therefore skip the True / True case.
                    if c and f:
                        continue

                    bench_suite.run(
                        seed=42,
                        loops=10,
                        size=s,
                        flatten=f,
                        contiguous=c,
                        use_argsort=a,
                        write_report=True,
                        report_folder=args.folder
                    )
