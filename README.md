# fastsort

A high-performance sorting library for NumPy arrays.

```py
import numpy as np
from fastsort import sort, argsort

arr = np.array([3, 1, 2], dtype=np.int32)

sorted_arr = sort(arr)
# Output: array([1, 2, 3], dtype=int32)

indices = argsort(arr)
# Output: array([1, 2, 0], dtype=int32)
```

Only two functions are exported: ``sort`` and ``argsort``. Both functions take an n-dimensional NumPy array as input, along with an optional axis. If the axis is None, the flattened array is sorted. The sorting is unstable and utilizes Rust's *sort_unstable* for parallel slices of arrays or *par_sort_unstable* for vectors.

You can install the package using ``pip install fastsort``. The library is in its early stages, and while further performance optimizations may be possible, it already achieves state-of-the-art performance, particularly on larger arrays, thanks to improved resource utilization.

### How to develop

- Use ``uv sync`` to install dependencies from the lock file.
- Use ``uv lock`` to update the lock file if necessary given the pinned dependencies.
- Use ``uv lock --upgrade`` to upgrade the lock file the latest valid dependencies.
- Use ``uv build`` to build the package.
- Use ``uv pip install --editable .`` to install the package.
- Use ``uv run black src tests bench`` for formatting.
- Use ``uv run ruff check --fix`` for linting.
- Use ``uv run pyright`` for typechecking.
- Use ``uv run pytest tests`` to test the local package.

During development its useful to run ``uv build && uv pip install --no-deps --force-reinstall  --editable .``, which builds and re-installs the built package. If you use [task](https://github.com/go-task/task), most of the above tasks are also defined in the ``Taskfile.yml``.

### How to benchmark

- Use ``uv run bench/benchmark.py`` to run an individual benchmark.
- Use ``uv run bench/experiment.py`` to run an experiment of multiple benchmarks.
- Use ``uv run bench/visualize.py`` to plot experiment results.

Have a look at the individual scripts to learn about the configuration options.
