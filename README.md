# fastsort

An exploration of fast Rust sorting algorithms in Python.

```py
import numpy as np
from fastsort import sort, argsort

arr = np.random.randn(1000, 1000)
sort_idx = argsort(arr, axis=-1)
sort_arr = np.take_along_axis(arr, sort_idx, axis=-1)
assert (sort_arr - sort(arr)).sum() == 0
```

Only two functions are exported: ``sort`` and ``argsort``. Both functions take an n-dimensional NumPy array as input, along with an optional axis. If the axis is None, the flattened array is sorted. The sorting is unstable and utilizes Rust's *sort_unstable* for parallel slices of arrays or *par_sort_unstable* for vectors.

The library is in its early stages, and while further performance optimizations may be possible, it already achieves state-of-the-art performance, particularly on larger arrays, thanks to improved resource utilization.

### How to develop

- Use ``uv sync`` to install dependencies from the lock file.
- Use ``uv lock`` to update the lock file if necessary given the pinned dependencies.
- Use ``uv lock --upgrade`` to upgrade the lock file the latest valid dependencies.
- Use ``uv build && uv pip install --no-deps --force-reinstall  --editable .`` to build the package and install it locally.
- Use ``uv run pytest tests`` to test the local package.

It might happen that the host ``github.com`` is not trusted, in this case use ``uv sync --allow-insecure-host https://github.com`` if you trust ``github.com``.
