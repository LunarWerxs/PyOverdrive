"""Fast path: numpy.median on small 1-D float64 arrays via partition.

Provenance (OPP-000037): numpy/numpy#18298 - np.median is written in
Python and piles up overhead (seberg's own diagnosis in-thread) that
dwarfs the actual partition work on small arrays. The honest candidate
replicates median's NaN semantics exactly: stock ALSO partitions the
last element as its NaN check (seberg's correction), so the route
partitions (k, n-1) for odd n / (k-1, k, n-1) for even n, checks
isnan on the partitioned tail, and computes the even-size mean with
stock's own 0.5 * (lo + hi) arithmetic - which made every battery cell
bit-identical, NaN-salted included.

Measured (OPP-000037 + BATCH5-CAL batteries, fp 9bbe7063c555, idle box,
0-1% load): 3.4x at n=11, 3.0x at 10, 2.9-3.2x at 100/101, 2.2-2.4x at
1000/1001, 2.03-2.09x at 2000/2001, 1.74x at 3001, 1.47-1.62x at
5000/5001; 10_000 drops to 1.15-1.26x (below min-win), 100k+ is a wash
- hence the size cap. Overhead-class win, exactly like roll_concat_1d.

Correctness contract:
- Applies only to median(a) where a is a plain 1-D float64 ndarray,
  axis absent or None, no other kwargs, 10 <= size <= SIZE_CAP.
  n-D input, other dtypes, axis given, out/overwrite_input/keepdims,
  and sizes outside the measured band stay on stock.
- Bit-identical including NaN propagation (any NaN -> nan result,
  detected exactly as stock detects it). Whether stock emits a warning
  on NaN input: it does not (verified by the battery running warnings-
  clean); the differential suite pins it.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=median_partition or
pyoverdrive.disable_path("median_partition").
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath

_F64 = np.dtype(np.float64)
SIZE_MIN, SIZE_CAP = 10, 5_001  # measured band; 10_000 measured below min-win


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 1:
        return False
    if set(kwargs) - {"axis"}:
        return False
    if kwargs.get("axis", None) is not None:
        return False
    a = args[0]
    if type(a) is not np.ndarray or a.dtype != _F64 or a.ndim != 1:
        return False
    return SIZE_MIN <= a.size <= SIZE_CAP


def _run(a, axis=None):
    n = a.size
    k = n // 2
    if n % 2:
        p = np.partition(a, (k, n - 1))
        if np.isnan(p[-1]):
            return np.float64(np.nan)
        return p[k]
    p = np.partition(a, (k - 1, k, n - 1))
    if np.isnan(p[-1]):
        return np.float64(np.nan)
    return np.float64(0.5 * (p[k - 1] + p[k]))


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="median_partition",
            op="numpy.median",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000037",
                "source": "https://github.com/numpy/numpy/issues/18298",
                "license": "the thread's own partition route; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
