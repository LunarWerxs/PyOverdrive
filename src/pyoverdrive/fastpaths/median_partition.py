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

This targets fixed wrapper overhead on small arrays. The bounded size
window avoids extrapolating that saving into large partitions, where
partition work dominates and bypassing the wrapper matters less.

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

Historical calibration ratios are omitted because the NumPy version was not
recorded. See docs/research/2026-09-19-burndown.md for current measured
evidence and its version, hardware and load qualifications.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath

_F64 = np.dtype(np.float64)
SIZE_MIN, SIZE_CAP = 10, 5_001  # bounded wrapper-overhead regime


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
