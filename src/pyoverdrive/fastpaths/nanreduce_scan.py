"""Fast paths: numpy.nanmean / nansum / nanstd / nanvar via one isnan scan.

Provenance (OPP-000041): the nan-aggregation wrappers are Python-level
functions that mask-and-copy even when the input holds no NaN at all,
which is the common defensive-usage case. numpy/numpy#5691 documented
the class in 2015 (nanmax ~3x); modern numpy fixed nanmax/nanmin (they
are NOT patched here: measured 0.61-1.08x, no win) but the four ops
above retain the overhead. The pydata/bottleneck README benchmark table
(numpy 2.4.2) documents the same class.

Route: one O(n) NaN probe (np.isnan(np.min(a)): min propagates NaN).
Clean input takes the plain reduction, which the BATCH6-CAL battery
measured bit-identical to the nan-wrapper on every winning cell.
NaN-present input falls back to stock INSIDE the run (no warning; the
result is stock's by construction), costing the scan: measured 0.96x.

CALIBRATION (fp 9bbe7063c555, idle box, 0% load, numpy 2.5.2,
benchmarks/results/BATCH6-CAL/): nanmean 1.99x at n=100, 2.11x at 1e3,
12.41x at 1e5, 11.1-12.3x 2-D; nansum 1.31x at 1e4 (1.22x at 3e3, so
its floor sits at 1e4), 2.86x at 1e5; nanstd 2.16-2.56x and nanvar
2.17-2.43x from 3e3 up. Floors below are those measured edges.

Correctness contract:
- plain float64 ndarray, non-empty; axis absent, None, or a single int;
  no other kwargs (dtype/out/keepdims/where force stock, as do other
  dtypes: every one is unmeasured or semantics this route must not
  guess). Zero-length reduced axes refuse (stock emits empty-slice
  warnings this route does not replicate).
- clean input: bit-identical to stock (same pairwise reduction
  arithmetic; battery-checked with array_equal on every cell).
- NaN present: stock's own result, via internal fallback.

Comparison mode: bit-identical. Kill switches: nanmean_scan,
nansum_scan, nanstd_scan, nanvar_scan (PYOVERDRIVE_DISABLE or
pyoverdrive.disable_path).
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

_F64 = np.dtype(np.float64)

# smallest measured winning cell per op (BATCH6-CAL, fp 9bbe7063c555)
_FLOORS = {
    "numpy.nanmean": 100,
    "numpy.nansum": 10_000,
    "numpy.nanstd": 3_000,
    "numpy.nanvar": 3_000,
}

_PLAIN = {
    "numpy.nanmean": "numpy.mean",
    "numpy.nansum": "numpy.sum",
    "numpy.nanstd": "numpy.std",
    "numpy.nanvar": "numpy.var",
}


def _normalize(args: tuple, kwargs: dict):
    """Return (a, axis) for an admissible call shape, else None."""
    if len(args) == 1:
        axis = kwargs.get("axis")
        if kwargs and set(kwargs) != {"axis"}:
            return None
    elif len(args) == 2:
        if kwargs:
            return None
        axis = args[1]
    else:
        return None
    return args[0], axis


def _applicable_for(op: str):
    floor = _FLOORS[op]

    def _applicable(args: tuple, kwargs: dict) -> bool:
        norm = _normalize(args, kwargs)
        if norm is None:
            return False
        a, axis = norm
        if type(a) is not np.ndarray or a.dtype != _F64:
            return False
        if a.size < floor:
            return False
        if axis is None:
            return True
        if isinstance(axis, bool) or not isinstance(axis, int):
            return False
        if not -a.ndim <= axis < a.ndim:
            return False
        return a.shape[axis] > 0

    return _applicable


def _run_for(op: str):
    plain_name = _PLAIN[op]

    def _run(a, axis=None):
        if np.isnan(np.min(a)):
            # NaN somewhere: stock's masking is the semantics, use it
            return GEARBOX.stock_fn(op)(a, axis=axis)
        return GEARBOX.stock_fn(plain_name)(a, axis=axis)

    return _run


def register(gearbox) -> None:
    for op, path_name in (
        ("numpy.nanmean", "nanmean_scan"),
        ("numpy.nansum", "nansum_scan"),
        ("numpy.nanstd", "nanstd_scan"),
        ("numpy.nanvar", "nanvar_scan"),
    ):
        gearbox.register(
            FastPath(
                name=path_name,
                op=op,
                applicable=_applicable_for(op),
                run=_run_for(op),
                provenance={
                    "opportunity": "OPP-000041",
                    "source": "https://github.com/numpy/numpy/issues/5691",
                    "license": "isnan scan + plain reduction, standard technique; no third-party code",
                    "comparison_mode": "bit-identical",
                },
            )
        )
