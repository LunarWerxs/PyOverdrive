"""Fast paths: numpy.nanargmax / nanargmin via one isnan scan.

Provenance (OPP-000042): both wrappers build a where(isnan, +-inf)
masked COPY before reducing, even when no NaN exists. For NaN-free
input, plain argmax/argmin returns the identical index (same
first-occurrence tie-breaking: it is literally the same reduction on
the same values). The pydata/bottleneck README table documents the
overhead class; OPP-000041 is the sibling record for the aggregations.

Route: NaN probe via np.isnan(np.min(a)) (min propagates NaN). Clean
input takes plain argmax/argmin; NaN-present input falls back to stock
INSIDE the run, preserving stock's semantics exactly, including the
all-NaN-slice ValueError.

CALIBRATION (fp 9bbe7063c555, idle box, 0% load, numpy 2.5.2,
benchmarks/results/BATCH6-CAL/): 1.88x at n=300 and n=1000, 1.75x at
1e4, 3.04x at 1e5, 5.34x at (1000,1000) axis=1; wasted-scan (1% NaN)
0.89x. Floor below is the measured edge.

Correctness contract:
- plain float64 ndarray, non-empty; axis absent, None, or a single int;
  no other kwargs (out/keepdims force stock). Zero-length reduced axes
  refuse (stock raises; the plain op raises a DIFFERENT message).
- clean input: bit-identical index, dtype intp.
- NaN present (all-NaN slices included): stock's own result or
  exception, via internal fallback.

Comparison mode: bit-identical. Kill switches: nanargmax_scan,
nanargmin_scan.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath, StockRaised

_F64 = np.dtype(np.float64)
SIZE_FLOOR = 300  # smallest measured winning cell


def _normalize(args: tuple, kwargs: dict):
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


def _applicable(args: tuple, kwargs: dict) -> bool:
    norm = _normalize(args, kwargs)
    if norm is None:
        return False
    a, axis = norm
    if type(a) is not np.ndarray or a.dtype != _F64:
        return False
    if a.size < SIZE_FLOOR:
        return False
    if axis is None:
        return True
    if isinstance(axis, bool) or not isinstance(axis, int):
        return False
    if not -a.ndim <= axis < a.ndim:
        return False
    return a.shape[axis] > 0


def _run_for(op: str, plain_name: str):
    def _run(a, axis=None):
        if np.isnan(np.min(a)):
            # NaN somewhere (all-NaN slices raise in stock): stock decides.
            # A stock exception is stock BEHAVIOR, not a path failure -
            # StockRaised makes the dispatcher re-raise it silently.
            try:
                return GEARBOX.stock_fn(op)(a, axis=axis)
            except Exception as exc:
                raise StockRaised(exc) from None
        return GEARBOX.stock_fn(plain_name)(a, axis=axis)

    return _run


def register(gearbox) -> None:
    for op, plain, path_name in (
        ("numpy.nanargmax", "numpy.argmax", "nanargmax_scan"),
        ("numpy.nanargmin", "numpy.argmin", "nanargmin_scan"),
    ):
        gearbox.register(
            FastPath(
                name=path_name,
                op=op,
                applicable=_applicable,
                run=_run_for(op, plain),
                provenance={
                    "opportunity": "OPP-000042",
                    "source": "https://github.com/pydata/bottleneck (README benchmark table)",
                    "license": "isnan scan + plain argmax/argmin, standard technique; no third-party code",
                    "comparison_mode": "bit-identical",
                },
            )
        )
