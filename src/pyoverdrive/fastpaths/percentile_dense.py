"""Fast path: numpy.percentile with an ARRAY of percentiles - the shipped
quantile_dense_sort route with numpy's own q/100 scaling.

Provenance (OPP-000022, sibling surface): numpy documents percentile as
quantile at q/100 and routes it through the same partition-based core,
so it inherits the same dense-q partitioning cost. The route scales q
with np.true_divide(q, 100), matching NumPy, then uses the shared
sort-and-interpolate implementation. It retains that implementation's
shape, dtype, quantile-count and reduced-length gates.

Correctness contract: identical to quantile_dense_sort's, with q in
[0, 100] instead of [0, 1] and method='linear' absent/explicit; the
predicate and run are shared with that module apart from the scaling.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=percentile_dense or
pyoverdrive.disable_path("percentile_dense").

Historical calibration ratios are omitted because the NumPy version was not
recorded. See docs/research/2026-09-19-burndown.md for current measured
evidence and its version, hardware and load qualifications.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath
from . import quantile_dense_sort as _q

_F64 = np.dtype(np.float64)


def _applicable(args: tuple, kwargs: dict) -> bool:
    norm = _q._normalize(args, kwargs)
    if norm is None:
        return False
    a, q, axis = norm
    if type(a) is not np.ndarray or a.dtype != _F64:
        return False
    if type(q) is not np.ndarray or q.dtype != _F64 or q.ndim != 1:
        return False
    if not _q.NQ_MIN <= q.size <= _q.NQ_MAX:
        return False
    if a.ndim == 1:
        if axis not in (None, 0, -1):
            return False
    elif a.ndim == 2:
        if axis not in (1, -1):
            return False
    else:
        return False
    if not _q.M_MIN <= a.shape[-1] <= _q.M_MAX:
        return False
    return bool(((q >= 0.0) & (q <= 100.0)).all())


def _run(a, q, axis=None, method="linear"):
    return _q._run(a, np.true_divide(q, 100), axis)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="percentile_dense",
            op="numpy.percentile",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000022",
                "source": "https://github.com/numpy/numpy/issues/32187",
                "license": "shared with quantile_dense_sort; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
