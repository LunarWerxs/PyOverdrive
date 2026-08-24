"""Fast path: numpy.unique(a, axis=0) for a single-column integer matrix.

Provenance (OPP-000014): numpy/numpy#15713 reports ``np.unique(arr, axis=0)``
being ~10x slower than the 1-D route even when the matrix has ONE column,
because the axis path packs rows into structured/void scalars before
sorting. Dyno reproduced 10.2x at the issue's own N=1000 int64 regime and up
to 66x at N=100k float32 (evidence: benchmarks/results/OPP-000014/,
contended run). For a single column the row view adds nothing: the unique
rows of an (n, 1) matrix are exactly the unique values of its column,
reshaped back.

Correctness contract:
- Applies only to ``np.unique(a, axis=0)``: one positional plain 2-D
  ndarray with ``shape[1] == 1``, the single keyword ``axis`` equal to 0,
  and a dtype in unique_sort's measured integer set (int32/int64/uint32/
  uint64, plus int8/uint8/int16/uint16 since OPP-000010's radix switch,
  measured 40-298x for this path at 1000-100k rows in SMALLINT-CAL: for
  integers the structured-field comparison stock uses per row equals
  numeric comparison per value, so the result is bit-identical to stock:
  sorted unique values, shape (u, 1), C order). Floats are excluded:
  stock's axis path compares NaNs as structured fields (all NaNs equal, and
  every NaN bit pattern is kept distinct from none), which does NOT match
  the 1-D NaN-run collapse, so no float claim is made without its own
  evidence.
- Every other axis value, shape, dtype and keyword combination stays on
  stock.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=unique_axis0_column or
pyoverdrive.disable_path("unique_axis0_column").

Size floor: the battery measured N in [1000, 100000], winning 10-66x
everywhere; below N=1000 is unmeasured and stays on stock.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath
from .unique_sort import _SUPPORTED_DTYPES, _sort_unique

SIZE_THRESHOLD = 1000  # rows; smallest measured winning size (10.2x there)


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 1 or len(kwargs) != 1:
        return False
    if kwargs.get("axis", None) != 0:
        return False
    a = args[0]
    return (
        type(a) is np.ndarray
        and a.ndim == 2
        and a.shape[1] == 1
        and a.shape[0] >= SIZE_THRESHOLD
        and a.dtype in _SUPPORTED_DTYPES
    )


def _unique_column(a: np.ndarray, axis: int = 0) -> np.ndarray:
    return _sort_unique(a.ravel()).reshape(-1, 1)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="unique_axis0_column",
            op="numpy.unique",
            applicable=_applicable,
            run=_unique_column,
            provenance={
                "opportunity": "OPP-000014",
                "source": "https://github.com/numpy/numpy/issues/15713",
                "license": "algorithm reimplemented; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
