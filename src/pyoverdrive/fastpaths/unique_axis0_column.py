"""Fast path: numpy.unique(a, axis=0) for a single-column integer matrix.

Provenance (OPP-000014): numpy/numpy#15713 reports ``np.unique(arr, axis=0)``
paying for structured/void row packing even for a single column. For
an (n, 1) integer matrix, unique rows are exactly the unique column values,
reshaped back. Avoiding row packing is the mechanism; it does not imply
that every dtype or NaN convention can use the same route.

Correctness contract:
- Applies only to ``np.unique(a, axis=0)``: one positional plain 2-D
  ndarray with ``shape[1] == 1``, the single keyword ``axis`` equal to 0,
  and one of the eight native integer dtypes (int8/uint8/int16/uint16/
  int32/uint32/int64/uint64; the small types use the radix switch from
  OPP-000010). This axis-specific regime is independent of 1-D unique: for
  integers the structured-field comparison stock uses per row equals
  numeric comparison per value, so the result is bit-identical to stock:
  sorted unique values, shape (u, 1), C order. Floats are excluded:
  stock's axis path compares NaNs as structured fields (all NaNs equal, and
  every NaN bit pattern is kept distinct from none), which does NOT match
  the 1-D NaN-run collapse, so no float claim is made without its own
  evidence.
- Every other axis value, shape, dtype and keyword combination stays on
  stock.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=unique_axis0_column or
pyoverdrive.disable_path("unique_axis0_column").

The size floor keeps small calls on stock rather than extrapolating the
row-packing benefit below the calibrated input range.

Historical calibration ratios are omitted because the NumPy version was not
recorded. See docs/research/2026-09-19-burndown.md for current measured
evidence and its version, hardware and load qualifications.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath
from .unique_sort import _sort_unique

SIZE_THRESHOLD = 1000  # rows; lower boundary of the served regime
# Reuse the sorting kernel, not the 1-D dispatch policy. Stock axis=0 pays
# for structured row packing; its competing algorithm and measured regime
# differ from 1-D hashing, so 1-D dtype withdrawals do not apply here.
_SUPPORTED_DTYPES = frozenset(np.dtype(t) for t in (
    np.int8, np.uint8, np.int16, np.uint16,
    np.int32, np.uint32, np.int64, np.uint64,
))


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
