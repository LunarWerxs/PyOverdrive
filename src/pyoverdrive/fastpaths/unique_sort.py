"""Fast path: sort-based numpy.unique / numpy.unique_values for numeric arrays.

Provenance (OPP-000001): numpy/numpy#31969 reports the 2.x hash-based
`_unique_hash` (std::unordered_set) losing to the SIMD-accelerated sort path
by up to 46x at 1M int64 on AVX hardware, and ~3x even under heavy
duplication. Algorithm here is the classic sort + adjacent-difference mask,
reimplemented from first principles (no upstream code reused).

Correctness contract:
- Applies only to plain ndarrays (subclasses excluded so overrides keep
  working), default-argument calls, supported numeric dtypes.
- Output is bit-identical to stock np.unique: sorted unique values, with a
  float NaN run collapsed to a single trailing NaN exactly as np.unique does.
- numpy.unique_values guarantees no output order; returning sorted values
  satisfies that contract strictly.

Comparison mode: bit-identical (spec section 9).

The size threshold below which stock wins is hardware-calibrated with
benchmarks/historical/opp_000001_unique.py; the shipped default is
conservative for AVX-capable x86. Kill switch: PYOVERDRIVE_DISABLE=unique_sort
or pyoverdrive.disable_path("unique_sort").

Implementation note: this path deliberately never calls np.unique (that name
is the Gearbox wrapper while patched; see the OPP-000000 recursion incident).
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

# Calibrated on fp 8f8198d9abab (Zen 4 AVX-512, numpy 2.4.5), evidence in
# benchmarks/results/OPP-000001/: sort wins at every measured size >= 64 for
# the supported dtypes (1.8x at n=64 up to 88x at 1M high cardinality, and
# still 1.2-2.3x at low cardinality). Below 64 is unmeasured, so excluded.
SIZE_THRESHOLD = 64  # 32/64-bit floor; the name is imported by tests

# Measured IN: int32/int64 (37-101x), uint32/uint64 (34-88x) with the
# default quicksort, floor 64.
# Measured IN via kind='stable' (OPP-000010, SMALLINT-CAL battery,
# 2026-08-23, 9-19% load): numpy's stable kind is a RADIX sort for
# itemsize <= 2, and with it the small dtypes flip from the original
# losses (quicksort int16 0.38-0.46x, int8 0.14x, remeasured as low as
# 0.13x at 100k) to wins at every measured size and cardinality:
# int8/uint8 1.6-3.7x from n=1000, uint16 1.6-27.8x from n=1000, int16
# 8.5-21.6x high-cardinality but only 1.22-1.48x at n=1000 low
# cardinality across two runs (straddles the 1.3x min-win, so its floor
# sits at the clean 10_000 measurement instead).
# Measured OUT: float64 (0.84-1.07x, regression risk), float32 (parity
# within noise), bools/strings/objects (never measured, structurally
# different).
_THRESHOLDS: dict[np.dtype, int] = {
    **{np.dtype(t): SIZE_THRESHOLD for t in (np.int32, np.int64, np.uint32, np.uint64)},
    np.dtype(np.int8): 1_000,
    np.dtype(np.uint8): 1_000,
    np.dtype(np.int16): 10_000,
    np.dtype(np.uint16): 1_000,
}
_SUPPORTED_DTYPES = frozenset(_THRESHOLDS)


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 1 or kwargs:
        return False
    a = args[0]
    if type(a) is not np.ndarray:
        return False
    threshold = _THRESHOLDS.get(a.dtype)
    return threshold is not None and a.size >= threshold


def _sort_unique(a: np.ndarray) -> np.ndarray:
    # itemsize <= 2: numpy's stable kind is a radix sort for narrow ints
    # (OPP-000010); wider dtypes keep the default quicksort, where forcing
    # mergesort measured 3-4x SLOWER
    kind = "stable" if a.dtype.itemsize <= 2 else None
    # stock_fn, not np.sort / np.searchsorted below: both names are patched
    # too (sort_char_view, searchsorted_sortqueries)
    s = GEARBOX.stock_fn("numpy.sort")(a, axis=None, kind=kind)  # flattens, matching np.unique semantics
    mask = np.empty(s.size, dtype=bool)
    mask[0] = True
    np.not_equal(s[1:], s[:-1], out=mask[1:])
    if s.dtype.kind == "f" and np.isnan(s[-1]):
        # np.unique collapses the NaN run (NaN != NaN keeps every one)
        first_nan = GEARBOX.stock_fn("numpy.searchsorted")(s, np.nan, side="left")
        mask[first_nan + 1 :] = False
        mask[first_nan] = True
    return s[mask]


_PROVENANCE = {
    "opportunity": "OPP-000001",
    "source": "https://github.com/numpy/numpy/issues/31969",
    "license": "algorithm reimplemented; no third-party code",
    "comparison_mode": "bit-identical",
}


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="unique_sort",
            op="numpy.unique",
            applicable=_applicable,
            run=_sort_unique,
            provenance=_PROVENANCE,
        )
    )
    gearbox.register(
        FastPath(
            name="unique_values_sort",
            op="numpy.unique_values",
            applicable=_applicable,
            run=_sort_unique,
            # np.unique_values promises no order; stock's hash path returns
            # an unordered set, this path a sorted one. Same set, by contract.
            provenance=dict(_PROVENANCE, comparison_mode="set-equal (unique_values promises no order)"),
        )
    )
