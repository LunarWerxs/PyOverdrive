"""Fast path: numpy.intersect1d via sort-unique + searchsorted for integer arrays.

Provenance (OPP-000007): numpy/numpy#27042 asks for an ``assume_sorted``
hint so set operations on already-sorted inputs can use a merge or binary
search instead of concatenate-and-sort, claiming 10x-1000x. Dyno reproduced
86x for intersect1d at 1M x 100k int64 with a searchsorted candidate on
sorted unique inputs. The mechanism behind stock's slowness is two-fold and
both halves are addressed here, transparently, with no new keyword:

1. Stock ``intersect1d(a, b)`` (assume_unique=False) first calls ``unique``
   on each input through its own module-internal name, so it always takes
   the hash-based unique that OPP-000001 measured losing 37-101x to the
   sort path for fixed-width integers, and patching ``numpy.unique`` cannot
   reach it. This path dedups with the same sort-based routine
   ``unique_sort`` ships.
2. Stock then concatenates both unique sets and sorts AGAIN to find
   duplicates. This path instead binary-searches the smaller set into the
   larger (O(m log n)) and, when an input is already strictly increasing
   (checked in one O(n) pass, ~3% of a sort), skips its dedup sort
   entirely; that is the "sorted input" regime the issue is about.

Correctness contract:
- Exactly two positional plain 1-D ndarrays of the same supported integer
  dtype, no keywords (assume_unique= and return_indices= stay on stock).
- Result is bit-identical to stock: the sorted unique values present in
  both inputs, in the common dtype. No NaN semantics exist for integers,
  which is why floats are excluded here for now.
- ``isin`` is deliberately NOT accelerated: the same OPP-000007 battery
  measured the searchsorted approach LOSING 0.13-0.70x to stock's table
  method at every size.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=intersect_sorted or
pyoverdrive.disable_path("intersect_sorted").

Implementation note: never calls np.unique or np.intersect1d (patched names
while enabled; OPP-000000 recursion incident).
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath
from .unique_sort import _sort_unique

# Calibrated on fp 8f8198d9abab (contended run, 46-53% foreign load), evidence
# in benchmarks/results/INTERSECT-CAL/: combined size is the knob because the
# path's fixed cost is two sortedness checks plus dispatch on a microsecond
# stock call. Measured at combined 200 (100 x 100): sorted 1.5-1.7x, mixed
# 1.2-1.3x, random LOSES 0.87-0.89x. At combined 400 (200 x 200) every regime
# and dtype wins (random 1.19x int64 / 1.45x int32, sorted 2.1x); from
# combined 1,100 up it is 2.4x-6x, reaching 24-40x random, 71-91x mixed and
# 90-433x sorted at 1e6 x 1e4..1e5. Below 400 stays on stock.
#
# Small integers (itemsize <= 2) joined via OPP-000010 (SMALLINT-CAL,
# 2026-08-23, 9-19% load): with _sort_unique's kind='stable' radix switch
# they win 1.85-10.1x at combined 12_000 up (every dtype, both
# cardinalities), but low-cardinality inputs are marginal-to-losing below
# that (0.73-1.17x at combined 480-3_600), so their floor is the measured
# 12_000 edge, not the 32/64-bit 400.
SIZE_THRESHOLD = 400  # 32/64-bit combined floor; the name is imported by tests

_THRESHOLDS: dict[np.dtype, int] = {
    **{np.dtype(t): SIZE_THRESHOLD for t in (np.int32, np.int64, np.uint32, np.uint64)},
    **{np.dtype(t): 12_000 for t in (np.int8, np.uint8, np.int16, np.uint16)},
}
_SUPPORTED_DTYPES = frozenset(_THRESHOLDS)


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 2 or kwargs:
        return False
    a, b = args
    if type(a) is not np.ndarray or type(b) is not np.ndarray:
        return False
    if a.ndim != 1 or b.ndim != 1 or a.dtype != b.dtype:
        return False
    threshold = _THRESHOLDS.get(a.dtype)
    return threshold is not None and a.size + b.size >= threshold


def _sorted_unique_view(a: np.ndarray) -> np.ndarray:
    """``a`` itself when strictly increasing (already sorted and unique),
    else its sorted unique values."""
    if a.size == 0:
        return a
    if bool(np.all(a[1:] > a[:-1])):
        return a
    return _sort_unique(a)


def _intersect_sorted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ua = _sorted_unique_view(a)
    ub = _sorted_unique_view(b)
    if ua.size == 0 or ub.size == 0:
        return np.empty(0, dtype=a.dtype)
    # binary-search the smaller set into the larger; query order is sorted,
    # so the masked query is the sorted intersection
    ref, query = (ua, ub) if ua.size >= ub.size else (ub, ua)
    # stock_fn, not np.searchsorted: that name is patched too (the
    # sortqueries path), and this sorted query would pay its disorder
    # sample only to be refused on every call
    idx = GEARBOX.stock_fn("numpy.searchsorted")(ref, query)
    idx[idx == ref.size] = ref.size - 1  # past-the-end can never match
    return query[ref[idx] == query]


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="intersect_sorted",
            op="numpy.intersect1d",
            applicable=_applicable,
            run=_intersect_sorted,
            provenance={
                "opportunity": "OPP-000007",
                "source": "https://github.com/numpy/numpy/issues/27042",
                "license": "algorithm reimplemented; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
