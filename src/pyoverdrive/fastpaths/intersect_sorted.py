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
- Only int32 and uint32 dispatch; other integer dtypes stay on stock after repeated
  low-cardinality losses withdrew those complete dtype rows.
- WHY assume_unique= is refused here while isin_object_hash and
  isin_string_hash ACCEPT it, which looks like two rules for one keyword
  and is not. assume_unique is a promise the caller makes and numpy does
  not verify. For isin the promise cannot change the answer: membership is
  idempotent under duplication, so dedup or not, every element gets the
  same verdict - checked with the promise deliberately false (duplicates
  in both operands) and dispatched matches stock exactly, both when it is
  True and when it is False. For intersect1d it CAN change the answer:
  stock with assume_unique=True skips its unique() calls entirely and
  takes a different route through the concatenate-and-sort, so a false
  promise produces a specific wrong result that this path would have to
  reproduce exactly rather than merely be correct about. Refusing is the
  cheaper honesty.
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
# and dtype wins (random 1.19x int64 / 1.45x int32, sorted 2.1x - all from a CONTENDED run.
# Re-measured on the idle box 2026-08-25 the same floor is 1.50x int64 /
# 1.46x int32 random, rising to 5.5-6.6x by combined 20k. A threshold
# audit flagged the 1.19x as too close to noise to trust; it was, and the
# honest number is comfortably above it); from
# combined 1,100 up it is 2.4x-6x, reaching 24-40x random, 71-91x mixed and
# 90-433x sorted at 1e6 x 1e4..1e5. Below 400 stays on stock.
#
# Small integers (itemsize <= 2) joined via OPP-000010 (SMALLINT-CAL,
# 2026-08-23, 9-19% load): with _sort_unique's kind='stable' radix switch
# they win 1.85-10.1x at combined 12_000 up (every dtype, both
# cardinalities), but low-cardinality inputs are marginal-to-losing below
# that (0.73-1.17x at combined 480-3_600), so their floor is the measured
# 12_000 edge, not the 32/64-bit 400.
# THE 32/64-BIT ROWS BELOW ARE NUMPY>=2.3 NUMBERS, and were shipped as if they
# were numpy>=2.0 numbers. Re-judged against every supported version on
# 2026-08-25 (tools/verify_across_numpy.py) the four WIDE integer rows lose
# below 2.3 and win at and above it, with no ambiguity in between:
#   int32   0.52x / 0.53x / 0.53x  ->  1.21x / 1.23x / 1.39x
#   int64   0.62x / 0.63x / 0.62x  ->  1.32x / 1.26x / 1.42x
#   uint32  0.53x / 0.53x / 0.52x  ->  1.22x / 1.23x / 1.39x
#   uint64  0.63x / 0.61x / 0.62x  ->  1.32x / 1.28x / 1.56x
#     (numpy 2.0.2 / 2.1.3 / 2.2.6  ->  2.3.5 / 2.4.6 / 2.5.2)
# The eight-cell consistency below the cliff is what makes this a version
# effect rather than noise. The narrow rows (itemsize <= 2) are unaffected -
# they win 3.5-7.7x on every version measured - so it is specifically the
# comparison-sort route, not the radix one. numpy 2.3 rewrote np.unique to
# try a hash table first (gh-26018), which is the likeliest cause on stock's
# side, and it is worth knowing that this row's margin therefore depends on a
# stock implementation that upstream is actively changing.
# The package floor moved to numpy>=2.3 rather than these rows being
# withdrawn; see pyproject.toml for the reasoning and the other family.
#
# FLOOR RAISED 2026-08-26, 400 -> 10_000, because 400 was measured at ONE
# cardinality and the row is 31% SLOWER than stock at another. Walking size
# against distinct-value count on the idle Intel box (fp 9bbe7063c555,
# tools/probe_cardinality.py, two independent grids agreeing within 15% on
# 232 of 240 cells), int64 reads:
#
#     n        16      256     4096    65536       u   (distinct values)
#   400      0.69x    1.23x    1.95x    2.01x    2.05x
#   1,000    0.75x    1.40x    2.98x    3.36x    3.33x
#   3,000    0.99x    1.26x    4.44x    5.70x    5.84x
#   10,000   1.31x    1.11x    3.34x    7.00x    7.52x
#   30,000   1.57x    1.41x    2.27x    6.59x    9.03x
#
# The old floor sat at the top-left corner, where the path loses a third of
# stock's speed. 10_000 is the first size whose WORST cardinality clears
# 1.0x (1.11x int64, 1.77x int32) - the rule calibrate_dispatch.py already
# states for its own variants: every one of them is something a caller can
# hit, so the threshold has to hold for the worst, and NOTHING in the gate
# can see cardinality cheaply at these sizes (counting distinct values is
# the work itself).
#
# This forfeits the 2.0-5.8x that high-cardinality inputs win between 400
# and 10_000, deliberately and with the same reasoning as relayout's floor
# earlier in batch 16: a threshold that only holds for the distribution its
# battery happened to draw is not a threshold.
SIZE_THRESHOLD = 10_000  # 32/64-bit combined floor; the name is imported by tests

# WITHDRAWN 2026-09-19: int16/uint16 at combined 12,000 and 16 distinct
# values lost in two fresh processes each on quiet Intel fp 9bbe7063c555,
# NumPy 2.5.2: int16 0.9398/0.9810x, uint16 0.9251/0.9215x. Raw samples,
# passing correctness and no runtime fallback are recorded in
# benchmarks/results/BURNDOWN-20260919/intel-resumed-full.json.
# Cardinality is not cheaply known at dispatch, and no new safe floor was
# measured. Withdraw both rows rather than infer a threshold or version gate.
# Also withdrawn after quiet NumPy 2.3.0 verification: int64/uint64 lose
# at combined 10,000 with balanced operands and uint8 at combined 12,000;
# canonical int64 also loses at combined 12,000. Cardinality and operand
# balance affect these losses; no higher safe floor is inferred. Evidence:
# BURNDOWN-20260919/floor-final/numpy-2.3.0.json (1.41-0.31% load).
# int8 subsequently WITHDRAWN after final quiet NumPy 2.3.0 verification
# confirmed dense low-cardinality losses at 12,000: 0.9501/0.9940x.
# BURNDOWN-20260919/gates-final/numpy-2.3.0.json (4.17-2.33% load).
_THRESHOLDS: dict[np.dtype, int] = {
    **{np.dtype(t): SIZE_THRESHOLD for t in (np.int32, np.uint32)},
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
