"""Fast path: sort-based numpy.unique / numpy.unique_values for numeric arrays.

Provenance (OPP-000001): numpy/numpy#31969 reports the 2.x hash-based
`_unique_hash` (std::unordered_set) losing to the SIMD-accelerated sort path
by up to 46x at 1M int64 on AVX hardware, and ~3x even under heavy
duplication. Algorithm here is the classic sort + adjacent-difference mask,
reimplemented from first principles (no upstream code reused).

Correctness contract:
- Applies only to plain ndarrays (subclasses excluded so overrides keep
  working), default-argument calls, supported numeric dtypes.
- int16, uint16, int64 and uint64 stay on stock for both operations: independently repeated
  low/middle-cardinality losses withdrew their entire dtype rows.
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
#
# FLOOR RAISED 2026-08-26, 64 -> 1_000, for the reason above being true of
# ONE distribution. The battery drew high-cardinality operands; walking the
# distinct-value count against size on the idle Intel box
# (tools/probe_cardinality.py, two grids agreeing within 15% on 232 of 240
# cells) shows what the other end does:
#
#   int64      16      256     4096    65536       u   (distinct values)
#      64    1.01x    1.46x    1.47x    1.50x    1.51x
#     400    0.96x    1.96x    3.48x    3.69x    3.70x
#   1,000    1.14x    1.85x    4.48x    5.06x    5.18x
#
#   int32      16      256     4096    65536       u
#      64    1.00x    1.36x    1.38x    1.40x    1.37x
#     400    1.19x    2.19x    3.77x    3.93x    4.08x
#   1,000    1.38x    2.43x    5.67x    6.35x    6.52x
#
# At the old floor of 64 the worst cardinality is a WASH (1.00-1.01x), and
# at 400 int64 is actually losing. 1_000 is the first size where both
# widths clear 1.0x at every cardinality measured and int32 clears the
# project's 1.3x min-win (1.38x). The sweep's own row cell agrees with the
# corner this closes: unique_sort#int64%d read 0.9198/0.9233/0.9206/
# 0.9272/0.9184x across five fresh processes, which is as reproducible as a
# number gets.
#
# Historically NOT closed by this floor: int64 still
# reads 0.88x at n=10_000 and ~0.97x at n=100_000 in the 256-distinct band,
# in both grids. No floor removes that - it is a band in the middle, not a
# tail - and no gate can see it, because counting distinct values is the
# work itself. See docs/research/batch16-notes.md section 11 for the
# keep-or-withdraw argument and the numbers behind it. The fresh evidence
# below supersedes the earlier decision to keep these rows.
SIZE_THRESHOLD = 10_000  # retained 32-bit floor; the name is imported by tests

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
# WITHDRAWN 2026-09-19, both APIs: int64 and uint64. Quiet NumPy 2.5.2 on
# Intel fp 9bbe7063c555, two fresh processes per case, confirmed uint64
# losses at n=1,000 / 16 distinct (unique 0.9608/0.9714x; unique_values
# 0.8808/0.8902x). At 256 distinct, int64 unique loses 0.9412/0.9464x at
# n=10,000; unique_values loses 0.9726/0.9832x there and 0.9761/0.9577x at
# n=100,000. Raw samples and passing correctness are in
# benchmarks/results/BURNDOWN-20260919/intel-resumed-full.json and
# intel-middle-cardinality-before.json. These are cardinality bands, not a
# measured lower-size boundary; no unmeasured floor or version gate rescues
# them. Other dtypes retain their existing floors. _sort_unique remains
# available to independently guarded callers such as intersect_sorted.
# Also withdrawn: int16 on both APIs. The quiet NumPy 2.4.5 cross-version
# sweep confirmed low-cardinality losses at its 10,000-element floor:
# unique 0.9917/0.9641x; unique_values 0.9802/0.9751x. Evidence:
# BURNDOWN-20260919/versions/numpy-2.4.5.json. No new floor is inferred.
# FLOORS RAISED 2026-09-19 after the quiet NumPy 2.3.0 floor sweep:
# int32/uint32 at 10,000 and uint16 at 100,000 win both APIs across
# 16/256/broad-cardinality inputs in two fresh processes each. uint16
# still loses around 10,000. Use these measured sizes, not an inferred
# crossover: BURNDOWN-20260919/intel-floor-unique-grid.json (0.78-8.75% load).
# uint16 subsequently WITHDRAWN: that grid spread 16 values over the dtype
# range, while the final sweep's dense 0..15 values still lose at 100,000
# (unique 0.7293/0.7210x, unique_values 0.7339/0.7287x). Cardinality alone
# did not establish a safe floor. Quiet NumPy 2.3.0 evidence:
# BURNDOWN-20260919/gates-final/numpy-2.3.0.json (4.17-2.33% load).
_THRESHOLDS: dict[np.dtype, int] = {
    **{np.dtype(t): SIZE_THRESHOLD for t in (np.int32, np.uint32)},
    np.dtype(np.int8): 1_000,
    np.dtype(np.uint8): 1_000,
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
