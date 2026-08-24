"""Fast path: numpy.searchsorted with a large unsorted query array.

Provenance (OPP-000015): numpy/numpy#10937 (open since 2018, no PR ever)
reports searchsorted on 5M sorted haystack x 5M UNSORTED queries taking
10.4s where sorting the queries first drops the same call to 971ms
(~10.7x): consecutive sorted queries land near each other, so the binary
search's memory access pattern turns cache-friendly and branch-predictable
(juliantaylor's hypothesis in-thread). Dyno reproduced 2.2-7.4x float64
and up to 14.5x int64 (benchmarks/results/OPP-000015/, contended run).

Mechanism: argsort the queries, searchsorted the sorted copy against the
same haystack, scatter the indices back through the permutation. When the
haystack satisfies searchsorted's documented precondition (sorted), each
element's insertion index depends only on its own value and the haystack,
so the unpermuted result is BIT-IDENTICAL to stock's for any query values,
NaN and inf included. With an UNSORTED haystack the precondition is
violated, and on numpy < 2.5 stock's own batched result is query-ORDER-
dependent (found by this family's differential battery: the C
implementation chains a locality hint across consecutive queries, harmless
on a sorted haystack, order-sensitive garbage otherwise; 17122/20000
elements differed in the 2.4.5 probe), so there the fast path returns
another member of the same undefined family, not stock's byte-for-byte
garbage. numpy 2.5 removed the order dependence (0/20000 in the same probe
on 2.5.2), making the result identical even for abusive input; the
version-conditional xfail in the differential battery tracks the boundary.

Correctness contract:
- Applies only to searchsorted(a, v[, side]) where a and v are plain 1-D
  ndarrays of the same float64/int64 dtype, side is 'left' (the default)
  or 'right' (same mechanism by per-element independence; the calibration
  battery carries its measurement), no sorter=, len(a) >= 10_000,
  len(v) at or above the measured dtype floor and at most 10x len(a),
  AND a sampled disorder estimate says the query order is genuinely
  random-like (the SEARCHSORTED-CAL battery measured stock already fast
  on sorted, nearly-sorted, lightly shuffled AND descending queries; only
  high-disorder orders repay the sort). Scalar v, other dtypes, 2-D+ v,
  and sorter= all stay on stock.
- The inner searchsorted call goes through Gearbox's stock_fn: the sorted
  copy would pass this predicate again and recurse (the OPP-000000
  incident class).

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=searchsorted_sortqueries or
pyoverdrive.disable_path("searchsorted_sortqueries").
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

# CALIBRATION, two batteries (fp 8f8198d9abab):
# - OPP-000015 sweep (contended 39-46%): float64 wins from len(v)=1e4
#   (2.18x equal, 2.32-2.56x vs a 1e6 haystack, 7.37x at the reporter's
#   5e6; 0.41-0.73x below), int64 needs 1e5 (0.99x wash at 1e4, 1.51x at
#   1e5, 14.5x at 1e6).
# - SEARCHSORTED-CAL (13-20% load) measured the gaps and CUT the region
#   hard. The win needs genuinely DISORDERED queries: already-sorted
#   0.79-0.93x, 1%-swapped 0.48x, 5%/10%-shuffled 1.07x/1.08x,
#   25%-shuffled 1.40x, fully random 3.28x; DESCENDING queries lose 0.62x
#   despite maximal descent count (stock's binary search loves any
#   locality, ascending or not). Descent fractions of those same arrays:
#   0.0 / 0.01 / 0.091 / 0.165 / 0.316 / 0.50 / 1.0 - so the gate is a
#   SAMPLED disorder estimate d = min(frac_descents, 1 - frac_descents)
#   over 4096 evenly strided adjacent pairs (sampling error ~0.016,
#   measured), requiring d >= 0.40: only random-like query orders pass,
#   and the 1.40x marginal point (0.316) stays on stock deliberately.
#   It also needs a real haystack: x=64 loses 0.56x, x=1e3 0.95x, x=1e4
#   only 1.32x under a 100x larger query set (while 2.18x when sizes
#   match), so a.size >= 10_000 AND len(v) <= 10 * len(a) (the measured
#   5:1 ratio case wins 1.46x; 100:1 is where it dies).
SUPPORTED: dict[np.dtype, int] = {
    np.dtype(np.float64): 10_000,
    np.dtype(np.int64): 100_000,
}
_FLOOR = min(SUPPORTED.values())
_HAYSTACK_FLOOR = 10_000
_SKEW_CAP = 10  # len(v) <= _SKEW_CAP * len(a)
_DISORDER_GATE = 0.40
_DISORDER_SAMPLES = 4096


def _disordered(v: np.ndarray) -> bool:
    """Sampled disorder of the query order: min(descents, ascents) fraction
    over ~4096 evenly strided adjacent pairs. NaN comparisons count as
    ordered, so NaN-heavy queries read as sorted and stay on stock."""
    p = np.linspace(0, v.size - 2, _DISORDER_SAMPLES).astype(np.intp)
    frac = float(np.count_nonzero(v[p + 1] < v[p])) / p.size
    return min(frac, 1.0 - frac) >= _DISORDER_GATE


def _operands(args: tuple, kwargs: dict):
    if len(args) == 3:
        if kwargs:
            return None
        a, v, side = args
    elif len(args) == 2:
        if kwargs and (len(kwargs) != 1 or "side" not in kwargs):
            return None
        a, v = args
        side = kwargs.get("side", "left")
    else:
        return None
    if side != "left" and side != "right":
        return None
    return a, v


def _applicable(args: tuple, kwargs: dict) -> bool:
    ops = _operands(args, kwargs)
    if ops is None:
        return False
    a, v = ops
    if type(a) is not np.ndarray or type(v) is not np.ndarray:
        return False
    if a.ndim != 1 or v.ndim != 1 or a.dtype != v.dtype:
        return False
    if v.size < _FLOOR or a.size < _HAYSTACK_FLOOR:
        return False
    if v.size > _SKEW_CAP * a.size:
        return False
    threshold = SUPPORTED.get(v.dtype)
    if threshold is None or v.size < threshold:
        return False
    return _disordered(v)


def _run(a, v, side="left"):
    perm = np.argsort(v)
    idx = GEARBOX.stock_fn("numpy.searchsorted")(a, v[perm], side=side)
    out = np.empty_like(idx)
    out[perm] = idx
    return out


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="searchsorted_sortqueries",
            op="numpy.searchsorted",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000015",
                "source": "https://github.com/numpy/numpy/issues/10937",
                "license": "sort-first idea from the issue report, reimplemented; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
