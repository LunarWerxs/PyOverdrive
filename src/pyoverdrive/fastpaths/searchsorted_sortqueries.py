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
on 2.5.2), making the result identical even for abusive input.

That divergence was known and tolerated - the path dispatched anyway and a
strict xfail recorded the boundary. Batch 16 reversed it and the predicate
now REFUSES an unsorted haystack outright (_haystack_sorted). Not because
the old call was unreasonable, since this is undefined behaviour on numpy's
side either way, but because the package floor is numpy>=2.3: the divergent
versions are SUPPORTED versions, and this project had already ruled the
other way on the same shape, isin_string_hash refusing lone-NUL strings so
stock keeps answering where stock is quirky. Two opposite decisions about
undefined behaviour is one too many, and the guard costs 0.1-0.5% of the
call. On numpy >= 2.5 it declines an input that would have agreed.

Correctness contract:
- Applies only to searchsorted(a, v[, side]) where a and v are plain 1-D
  ndarrays of the same float64 dtype (int64 was withdrawn - see SUPPORTED),
  side is 'left' (the default)
  or 'right' (same mechanism by per-element independence; the calibration
  battery carries its measurement), no sorter=, len(a) >= 10_000,
  len(v) at or above the measured dtype floor and at most 10x len(a),
  AND a sampled disorder estimate says the query order is genuinely
  random-like (the SEARCHSORTED-CAL battery measured stock already fast
  on sorted, nearly-sorted, lightly shuffled AND descending queries; only
  high-disorder orders repay the sort), AND the HAYSTACK IS ACTUALLY
  SORTED, checked in one O(n) pass (_haystack_sorted). Scalar v, other
  dtypes, 2-D+ v, and sorter= all stay on stock.
- That haystack check is a correctness guard, not a performance one, and
  it was added after the fact: numpy does not verify `a` is sorted, an
  unsorted one yields a meaningless-but-deterministic answer, and on
  numpy < 2.5 this path returned a DIFFERENT one - 86.8% of positions
  disagreed with stock on a 50,000-element unsorted float64 haystack under
  2.4.5, a supported version. It costs 0.1-0.5% of the
  call it guards (3 us against 593 us at n=10,000; 250 us against 203 ms
  at n=1e6), which cannot reach the 1.28-1.97x margin.
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
# INT64 WITHDRAWN 2026-08-25, and the reason is a new failure class here:
# the row was not measured wrong, it ROTTED. numpy 2.5 made stock
# searchsorted substantially faster and took the whole int64 margin with it.
# Same grid, same idle box, same code, two numpy versions
# (tools/probe_searchsorted_haystack.py, haystack x queries, both dtypes):
#   numpy 2.4.5  int64: 0.73x at a 10k haystack, 1.06x at 100k, 1.90x at
#                300k, 2.46x at 1M, 3.97x at 3M - the row as calibrated.
#   numpy 2.5.2  int64: 0.26x, 0.29x, 0.37x, 0.46x, 0.74x - NOT ONE CELL
#                of its admissible region reaches break-even, and the
#                worst is 3.8x slower than stock.
# float64 survived the same change with a reduced margin (1.77-4.41x on
# 2.4.5, 1.28-1.97x on 2.5.2), so the dtype is what separates them: stock's
# int64 comparison is now cheap enough that the argsort can never be repaid,
# while float64 still leaves room. A supported numpy that loses is a loss,
# so the row goes rather than becoming version-conditional.
# The original 1.51x-at-1e5 reading also used a LARGER haystack than the
# query count it was recorded against; at matched sizes 2.4.5 gives 1.06x.
SUPPORTED: dict[np.dtype, int] = {
    np.dtype(np.float64): 10_000,
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


def _haystack_sorted(a: np.ndarray) -> bool:
    """Is the haystack actually sorted? One O(n) pass, and it is REQUIRED.

    numpy documents `a` as needing to be sorted and does not check it: an
    unsorted haystack gets you a meaningless answer rather than an error.
    Meaningless is not the same as arbitrary, though - stock's answer is a
    deterministic function of the array it was handed, and on numpy < 2.5
    THIS PATH RETURNS A DIFFERENT ONE. Measured on a 50,000-element
    unsorted float64 haystack under numpy 2.4.5: dispatched, and 43,407 of
    50,000 positions disagree with stock (86.8%). The package floor is
    numpy>=2.3, so those are SUPPORTED versions.

    On numpy 2.5 and later the two agree - the order dependence was removed
    upstream - so there this check refuses an input that would have
    matched. That is deliberate: a predicate whose correctness depends on
    the numpy version is a predicate nobody can reason about, and the cost
    is 0.1-0.5%.

    The reason is the path's own mechanism. Sorting the queries is only
    order-neutral if each query is answered independently; numpy narrows
    the search range as it walks a SORTED query list, which is valid when
    the haystack is sorted and wrong when it is not. So the very
    optimisation this path exists for is what makes it diverge here.

    Undefined behaviour is still behaviour a caller can be relying on, and
    this project has ruled on exactly this shape once already: isin_string_
    hash refuses lone-NUL strings so stock keeps answering for an input
    class where stock is quirky. Bug-for-bug faithfulness wins, so an
    unsorted haystack goes to stock.

    Non-decreasing, not strictly increasing: searchsorted is defined on
    haystacks with duplicates and this is not a uniqueness check.

    A haystack containing NaN is refused as a side effect, since NaN
    compares False in either direction - conservative, consistent with
    _disordered's treatment of NaN queries, and it only ever costs a
    dispatch.
    """
    return a.size < 2 or bool(np.all(a[1:] >= a[:-1]))


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
    # _disordered is a 4096-point sample and _haystack_sorted is a full
    # O(n) pass, so the sampled one goes first: the scan is only paid by an
    # input that would otherwise have dispatched.
    if not _disordered(v):
        return False
    return _haystack_sorted(a)


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
