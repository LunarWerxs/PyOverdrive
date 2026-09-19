"""Differential tests: searchsorted_sortqueries fast path vs stock numpy.searchsorted.

Contract (src/pyoverdrive/fastpaths/searchsorted_sortqueries.py): applies only
to searchsorted(a, v[, side]) where side is the 3rd positional arg or a
'side' kwarg with value 'left' (default) or 'right'; a and v are plain 1-D
ndarrays of the same dtype from {float64, int64}; no sorter=; len(a) >=
10_000 (HAYSTACK_FLOOR); len(v) >= the dtype floor (10_000 float64, 100_000
int64, the SUPPORTED dict) and len(v) <= 10 * len(a) (SKEW_CAP); and a
sampled disorder estimate over 4096 evenly strided adjacent pairs of v must
have min(descent_fraction, 1 - descent_fraction) >= 0.40, so only
random-ordered queries dispatch (sorted, nearly-sorted, and descending
queries all refuse). Dispatch argsorts v, searchsorted's the sorted copy
against a via GEARBOX.stock_fn (bypassing the dispatcher so the sorted copy
cannot re-trigger this same path), then scatters the indices back through
the permutation. Since each element's insertion index depends only on its
own value and a, never on its neighbors, the result is bit-identical to
stock for any input (NaNs and even an unsorted haystack included: garbage
in, the same garbage out, element for element). Comparison mode:
bit-identical.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

OP = "numpy.searchsorted"
PATH = "searchsorted_sortqueries"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _sorted_float(n, seed, lo=-1000.0, hi=1000.0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(lo, hi, size=n))


def _random_float(n, seed, lo=-1000.0, hi=1000.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=n)


def _sorted_int(n, seed, lo=-(10**12), hi=10**12):
    rng = np.random.default_rng(seed)
    return np.sort(rng.integers(lo, hi, size=n, dtype=np.int64))


def _random_int(n, seed, lo=-(10**12), hi=10**12):
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, size=n, dtype=np.int64)


def _assert_dispatched_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.searchsorted(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert np.asarray(got).dtype == np.asarray(stock).dtype
    assert np.asarray(got).shape == np.asarray(stock).shape
    assert np.array_equal(got, stock)
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.searchsorted(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert np.array_equal(got, stock)
    return got


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.searchsorted(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + bit-identity
# ---------------------------------------------------------------------------

def test_dispatch_float64_side_omitted():
    x = _sorted_float(50_000, seed=1)
    v = _random_float(50_000, seed=2)
    _assert_dispatched_equal((x, v), {})


def test_dispatch_float64_side_left_kwarg():
    x = _sorted_float(50_000, seed=1)
    v = _random_float(50_000, seed=2)
    _assert_dispatched_equal((x, v), {"side": "left"})


def test_dispatch_float64_side_right_kwarg():
    x = _sorted_float(50_000, seed=1)
    v = _random_float(50_000, seed=2)
    _assert_dispatched_equal((x, v), {"side": "right"})


def test_dispatch_float64_side_positional():
    x = _sorted_float(50_000, seed=1)
    v = _random_float(50_000, seed=2)
    _assert_dispatched_equal((x, v, "right"), {})


# int64 WAS a dispatching dtype until batch 16 withdrew the row: numpy 2.5
# made stock searchsorted fast enough that sorting the queries can no longer
# repay the argsort for integers, and the row measured 0.26-0.74x across its
# whole admissible region (see searchsorted_sortqueries.SUPPORTED). These two
# stay as REFUSAL tests rather than being deleted, because they are the exact
# shapes that used to dispatch - if int64 ever comes back it must come back
# on a measurement, not by accident.
def test_refusal_int64_wide_range():
    x = _sorted_int(200_000, seed=3)
    v = _random_int(200_000, seed=4)
    _assert_refused_equal((x, v), {})


def test_refusal_int64_duplicates_heavy_both_sides():
    rng_x = np.random.default_rng(5)
    rng_v = np.random.default_rng(6)
    x = np.sort(rng_x.integers(0, 50, size=150_000, dtype=np.int64))
    v = rng_v.integers(0, 50, size=150_000, dtype=np.int64)
    _assert_refused_equal((x, v), {"side": "left"})
    _assert_refused_equal((x, v), {"side": "right"})


def test_dispatch_float64_queries_with_few_nans():
    x = _sorted_float(50_000, seed=7)
    v = _random_float(50_000, seed=8).copy()
    v[[100, 12345, 25000, 37777, 49999]] = np.nan
    _assert_dispatched_equal((x, v), {})


def test_dispatch_float64_queries_with_inf():
    x = _sorted_float(50_000, seed=9)
    v = _random_float(50_000, seed=10).copy()
    v[0] = np.inf
    v[1] = -np.inf
    v[2] = np.inf
    v[3] = -np.inf
    _assert_dispatched_equal((x, v), {})


def test_dispatch_float64_queries_outside_haystack_range():
    x = _sorted_float(50_000, seed=11, lo=0.0, hi=1000.0)
    v = _random_float(50_000, seed=12, lo=0.0, hi=1000.0).copy()
    v[:100] = -5000.0  # below x.min()
    v[100:200] = 5000.0  # above x.max()
    _assert_dispatched_equal((x, v), {})


# ---------------------------------------------------------------------------
# 2. refusal routes
# ---------------------------------------------------------------------------

def test_refusal_sorted_v():
    x = _sorted_float(50_000, seed=13)
    v = np.sort(_random_float(50_000, seed=14))
    _assert_refused_equal((x, v), {})


def test_refusal_descending_v():
    x = _sorted_float(50_000, seed=15)
    v = np.sort(_random_float(50_000, seed=16))[::-1]
    _assert_refused_equal((x, v), {})


def test_refusal_nearly_sorted_v():
    x = _sorted_float(50_000, seed=17)
    v = np.sort(_random_float(50_000, seed=18))
    rng = np.random.default_rng(19)
    n_swaps = v.size // 100  # ~1% of adjacent pairs swapped
    swap_starts = rng.choice(v.size - 1, size=n_swaps, replace=False)
    for i in swap_starts:
        v[i], v[i + 1] = v[i + 1], v[i]
    _assert_refused_equal((x, v), {})


def test_refusal_v_below_float_floor():
    x = _sorted_float(50_000, seed=20)
    v = _random_float(5_000, seed=21)
    _assert_refused_equal((x, v), {})


def test_refusal_int64_at_a_size_float64_would_take():
    # 50_000 clears the float64 floor (10_000) and int64 is refused anyway:
    # the dtype table is a membership test now, not just a per-dtype size.
    x = _sorted_int(50_000, seed=22)
    v = _random_int(50_000, seed=23)
    _assert_refused_equal((x, v), {})


def test_refusal_x_below_haystack_floor():
    x = _sorted_float(5_000, seed=24)
    v = _random_float(50_000, seed=25)
    _assert_refused_equal((x, v), {})


def test_refusal_skew_cap_exceeded():
    x = _sorted_float(10_000, seed=26)
    v = _random_float(1_000_000, seed=27)
    _assert_refused_equal((x, v), {})


def test_refusal_sorter_kwarg():
    x = _random_float(50_000, seed=28)  # unsorted; sorter= corrects it
    sorter = np.argsort(x)
    v = _random_float(50_000, seed=29)
    _assert_refused_equal((x, v), {"sorter": sorter})


def test_refusal_bad_side_string_raises_both():
    x = _sorted_float(50_000, seed=30)
    v = _random_float(50_000, seed=31)
    _assert_refused_raises((x, v, "wrong"), {})


def test_refusal_scalar_v():
    x = _sorted_float(50_000, seed=32)
    _assert_refused_equal((x, 0.5), {})


def test_refusal_2d_v():
    x = _sorted_float(50_000, seed=33)
    v = _random_float(50_000, seed=34).reshape(500, 100)
    _assert_refused_equal((x, v), {})


def test_refusal_mixed_dtypes():
    x = _sorted_float(50_000, seed=35)
    v = _random_int(50_000, seed=36)
    _assert_refused_equal((x, v), {})


def test_refusal_float32_arrays():
    x = _sorted_float(50_000, seed=37).astype(np.float32)
    v = _random_float(50_000, seed=38).astype(np.float32)
    _assert_refused_equal((x, v), {})


def test_kill_switch_restores_stock_routing():
    x = _sorted_float(50_000, seed=39)
    v = _random_float(50_000, seed=40)
    decision, reason = GEARBOX.decide(OP, (x, v), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (x, v), {})
        assert decision == "stock", (decision, reason)
        got = np.searchsorted(x, v)
        stock = _stock(x, v)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)


# ---------------------------------------------------------------------------
# 3. unsorted haystack: REFUSED now, and this is the record of why.
#
#    The original contract was "garbage in, the same garbage out" - an
#    unsorted haystack would dispatch and return exactly what stock
#    returns. That was verified NOT to hold: stock's own searchsorted on an
#    unsorted haystack is query-BATCH-ORDER dependent (confirmed by
#    comparing scalar-per-element calls, which match a manual textbook
#    bisect_left, against a batched call, which does not; an internal
#    locality hint chained between consecutive queries -- correct on a
#    truly sorted haystack, order-sensitive on an unsorted one). The
#    fastpath reorders v via argsort, so its unpermuted output diverges
#    from stock's call on the original order: 17122/20000 elements on numpy
#    2.4.5, and 43407/50000 measured again in batch 16.
#
#    That was known and ACCEPTED for a while - dispatch happened anyway,
#    with the bit-identity half held as a strict xfail below numpy 2.5,
#    which is where the order dependence was removed. Batch 16 reversed the
#    call. The reasoning is not that the old one was unreasonable, since
#    this is undefined behaviour on numpy's side either way; it is that the
#    package floor is numpy>=2.3, so the divergent versions are SUPPORTED
#    versions, and this project had already ruled the other way on the same
#    shape (isin_string_hash refuses lone-NUL strings so stock keeps
#    answering where stock is quirky). Two opposite decisions about
#    undefined behaviour is one too many.
#
#    The guard costs 0.1-0.5% of the call it protects, so the trade is not
#    close. On numpy >= 2.5 it refuses an input that would have agreed.
# ---------------------------------------------------------------------------

def test_unsorted_haystack_now_refused_rather_than_matched():
    """SUPERSEDED, and the history is the point.

    This used to assert that an unsorted haystack DISPATCHES and returns
    stock's garbage exactly, under a strict xfail for numpy < 2.5 because
    the batched result is query-order dependent there (17122/20000 measured
    on 2.4.5; 0/20000 on 2.5.2). That contract was "garbage in, the same
    garbage out", and it held only on the newest numpy.

    The predicate now checks the haystack, so the question does not arise:
    an unsorted haystack goes to stock on every version. Kept as a refusal
    test rather than deleted because it is the exact shape that used to
    dispatch - if the check is ever removed, this fails and the reviewer
    reads the paragraph above instead of rediscovering it.
    """
    x = _random_float(50_000, seed=41)  # deliberately NOT sorted
    v = _random_float(50_000, seed=42)
    decision, reason = GEARBOX.decide(OP, (x, v), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((x, v), {})


# ---------------------------------------------------------------------------
# 4. empty haystack edge
# ---------------------------------------------------------------------------

def test_empty_haystack_refuses_and_matches_stock():
    x = np.array([], dtype=np.float64)
    v = _random_float(50_000, seed=43)
    got = _assert_refused_equal((x, v), {})
    assert np.all(got == 0)


# ---------------------------------------------------------------------------
# 5. the haystack must actually BE a haystack
# ---------------------------------------------------------------------------
# numpy documents `a` as needing to be sorted and does not check it, so an
# unsorted haystack gets a meaningless answer rather than an error. The
# answer is meaningless but not arbitrary - it is a deterministic function
# of the array - and this path returned a DIFFERENT one, because sorting
# the queries is only order-neutral when numpy's progressive narrowing of
# the search range is valid, which it is not on an unsorted haystack. It
# dispatched, and 43,407 of 50,000 positions disagreed with stock.

def test_refusal_unsorted_haystack():
    x = _random_float(50_000, seed=301)          # NOT sorted
    v = _random_float(50_000, seed=302)
    assert not bool(np.all(x[1:] >= x[:-1]))
    _assert_refused_equal((x, v), {})


def test_unsorted_haystack_divergence_tracks_the_numpy_boundary():
    """Is the refusal load-bearing, or precautionary? It depends on numpy.

    Sorting the queries and inverting - exactly what the path does - gives
    a DIFFERENT answer from stock on numpy < 2.5, where the batched
    searchsorted narrows its range as it walks the query list and that
    narrowing is invalid on an unsorted haystack. numpy 2.5 removed the
    order dependence, so there the two agree and the guard costs a
    dispatch it did not have to refuse.

    Asserted in BOTH directions rather than xfailed in one, so the day the
    boundary moves again this says which side it moved to. The package
    floor is numpy>=2.3, so the divergent versions are supported ones and
    the guard stays either way.
    """
    x = _random_float(50_000, seed=303)
    v = _random_float(50_000, seed=304)
    order = np.argsort(v, kind="stable")
    round_trip = np.empty_like(order)
    round_trip[order] = _stock(x, v[order])
    agrees = np.array_equal(round_trip, _stock(x, v))
    if np.lib.NumpyVersion(np.__version__) < "2.5.0":
        assert not agrees, (
            "numpy < 2.5 used to be query-order dependent on an unsorted "
            "haystack; if that is gone, re-measure the guard's value")
    else:
        assert agrees, (
            "numpy >= 2.5 removed the order dependence; a divergence here "
            "means it is back and the guard is load-bearing again")


def test_dispatch_haystack_sorted_with_duplicates():
    """Non-decreasing is the bar, not strictly increasing: searchsorted is
    defined on haystacks with repeats and this is not a uniqueness check."""
    rng = np.random.default_rng(305)
    x = np.sort(rng.integers(0, 50, size=50_000).astype(np.float64))
    v = rng.integers(0, 50, size=50_000).astype(np.float64)
    assert x.size > np.unique(x).size
    _assert_dispatched_equal((x, v), {})


def test_refusal_haystack_containing_nan():
    """NaN compares False in both directions, so a NaN anywhere in the
    haystack reads as unsorted and goes to stock. Conservative on purpose -
    it costs a dispatch and never an answer."""
    x = np.concatenate([_sorted_float(49_999, seed=306), [np.nan]])
    v = _random_float(50_000, seed=307)
    _assert_refused_equal((x, v), {})
