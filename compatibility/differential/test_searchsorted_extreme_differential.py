"""Differential tests: searchsorted_extreme_key fast path vs stock numpy.searchsorted.

Contract (src/pyoverdrive/fastpaths/searchsorted_extreme_key.py): applies only
to searchsorted(a, v[, side]) where a is a plain 1-D integer-dtype ndarray
(dtype.kind 'i' or 'u'), v is a Python int (bool excluded) strictly outside
a.dtype's representable range (v < iinfo(a.dtype).min or v > iinfo(a.dtype).max),
side is absent/'left'/'right' (positional 3rd arg or keyword), and no sorter=.
Returns np.intp(a.size) for v above the range, np.intp(0) below - an
order-theoretic proof, true for any array contents including unsorted ones,
since every representable value compares on one uniform side of an
out-of-range key. Comparison mode: bit-identical, including the returned
scalar type.

The op "numpy.searchsorted" also carries the searchsorted_sortqueries fast
path (array-valued v, large disordered query batches). This file tests only
extreme_key's own predicate: array-valued-v cases assert dispatch is NOT this
path (they may land on stock or on searchsorted_sortqueries).
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

OP = "numpy.searchsorted"
PATH = "searchsorted_extreme_key"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _arr(n, dtype, seed=1, lo=-1000, hi=1000):
    dtype = np.dtype(dtype)
    if dtype.kind == "u":
        lo = max(lo, 0)
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, size=n).astype(dtype)


def _call(fn, args, kwargs):
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as e:
        return "err", e


def _assert_dispatched_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.searchsorted(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert got.dtype == stock.dtype
    assert got == stock
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got_kind, got = _call(np.searchsorted, args, kwargs)
    stock_kind, stock = _call(_stock, args, kwargs)
    assert got_kind == stock_kind, (got_kind, stock_kind, got, stock)
    if got_kind == "ok":
        assert type(got) is type(stock)
        assert np.array_equal(got, stock)
    else:
        assert type(got) is type(stock)
    return got if got_kind == "ok" else None


def _assert_not_this_path(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision != PATH, (args, kwargs, decision, reason)
    got_kind, got = _call(np.searchsorted, args, kwargs)
    stock_kind, stock = _call(_stock, args, kwargs)
    assert got_kind == stock_kind, (got_kind, stock_kind, got, stock)
    if got_kind == "ok":
        assert type(got) is type(stock)
        assert np.array_equal(got, stock)


# ---------------------------------------------------------------------------
# 1. dispatch + bit-identity
# ---------------------------------------------------------------------------


def test_dispatch_int64_above_range_key_2p70():
    a = _arr(500, np.int64, seed=1)
    got, stock = _assert_dispatched_equal((a, 2**70), {})
    assert got == a.size
    assert type(got) is type(stock) is np.intp


def test_dispatch_int64_below_range_key_neg_2p70():
    a = _arr(500, np.int64, seed=2)
    got, stock = _assert_dispatched_equal((a, -(2**70)), {})
    assert got == 0


def test_dispatch_side_left_kwarg():
    a = _arr(500, np.int64, seed=3)
    got, _ = _assert_dispatched_equal((a, 2**70), {"side": "left"})
    assert got == a.size


def test_dispatch_side_right_kwarg():
    a = _arr(500, np.int64, seed=4)
    got, _ = _assert_dispatched_equal((a, 2**70), {"side": "right"})
    assert got == a.size


def test_dispatch_side_positional():
    a = _arr(500, np.int64, seed=5)
    got, _ = _assert_dispatched_equal((a, -(2**70), "right"), {})
    assert got == 0


def test_dispatch_int32_array_key_2p40_within_int64():
    # 2**40 exceeds int32's range but is well within int64's range - it is
    # still out-of-range for THIS array's dtype, so the path dispatches.
    a = _arr(500, np.int32, seed=6)
    got, stock = _assert_dispatched_equal((a, 2**40), {})
    assert got == a.size
    assert type(got) is type(stock) is np.intp


def test_dispatch_uint64_array_key_negative_5():
    a = _arr(500, np.uint64, seed=7, lo=0, hi=2000)
    got, stock = _assert_dispatched_equal((a, -5), {})
    assert got == 0
    assert type(got) is type(stock) is np.intp


def test_dispatch_int8_array_key_300():
    a = _arr(500, np.int8, seed=8, lo=-100, hi=100)
    got, stock = _assert_dispatched_equal((a, 300), {})
    assert got == a.size


def test_dispatch_empty_int64_array_above_range():
    a = np.array([], dtype=np.int64)
    got, stock = _assert_dispatched_equal((a, 2**70), {})
    assert got == 0
    assert stock == 0


def test_dispatch_empty_int64_array_below_range():
    a = np.array([], dtype=np.int64)
    got, stock = _assert_dispatched_equal((a, -(2**70)), {})
    assert got == 0
    assert stock == 0


def test_dispatch_unsorted_array_above_range_matches_stock():
    # Provability claim: the fast path's answer equals stock's even though
    # a is not sorted, because both routes reduce to a uniform per-element
    # comparison against a key outside the dtype's whole representable range.
    a = _arr(500, np.int64, seed=9)
    assert not np.all(a[:-1] <= a[1:])  # sanity: genuinely unsorted
    got, stock = _assert_dispatched_equal((a, 2**70), {})
    assert got == a.size == stock


def test_dispatch_unsorted_array_below_range_matches_stock():
    a = _arr(500, np.int64, seed=10)
    assert not np.all(a[:-1] <= a[1:])
    got, stock = _assert_dispatched_equal((a, -(2**70)), {})
    assert got == 0 == stock


def test_dispatch_key_exactly_iinfo_max_plus_one():
    a = _arr(500, np.int64, seed=11)
    hi = np.iinfo(np.int64).max
    got, stock = _assert_dispatched_equal((a, hi + 1), {})
    assert got == a.size == stock


def test_dispatch_key_exactly_iinfo_min_minus_one():
    a = _arr(500, np.int64, seed=12)
    lo = np.iinfo(np.int64).min
    got, stock = _assert_dispatched_equal((a, lo - 1), {})
    assert got == 0 == stock


# ---------------------------------------------------------------------------
# 2. refusal routes (parity with stock)
# ---------------------------------------------------------------------------


def test_refusal_key_exactly_iinfo_max_in_range():
    a = _arr(500, np.int64, seed=13)
    hi = np.iinfo(np.int64).max
    _assert_refused_equal((a, hi), {})


def test_refusal_key_exactly_iinfo_min_in_range():
    a = _arr(500, np.int64, seed=14)
    lo = np.iinfo(np.int64).min
    _assert_refused_equal((a, lo), {})


def test_refusal_in_range_python_int():
    a = _arr(500, np.int64, seed=15)
    _assert_refused_equal((a, 7), {})


def test_refusal_np_int64_key_not_python_int():
    # np.int64 is not a subclass of Python int - excluded by the predicate's
    # isinstance(v, int) check regardless of how far out of range it is.
    a = _arr(500, np.int8, seed=16, lo=-100, hi=100)
    _assert_refused_equal((a, np.int64(300)), {})


def test_refusal_float_key_huge():
    a = _arr(500, np.int64, seed=17)
    _assert_refused_equal((a, 1e30), {})


def test_refusal_bool_key_true():
    a = _arr(500, np.int64, seed=18)
    _assert_refused_equal((a, True), {})


def test_refusal_float64_array_huge_int_key():
    a = _arr(500, np.int64, seed=19).astype(np.float64) / 3.0
    _assert_refused_equal((a, 2**70), {})


def test_refusal_sorter_kwarg():
    a = np.sort(_arr(500, np.int64, seed=20))
    sorter = np.arange(a.size)
    _assert_refused_equal((a, 2**70), {"sorter": sorter})


def test_refusal_array_valued_v_not_this_path():
    a = _arr(500, np.int64, seed=21)
    v = _arr(20, np.int64, seed=22)
    _assert_not_this_path((a, v), {})


# ---------------------------------------------------------------------------
# 3. kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_restores_stock_routing():
    a = _arr(500, np.int64, seed=23)
    decision, reason = GEARBOX.decide(OP, (a, 2**70), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (a, 2**70), {})
        assert decision == "stock", (decision, reason)
        got = np.searchsorted(a, 2**70)
        stock = _stock(a, 2**70)
        assert got == stock
    finally:
        pyoverdrive.enable_path(PATH)
