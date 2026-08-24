"""Differential tests: median_partition fast path vs stock numpy.median.

Contract (src/pyoverdrive/fastpaths/median_partition.py): applies only to
median(a[, axis]) where a is a plain 1-D float64 ndarray, axis absent or
None, no other kwargs, and SIZE_MIN <= a.size <= SIZE_CAP. n-D input,
other dtypes, axis given, out/overwrite_input/keepdims, and sizes outside
the measured band stay on stock.

Comparison mode: bit-identical (spec section 9), including the returned
SCALAR TYPE: stock numpy.median on a 1-D float64 array returns a plain
np.float64 scalar, and the fast path must match that exactly (not just
the numeric value). NaN propagation and stock's own warning behavior
(none expected on NaN input; a warning IS expected on an empty array,
which naturally refuses on size) are pinned as well.
"""

import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.median_partition import SIZE_CAP, SIZE_MIN

OP = "numpy.median"
PATH = "median_partition"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _arr(n, seed=1, low=-500.0, high=500.0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) * (high - low) + low).astype(np.float64)


def _assert_scalar_equal(got, stock):
    assert type(got) is np.float64, type(got)
    assert type(stock) is np.float64, type(stock)
    if np.isnan(stock):
        assert np.isnan(got)
    else:
        assert got == stock
        assert got.tobytes() == stock.tobytes()


def _assert_value_type_equal(got, stock):
    # Generic comparator for refusal-parity cases: stock's return can be
    # a scalar np.float64 or an ndarray (keepdims, out) depending on the
    # kwargs used, so branch on what stock actually produced.
    assert type(got) is type(stock)
    if isinstance(got, np.ndarray):
        assert got.dtype == stock.dtype
        assert got.shape == stock.shape
        assert np.array_equal(got, stock, equal_nan=True)
    elif np.isnan(stock):
        assert np.isnan(got)
    else:
        assert got == stock


def _assert_dispatched_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.median(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    _assert_scalar_equal(got, stock)
    return got, stock


def _assert_dispatched_with_warnings(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    with warnings.catch_warnings(record=True) as got_w:
        warnings.simplefilter("always")
        got = np.median(*args, **kwargs)
    with warnings.catch_warnings(record=True) as stock_w:
        warnings.simplefilter("always")
        stock = _stock(*args, **kwargs)
    _assert_scalar_equal(got, stock)
    assert len(got_w) == len(stock_w), (got_w, stock_w)
    assert [w.category for w in got_w] == [w.category for w in stock_w]
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.median(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    _assert_value_type_equal(got, stock)
    return got


def _assert_refused_with_warnings(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with warnings.catch_warnings(record=True) as got_w:
        warnings.simplefilter("always")
        got = np.median(*args, **kwargs)
    with warnings.catch_warnings(record=True) as stock_w:
        warnings.simplefilter("always")
        stock = _stock(*args, **kwargs)
    _assert_value_type_equal(got, stock)
    assert len(got_w) == len(stock_w), (got_w, stock_w)
    assert [w.category for w in got_w] == [w.category for w in stock_w]
    return got


# ---------------------------------------------------------------------------
# 1. dispatch + bit-identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [SIZE_MIN, 11, 100, 1001, 5000, SIZE_CAP])
def test_dispatch_bit_identical_across_band(n):
    a = _arr(n, seed=1000 + n)
    _assert_dispatched_equal((a,), {})


def test_dispatch_explicit_axis_none_kwarg():
    a = _arr(101, seed=2001)
    _assert_dispatched_equal((a,), {"axis": None})


@pytest.mark.parametrize("n", [500, 501])
def test_dispatch_duplicate_heavy(n):
    rng = np.random.default_rng(3000 + n)
    a = rng.choice(np.array([1.0, 2.0, 3.0], dtype=np.float64), size=n).astype(np.float64)
    _assert_dispatched_equal((a,), {})


@pytest.mark.parametrize("n", [300, 301])
def test_dispatch_already_sorted(n):
    a = np.arange(n, dtype=np.float64)
    _assert_dispatched_equal((a,), {})


@pytest.mark.parametrize("n", [400, 401])
def test_dispatch_plus_minus_inf_values(n):
    a = _arr(n, seed=4000 + n)
    a[0] = np.inf
    a[1] = -np.inf
    a[2] = np.inf
    _assert_dispatched_equal((a,), {})


@pytest.mark.parametrize("n", [100, 101])
def test_dispatch_nan_salted_matches_warnings(n):
    a = _arr(n, seed=5000 + n)
    a[5] = np.nan
    got, stock = _assert_dispatched_with_warnings((a,), {})
    assert np.isnan(got)
    assert np.isnan(stock)


@pytest.mark.parametrize("n", [200, 201])
def test_dispatch_all_equal(n):
    a = np.full(n, 7.5, dtype=np.float64)
    _assert_dispatched_equal((a,), {})


def test_dispatch_size_equals_size_cap():
    a = _arr(SIZE_CAP, seed=6001)
    _assert_dispatched_equal((a,), {})


def test_refusal_size_cap_plus_one():
    a = _arr(SIZE_CAP + 1, seed=6002)
    _assert_refused_equal((a,), {})


def test_dispatch_size_equals_size_min():
    a = _arr(SIZE_MIN, seed=6003)
    _assert_dispatched_equal((a,), {})


def test_refusal_size_min_minus_one():
    a = _arr(SIZE_MIN - 1, seed=6004)
    _assert_refused_equal((a,), {})


# ---------------------------------------------------------------------------
# 2. refusal routes, with parity
# ---------------------------------------------------------------------------


def test_refusal_2d_array():
    a = _arr(200, seed=7001).reshape(20, 10)
    _assert_refused_equal((a,), {})


def test_refusal_float32_dtype():
    a = _arr(200, seed=7002).astype(np.float32)
    _assert_refused_equal((a,), {})


def test_refusal_int64_dtype_returns_float():
    # Stock median of an int array returns a float scalar; this is a
    # dtype-parity check, not just a value check.
    a = np.random.default_rng(7003).integers(-1000, 1000, size=200).astype(np.int64)
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision == "stock", (decision, reason)
    got = np.median(a)
    stock = _stock(a)
    assert type(got) is type(stock)
    assert np.issubdtype(type(stock), np.floating)
    assert got == stock


def test_refusal_axis_0_kwarg():
    a = _arr(200, seed=7004)
    _assert_refused_equal((a,), {"axis": 0})


def test_refusal_keepdims_true():
    a = _arr(200, seed=7005)
    _assert_refused_equal((a,), {"keepdims": True})


def test_refusal_out_kwarg():
    a = _arr(200, seed=7006)
    out_got = np.empty((), dtype=np.float64)
    out_stock = np.empty((), dtype=np.float64)
    decision, reason = GEARBOX.decide(OP, (a,), {"out": out_got})
    assert decision == "stock", (decision, reason)
    got = np.median(a, out=out_got)
    stock = _stock(a, out=out_stock)
    assert got is out_got
    _assert_value_type_equal(got, stock)


def test_refusal_overwrite_input_true():
    # overwrite_input is not in the {"axis"} kwargs whitelist, so this is
    # refused purely on kwargs shape, independent of the boolean value.
    base = _arr(200, seed=7007)
    a_got = base.copy()
    a_stock = base.copy()
    decision, reason = GEARBOX.decide(OP, (a_got,), {"overwrite_input": True})
    assert decision == "stock", (decision, reason)
    got = np.median(a_got, overwrite_input=True)
    stock = _stock(a_stock, overwrite_input=True)
    _assert_value_type_equal(got, stock)


def test_refusal_python_list_input():
    a = [3.0, 1.0, 2.0, 5.0, 4.0, 9.0, 8.0, 7.0, 6.0, 0.0, 10.0]
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision == "stock", (decision, reason)
    got = np.median(a)
    stock = _stock(a)
    _assert_value_type_equal(got, stock)


def test_refusal_empty_array_warns_and_nan():
    a = np.array([], dtype=np.float64)
    got = _assert_refused_with_warnings((a,), {})
    assert np.isnan(got)


# ---------------------------------------------------------------------------
# 3. kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_restores_stock_routing():
    a = _arr(500, seed=8001)
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (a,), {})
        assert decision == "stock", (decision, reason)
        got = np.median(a)
        stock = _stock(a)
        _assert_scalar_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)
