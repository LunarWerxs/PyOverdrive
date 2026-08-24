"""Differential tests: nanargmax_scan / nanargmin_scan fast paths vs stock
numpy.nanargmax / numpy.nanargmin.

Both paths share one contract (src/pyoverdrive/fastpaths/nanargminmax_scan.py):
a single isnan probe over the whole array (np.isnan(np.min(a))) selects plain
argmax/argmin for NaN-free input, or falls back to stock nanargmax/nanargmin
(index and exceptions, including the all-NaN-slice ValueError) when any NaN
is present anywhere in the array.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

OPS = [
    ("numpy.nanargmax", "nanargmax_scan", "nanargmax"),
    ("numpy.nanargmin", "nanargmin_scan", "nanargmin"),
]
OP_NAMES = [op for op, _, _ in OPS]
OP_IDS = [np_name for _, _, np_name in OPS]


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(OP_NAMES)
    yield
    pyoverdrive.disable()


def _dispatched(np_name):
    return getattr(np, np_name)


def _stock(op, *args, **kwargs):
    return GEARBOX.stock_fn(op)(*args, **kwargs)


def _assert_dispatched_exact(op, path, np_name, args, kwargs):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == path, (decision, reason)
    got = _dispatched(np_name)(*args, **kwargs)
    stock = _stock(op, *args, **kwargs)
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.array_equal(got, stock)


def _assert_refused_exact(op, np_name, args, kwargs):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == "stock", (decision, reason)
    got = _dispatched(np_name)(*args, **kwargs)
    stock = _stock(op, *args, **kwargs)
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.array_equal(got, stock)


def _assert_both_raise_valueerror(op, path, np_name, args, kwargs, expect_path):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    if expect_path:
        assert decision == path, (decision, reason)
    else:
        assert decision == "stock", (decision, reason)
    with pytest.raises(ValueError) as dispatched_exc:
        _dispatched(np_name)(*args, **kwargs)
    with pytest.raises(ValueError) as stock_exc:
        _stock(op, *args, **kwargs)
    assert dispatched_exc.type is stock_exc.type


# -- 1. clean 1-D --------------------------------------------------------


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_clean_1d_dispatch(op, path, np_name):
    rng = np.random.default_rng(1)
    a = rng.standard_normal(5000)
    _assert_dispatched_exact(op, path, np_name, (a,), {})


# -- 2. clean 2-D, axis variants ------------------------------------------


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_clean_2d_axis_variants(op, path, np_name):
    rng = np.random.default_rng(2)
    a = rng.standard_normal((100, 50))
    _assert_dispatched_exact(op, path, np_name, (a,), {"axis": 0})
    _assert_dispatched_exact(op, path, np_name, (a,), {"axis": 1})
    _assert_dispatched_exact(op, path, np_name, (a,), {"axis": -1})
    _assert_dispatched_exact(op, path, np_name, (a, 1), {})  # positional axis


# -- 3. duplicate-extreme ties: first-occurrence index --------------------


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_duplicate_extreme_first_occurrence(op, path, np_name):
    rng = np.random.default_rng(3)
    a = rng.standard_normal(1000)
    extreme = 1000.0 if np_name == "nanargmax" else -1000.0
    a[[50, 300, 700]] = extreme
    _assert_dispatched_exact(op, path, np_name, (a,), {})


# -- 4. NaN present: still dispatches, matches stock -----------------------


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_nan_present_1d(op, path, np_name):
    rng = np.random.default_rng(4)
    a = rng.standard_normal(2000)
    a[500] = np.nan
    _assert_dispatched_exact(op, path, np_name, (a,), {})


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_nan_present_2d(op, path, np_name):
    rng = np.random.default_rng(5)
    a = rng.standard_normal((100, 50))
    a[10, 20] = np.nan
    _assert_dispatched_exact(op, path, np_name, (a,), {"axis": 1})


# -- 5. all-NaN 1-D: raises ValueError like stock --------------------------


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_all_nan_1d_raises(op, path, np_name):
    a = np.full(400, np.nan, dtype=np.float64)
    _assert_both_raise_valueerror(op, path, np_name, (a,), {}, expect_path=True)


# -- 6. 2-D with one all-NaN row, axis=1: raises ValueError like stock ----


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_2d_one_all_nan_row_axis1_raises(op, path, np_name):
    rng = np.random.default_rng(6)
    a = rng.standard_normal((100, 50))
    a[7, :] = np.nan
    _assert_both_raise_valueerror(
        op, path, np_name, (a,), {"axis": 1}, expect_path=True
    )


# -- 7. refusals: decision "stock", equality still holds -------------------


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_refusal_float32(op, path, np_name):
    rng = np.random.default_rng(7)
    a = rng.standard_normal(1000).astype(np.float32)
    _assert_refused_exact(op, np_name, (a,), {})


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_refusal_int64(op, path, np_name):
    rng = np.random.default_rng(8)
    a = rng.integers(-1000, 1000, size=1000).astype(np.int64)
    _assert_refused_exact(op, np_name, (a,), {})


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_refusal_size_below_floor(op, path, np_name):
    rng = np.random.default_rng(9)
    a = rng.standard_normal(200)  # SIZE_FLOOR is 300
    _assert_refused_exact(op, np_name, (a,), {})


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_refusal_out_kwarg(op, path, np_name):
    rng = np.random.default_rng(10)
    a = rng.standard_normal((100, 50))
    decision, reason = GEARBOX.decide(op, (a,), {"axis": 1, "out": np.empty(100, dtype=np.intp)})
    assert decision == "stock", (decision, reason)
    got_out = np.empty(100, dtype=np.intp)
    got = _dispatched(np_name)(a, axis=1, out=got_out)
    stock_out = np.empty(100, dtype=np.intp)
    stock = _stock(op, a, axis=1, out=stock_out)
    assert np.array_equal(got, stock)
    assert np.array_equal(got_out, stock_out)


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_refusal_keepdims_kwarg(op, path, np_name):
    rng = np.random.default_rng(11)
    a = rng.standard_normal((100, 50))
    _assert_refused_exact(op, np_name, (a,), {"axis": 1, "keepdims": True})


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_refusal_empty_array_raises(op, path, np_name):
    a = np.array([], dtype=np.float64)
    _assert_both_raise_valueerror(op, path, np_name, (a,), {}, expect_path=False)


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_refusal_zero_length_reduced_axis_raises(op, path, np_name):
    a = np.empty((5, 0), dtype=np.float64)
    _assert_both_raise_valueerror(
        op, path, np_name, (a,), {"axis": 1}, expect_path=False
    )


# -- 8. kill switch ---------------------------------------------------------


@pytest.mark.parametrize("op,path,np_name", OPS, ids=OP_IDS)
def test_kill_switch_restores_stock_routing(op, path, np_name):
    rng = np.random.default_rng(12)
    a = rng.standard_normal(2000)
    decision, _ = GEARBOX.decide(op, (a,), {})
    assert decision == path
    pyoverdrive.disable_path(path)
    try:
        decision, _ = GEARBOX.decide(op, (a,), {})
        assert decision == "stock"
        assert np.array_equal(_dispatched(np_name)(a), _stock(op, a))
    finally:
        pyoverdrive.enable_path(path)


def test_all_nan_raises_without_spurious_path_failure_warning():
    # StockRaised protocol: an all-NaN input must raise stock's ValueError
    # with NO "fast path ... raised ... falling back" RuntimeWarning and
    # no second stock call (the path reproduced stock behavior, it did
    # not fail).
    import warnings as _w

    a = np.full(5_000, np.nan)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        with pytest.raises(ValueError):
            np.nanargmax(a)
    assert not any("fast path" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]
