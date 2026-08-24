"""Differential tests: nanreduce_scan fast paths vs stock numpy nan-reducers.

Covers all four registered paths (nanmean_scan, nansum_scan, nanstd_scan,
nanvar_scan): clean-input dispatch (bit-identical to stock), NaN-present
dispatch (internal fallback to stock, result and warnings identical to
calling stock directly), and the refusal surface documented in the module
docstring (dtype, floor, axis shape, keepdims/dtype/out kwargs, empty
input, zero-length reduced axis).
"""

import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

OPS = [
    ("numpy.nanmean", "nanmean_scan", 100, np.nanmean),
    ("numpy.nansum", "nansum_scan", 10_000, np.nansum),
    ("numpy.nanstd", "nanstd_scan", 3_000, np.nanstd),
    ("numpy.nanvar", "nanvar_scan", 3_000, np.nanvar),
]
OP_IDS = [op for op, _, _, _ in OPS]

SHAPE2D = {
    "numpy.nanmean": (15, 20),
    "numpy.nansum": (30, 500),
    "numpy.nanstd": (40, 100),
    "numpy.nanvar": (40, 100),
}


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([op for op, _, _, _ in OPS])
    yield
    pyoverdrive.disable()


def _capture(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [(w.category, str(w.message)) for w in rec]


def _assert_dispatched_exact(op, path, npfn, args, kwargs):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == path, (op, decision, reason)
    got, got_warn = _capture(npfn, *args, **kwargs)
    stock, stock_warn = _capture(GEARBOX.stock_fn(op), *args, **kwargs)
    assert got_warn == stock_warn
    assert np.asarray(got).dtype == np.asarray(stock).dtype
    assert np.asarray(got).shape == np.asarray(stock).shape
    assert np.array_equal(got, stock, equal_nan=True)


def _compare_call(npfn, stockfn, args, kwargs):
    np_exc = stock_exc = None
    got = got_warn = stock = stock_warn = None
    try:
        got, got_warn = _capture(npfn, *args, **kwargs)
    except Exception as exc:
        np_exc = exc
    try:
        stock, stock_warn = _capture(stockfn, *args, **kwargs)
    except Exception as exc:
        stock_exc = exc
    if np_exc is not None or stock_exc is not None:
        assert type(np_exc) is type(stock_exc), (np_exc, stock_exc)
        assert str(np_exc) == str(stock_exc)
    else:
        assert got_warn == stock_warn
        assert np.array_equal(got, stock, equal_nan=True)


def _assert_refused_exact(op, npfn, args, kwargs):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == "stock", (op, decision, reason)
    _compare_call(npfn, GEARBOX.stock_fn(op), args, kwargs)


# -- 1: clean 1-D above floor ------------------------------------------------

@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_dispatch_1d_clean_bit_identical(op, path, floor, npfn):
    rng = np.random.default_rng(1)
    a = rng.standard_normal(floor + 50)
    _assert_dispatched_exact(op, path, npfn, (a,), {})


# -- 2: clean 2-D, axis kwarg and axis positional -----------------------------

@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_dispatch_2d_axis_kwarg_and_positional(op, path, floor, npfn):
    rng = np.random.default_rng(2)
    a = rng.standard_normal(SHAPE2D[op])
    _assert_dispatched_exact(op, path, npfn, (a,), {"axis": 1})
    _assert_dispatched_exact(op, path, npfn, (a, 1), {})


# -- 3: NaN-present (not all-NaN), decision still the path --------------------

@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_dispatch_partial_nan(op, path, floor, npfn):
    rng = np.random.default_rng(3)
    a = rng.standard_normal(floor + 50)
    a[::7] = np.nan
    assert np.isnan(a).any() and not np.isnan(a).all()
    _assert_dispatched_exact(op, path, npfn, (a,), {})


# -- 4: all-NaN above floor: dispatches, warnings identical to stock ----------

@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_dispatch_all_nan(op, path, floor, npfn):
    a = np.full(floor + 50, np.nan, dtype=np.float64)
    decision, reason = GEARBOX.decide(op, (a,), {})
    assert decision == path, (op, decision, reason)
    got, got_warn = _capture(npfn, a)
    stock, stock_warn = _capture(GEARBOX.stock_fn(op), a)
    assert got_warn == stock_warn
    assert np.array_equal(got, stock, equal_nan=True)
    if op == "numpy.nanmean":
        assert any("empty slice" in msg.lower() for _, msg in got_warn)


# -- 5: refusals ---------------------------------------------------------------

@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_refusal_float32(op, path, floor, npfn):
    rng = np.random.default_rng(4)
    a = rng.standard_normal(floor + 50).astype(np.float32)
    _assert_refused_exact(op, npfn, (a,), {})


@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_refusal_int64(op, path, floor, npfn):
    rng = np.random.default_rng(5)
    a = rng.integers(-1000, 1000, size=floor + 50, dtype=np.int64)
    _assert_refused_exact(op, npfn, (a,), {})


@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_refusal_below_floor(op, path, floor, npfn):
    rng = np.random.default_rng(6)
    a = rng.standard_normal(floor - 1)
    _assert_refused_exact(op, npfn, (a,), {})


@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_refusal_keepdims(op, path, floor, npfn):
    rng = np.random.default_rng(7)
    a = rng.standard_normal(floor + 50)
    _assert_refused_exact(op, npfn, (a,), {"keepdims": True})


@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_refusal_dtype_kwarg(op, path, floor, npfn):
    rng = np.random.default_rng(8)
    a = rng.standard_normal(floor + 50)
    _assert_refused_exact(op, npfn, (a,), {"dtype": np.float32})


@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_refusal_out_kwarg(op, path, floor, npfn):
    rng = np.random.default_rng(9)
    a = rng.standard_normal(floor + 50)
    decision, reason = GEARBOX.decide(op, (a,), {"out": np.array(0.0)})
    assert decision == "stock", (op, decision, reason)
    out_np = np.array(0.0)
    out_stock = np.array(0.0)
    got, got_warn = _capture(npfn, a, out=out_np)
    stock, stock_warn = _capture(GEARBOX.stock_fn(op), a, out=out_stock)
    assert got_warn == stock_warn
    assert np.array_equal(out_np, out_stock, equal_nan=True)
    assert np.array_equal(got, stock, equal_nan=True)


@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_refusal_axis_tuple(op, path, floor, npfn):
    rng = np.random.default_rng(10)
    a = rng.standard_normal(SHAPE2D[op])
    _assert_refused_exact(op, npfn, (a,), {"axis": (0, 1)})


@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_refusal_empty_array(op, path, floor, npfn):
    a = np.array([], dtype=np.float64)
    decision, reason = GEARBOX.decide(op, (a,), {})
    assert decision == "stock", (op, decision, reason)
    _compare_call(npfn, GEARBOX.stock_fn(op), (a,), {})


@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_refusal_zero_length_reduced_axis(op, path, floor, npfn):
    a = np.empty((5, 0), dtype=np.float64)
    decision, reason = GEARBOX.decide(op, (a,), {"axis": 1})
    assert decision == "stock", (op, decision, reason)
    _compare_call(npfn, GEARBOX.stock_fn(op), (a,), {"axis": 1})


# -- 6: negative axis on 2-D ----------------------------------------------------

@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_dispatch_negative_axis(op, path, floor, npfn):
    rng = np.random.default_rng(11)
    a = rng.standard_normal(SHAPE2D[op])
    _assert_dispatched_exact(op, path, npfn, (a,), {"axis": -1})


# -- 7: kill switch --------------------------------------------------------------

@pytest.mark.parametrize("op,path,floor,npfn", OPS, ids=OP_IDS)
def test_kill_switch_restores_stock_routing(op, path, floor, npfn):
    rng = np.random.default_rng(12)
    a = rng.standard_normal(floor + 50)
    decision, _ = GEARBOX.decide(op, (a,), {})
    assert decision == path
    pyoverdrive.disable_path(path)
    try:
        decision, _ = GEARBOX.decide(op, (a,), {})
        assert decision == "stock"
        assert np.array_equal(npfn(a), GEARBOX.stock_fn(op)(a))
    finally:
        pyoverdrive.enable_path(path)
