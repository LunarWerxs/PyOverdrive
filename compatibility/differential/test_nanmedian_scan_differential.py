"""Differential tests: nanmedian_scan fast path vs stock numpy.nanmedian.

Covers the two-branch contract: clean 2-D C-contiguous float64 input
along axis 1/-1 dispatches to a vectorized np.median (bit-identical to
stock nanmedian on NaN-free data), and any NaN present (including an
all-NaN row) falls back internally to stock nanmedian, including its
RuntimeWarning. All the non-admissible shapes/dtypes/kwargs refuse to
stock, where stock and the monkeypatched call must still agree exactly.
"""

import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.nanmedian_scan import SIZE_FLOOR, REDUCED_LEN_CAP

OP = "numpy.nanmedian"
PATH = "nanmedian_scan"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _mk(shape, seed=1, nan_frac=0.0, all_nan_row=False):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape)
    if nan_frac:
        n = a.size
        k = int(n * nan_frac)
        idx = rng.choice(n, size=k, replace=False)
        a.reshape(-1)[idx] = np.nan
    if all_nan_row:
        a[0, :] = np.nan
    return a


def _assert_dispatched_exact(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.nanmedian(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert got.dtype == stock.dtype and got.shape == stock.shape
    assert np.array_equal(got, stock, equal_nan=True)


def _assert_dispatched_exact_with_warnings(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    with warnings.catch_warnings(record=True) as w_got:
        warnings.simplefilter("always")
        got = np.nanmedian(*args, **kwargs)
    with warnings.catch_warnings(record=True) as w_stock:
        warnings.simplefilter("always")
        stock = _stock(*args, **kwargs)
    assert got.dtype == stock.dtype and got.shape == stock.shape
    assert np.array_equal(got, stock, equal_nan=True)
    assert len(w_got) == len(w_stock)
    assert len(w_got) >= 1
    for wg, ws in zip(w_got, w_stock):
        assert wg.category is ws.category
        assert str(wg.message) == str(ws.message)
    assert any("All-NaN slice encountered" in str(w.message) for w in w_got)


def _assert_refused_exact(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (decision, reason)
    got = np.nanmedian(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert np.array_equal(got, stock, equal_nan=True)


def test_dispatch_clean_500x500_axis_kwarg_and_positional():
    a = _mk((500, 500), seed=1)
    assert a.size >= SIZE_FLOOR and a.shape[1] <= REDUCED_LEN_CAP
    _assert_dispatched_exact((a,), {"axis": 1})
    _assert_dispatched_exact((a, 1), {})
    _assert_dispatched_exact((a,), {"axis": -1})
    _assert_dispatched_exact((a, -1), {})


def test_dispatch_clean_2000x200_even_reduced_length():
    a = _mk((2000, 200), seed=2)
    _assert_dispatched_exact((a,), {"axis": 1})


def test_dispatch_clean_600x401_odd_reduced_length():
    a = _mk((600, 401), seed=3)
    assert a.size >= SIZE_FLOOR
    _assert_dispatched_exact((a,), {"axis": 1})


def test_dispatch_one_percent_nan_1000x1000():
    a = _mk((1000, 1000), seed=4, nan_frac=0.01)
    _assert_dispatched_exact((a,), {"axis": 1})


def test_dispatch_all_nan_row_matches_stock_warning():
    a = _mk((600, 400), seed=5, all_nan_row=True)
    _assert_dispatched_exact_with_warnings((a,), {"axis": 1})


def test_refusal_axis_0():
    a = _mk((600, 401), seed=6)
    _assert_refused_exact((a,), {"axis": 0})


def test_refusal_1d_array():
    a = _mk((300_000,), seed=7)
    _assert_refused_exact((a,), {"axis": 0})


def test_refusal_3d_array():
    a = _mk((10, 10, 3001), seed=8)
    _assert_refused_exact((a,), {"axis": -1})


def test_refusal_f_order_copy():
    a = np.asfortranarray(_mk((600, 401), seed=9))
    assert a.flags.f_contiguous and not a.flags.c_contiguous
    _assert_refused_exact((a,), {"axis": 1})


def test_refusal_reduced_len_over_cap():
    a = _mk((100, 2500), seed=10)
    assert a.size >= SIZE_FLOOR and a.shape[1] > REDUCED_LEN_CAP
    _assert_refused_exact((a,), {"axis": 1})


def test_refusal_size_below_floor():
    a = _mk((100, 1000), seed=11)
    assert a.size < SIZE_FLOOR
    _assert_refused_exact((a,), {"axis": 1})


def test_refusal_float32():
    a = _mk((600, 401), seed=12).astype(np.float32)
    _assert_refused_exact((a,), {"axis": 1})


def test_refusal_keepdims():
    a = _mk((600, 401), seed=13)
    _assert_refused_exact((a,), {"axis": 1, "keepdims": True})


def test_refusal_out_kwarg():
    a = _mk((600, 401), seed=14)
    decision, reason = GEARBOX.decide(OP, (a,), {"axis": 1, "out": np.empty(600)})
    assert decision == "stock", (decision, reason)
    out_got = np.empty(600, dtype=np.float64)
    out_stock = np.empty(600, dtype=np.float64)
    got = np.nanmedian(a, axis=1, out=out_got)
    stock = _stock(a, axis=1, out=out_stock)
    assert got is out_got and stock is out_stock
    assert np.array_equal(out_got, out_stock, equal_nan=True)


def test_refusal_overwrite_input():
    a = _mk((600, 401), seed=15)
    decision, reason = GEARBOX.decide(OP, (a,), {"axis": 1, "overwrite_input": True})
    assert decision == "stock", (decision, reason)
    a_got = a.copy()
    a_stock = a.copy()
    got = np.nanmedian(a_got, axis=1, overwrite_input=True)
    stock = _stock(a_stock, axis=1, overwrite_input=True)
    assert np.array_equal(got, stock, equal_nan=True)


def test_refusal_no_axis_call():
    a = _mk((600, 401), seed=16)
    _assert_refused_exact((a,), {})


def test_kill_switch_restores_stock_routing():
    a = _mk((600, 401), seed=17)
    decision, _ = GEARBOX.decide(OP, (a,), {"axis": 1})
    assert decision == PATH
    pyoverdrive.disable_path(PATH)
    try:
        decision, _ = GEARBOX.decide(OP, (a,), {"axis": 1})
        assert decision == "stock"
        assert np.array_equal(np.nanmedian(a, axis=1), _stock(a, axis=1))
    finally:
        pyoverdrive.enable_path(PATH)
