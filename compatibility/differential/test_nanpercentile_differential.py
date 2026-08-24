"""Differential tests: nanpercentile_masked fast path vs stock numpy.nanpercentile.

Sibling of nanquantile_masked (its differential suite carries the deep
regime coverage); this suite pins the percentile-specific surfaces: the
q-in-[0, 100] domain, numpy's exact q/100 scaling (the run performs the
same np.true_divide(q, 100) stock does, so dispatch cases assert exact
equality), and the floors/admission rule imported from the parent module.
"""

import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.nanquantile_masked import SIZE_FLOOR, _REDUCED_LEN_CAP

OP = "numpy.nanpercentile"
PATH = "nanpercentile_masked"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _make(shape, nan_frac=0.0, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape)
    if nan_frac:
        a[rng.random(shape) < nan_frac] = np.nan
    return a


def _assert_dispatched_exact(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.nanpercentile(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert got.dtype == stock.dtype and got.shape == stock.shape
    assert np.array_equal(got, stock, equal_nan=True)


def _assert_refused_equal(args, kwargs, equal_nan=False):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.nanpercentile(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert got.dtype == stock.dtype
    assert np.array_equal(got, stock, equal_nan=equal_nan)


def test_dispatch_2d_with_nans_exact():
    a = _make((27, 100), nan_frac=0.1)
    _assert_dispatched_exact((a, 80.0), {"axis": 0})
    _assert_dispatched_exact((a, 33.0, 1), {})


def test_dispatch_no_nans_and_int_q():
    a = _make((50, 40), seed=2)
    _assert_dispatched_exact((a, 25), {"axis": 1})


def test_dispatch_q_boundaries_0_50_100():
    a = _make((30, 64), nan_frac=0.05, seed=3)
    for q in (0.0, 50.0, 100.0):
        _assert_dispatched_exact((a, q), {"axis": 1})


def test_dispatch_3d_negative_axis():
    a = _make((8, 20, 30), nan_frac=0.02, seed=4)
    _assert_dispatched_exact((a, 90.0), {"axis": -1})


def test_all_nan_slice_warns_like_stock():
    a = _make((20, 40), seed=5)
    a[3, :] = np.nan
    with warnings.catch_warnings(record=True) as got_w:
        warnings.simplefilter("always")
        got = np.nanpercentile(a, 60.0, axis=1)
    with warnings.catch_warnings(record=True) as stock_w:
        warnings.simplefilter("always")
        stock = _stock(a, 60.0, axis=1)
    assert np.array_equal(got, stock, equal_nan=True)
    assert [str(w.message) for w in got_w] == [str(w.message) for w in stock_w]


def test_refusal_q_above_100_raises_like_stock():
    a = _make((30, 30), seed=6)
    decision, _ = GEARBOX.decide(OP, (a, 101.0), {"axis": 0})
    assert decision == "stock"
    with pytest.raises(ValueError):
        np.nanpercentile(a, 101.0, axis=0)
    with pytest.raises(ValueError):
        _stock(a, 101.0, axis=0)


def test_refusal_q_between_1_and_100_still_dispatches():
    # the percentile domain is [0, 100]; 0.5 means half a percent here,
    # not a median - it must dispatch and still agree with stock
    a = _make((30, 40), nan_frac=0.05, seed=7)
    _assert_dispatched_exact((a, 0.5), {"axis": 1})


def test_refusal_q_sequence_axis_none_1d_and_kwargs():
    a = _make((30, 40), nan_frac=0.05, seed=8)
    _assert_refused_equal((a, [25.0, 75.0]), {"axis": 1}, equal_nan=True)
    _assert_refused_equal((a, 50.0), {}, equal_nan=True)
    _assert_refused_equal((_make(5_000, seed=9), 50.0), {"axis": 0})
    _assert_refused_equal((a, 50.0), {"axis": 1, "keepdims": True}, equal_nan=True)
    _assert_refused_equal((a, 50.0), {"axis": 1, "method": "nearest"}, equal_nan=True)


def test_refusal_below_size_floor_and_reduced_len_rule():
    small = _make((10, 20), seed=10)
    assert small.size < SIZE_FLOOR
    _assert_refused_equal((small, 40.0), {"axis": 0})
    tall = _make((100_000, 5), nan_frac=0.01, seed=11)
    assert tall.shape[0] > _REDUCED_LEN_CAP and tall.size < tall.shape[0] ** 2
    _assert_refused_equal((tall, 40.0), {"axis": 0}, equal_nan=True)


def test_refusal_float32():
    a = _make((30, 40), seed=12).astype(np.float32)
    _assert_refused_equal((a, 50.0), {"axis": 1})


def test_kill_switch_restores_stock_routing():
    a = _make((27, 100), nan_frac=0.1, seed=13)
    decision, _ = GEARBOX.decide(OP, (a, 80.0), {"axis": 0})
    assert decision == PATH
    pyoverdrive.disable_path(PATH)
    try:
        decision, _ = GEARBOX.decide(OP, (a, 80.0), {"axis": 0})
        assert decision == "stock"
        assert np.array_equal(
            np.nanpercentile(a, 80.0, axis=0), _stock(a, 80.0, axis=0), equal_nan=True
        )
    finally:
        pyoverdrive.enable_path(PATH)
