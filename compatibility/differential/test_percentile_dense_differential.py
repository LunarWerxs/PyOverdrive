"""Differential tests: percentile_dense fast path vs stock numpy.percentile.

Sibling of quantile_dense_sort (its differential suite carries the deep
regime coverage); this suite pins the percentile-specific surfaces: the
q-in-[0,100] domain, numpy's exact q/100 scaling (bit-identity), and the
shared floors imported from the quantile module.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.quantile_dense_sort import M_MIN, NQ_MIN

OP = "numpy.percentile"
PATH = "percentile_dense"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _mk(slices, m, nq, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((slices, m) if slices else (m,))
    q = np.linspace(0.0, 100.0, nq)
    return a, q


def _assert_dispatched_exact(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.percentile(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert got.dtype == stock.dtype and got.shape == stock.shape
    assert np.array_equal(got, stock, equal_nan=True)


def _assert_refused_exact(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (decision, reason)
    got = np.percentile(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert np.array_equal(got, stock, equal_nan=True)


def test_dispatch_2d_dense_bit_identical():
    a, q = _mk(30, 1024, 128)
    _assert_dispatched_exact((a, q), {"axis": -1})


def test_dispatch_1d_and_method_linear_explicit():
    a, q = _mk(None, 2048, 64, seed=2)
    _assert_dispatched_exact((a, q), {})
    _assert_dispatched_exact((a, q), {"method": "linear"})


def test_dispatch_q_boundaries_0_and_100():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(1024)
    q = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    _assert_dispatched_exact((a, q), {})


def test_dispatch_nan_salted_slices():
    a, q = _mk(20, 1024, 32, seed=4)
    a[3, 5] = np.nan
    _assert_dispatched_exact((a, q), {"axis": -1})


def test_refusal_q_above_100():
    rng = np.random.default_rng(5)
    a = rng.standard_normal(1024)
    q = np.array([10.0, 50.0, 101.0, 90.0])
    decision, _ = GEARBOX.decide(OP, (a, q), {})
    assert decision == "stock"
    with pytest.raises(ValueError):
        np.percentile(a, q)
    with pytest.raises(ValueError):
        _stock(a, q)


def test_refusal_scalar_q_and_small_q_and_short_m():
    rng = np.random.default_rng(6)
    a = rng.standard_normal(1024)
    _assert_refused_exact((a, 50.0), {})
    q_small = np.linspace(0.0, 100.0, NQ_MIN - 1)
    _assert_refused_exact((a, q_small), {})
    short = rng.standard_normal(M_MIN - 1)
    _assert_refused_exact((short, np.linspace(0.0, 100.0, 64)), {})


def test_refusal_other_method_and_keepdims():
    a, q = _mk(None, 2048, 64, seed=7)
    _assert_refused_exact((a, q), {"method": "nearest"})
    _assert_refused_exact((a, q), {"keepdims": True})


def test_kill_switch_restores_stock_routing():
    a, q = _mk(None, 2048, 64, seed=8)
    decision, _ = GEARBOX.decide(OP, (a, q), {})
    assert decision == PATH
    pyoverdrive.disable_path(PATH)
    try:
        decision, _ = GEARBOX.decide(OP, (a, q), {})
        assert decision == "stock"
        assert np.array_equal(np.percentile(a, q), _stock(a, q))
    finally:
        pyoverdrive.enable_path(PATH)
