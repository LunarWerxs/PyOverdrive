"""Differential tests: nanquantile_masked fast path vs stock numpy.nanquantile.

Contract (src/pyoverdrive/fastpaths/nanquantile_masked.py): applies only to
nanquantile(a, q[, axis]) where a is a plain 2-D+ float64 ndarray with
a.size >= SIZE_FLOOR (300), q is a single Python int/float in [0, 1] (bool
refused, sequences refused), axis is a single int (bool refused, None
refused, omitted refused), and NO other argument is passed. A regime guard
also applies: reduced = a.shape[axis] must satisfy reduced <= 1000 OR
a.size >= reduced * reduced. The interpolation replicates numpy's internal
_lerp bit-for-bit (a calibration probe measured 0.0 worst-case difference
across 140 cases), so dispatch cases below assert exact equality
(np.array_equal with equal_nan=True), not allclose.
"""

import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

RNG = np.random.default_rng(20260823)


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(["numpy.nanquantile"])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn("numpy.nanquantile")(*args, **kwargs)


def _assert_exact(got, stock):
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.array_equal(got, stock, equal_nan=True)


def _make(shape, nan_frac=0.0, seed=None):
    rng = RNG if seed is None else np.random.default_rng(seed)
    a = rng.standard_normal(shape)
    if nan_frac:
        mask = rng.random(shape) < nan_frac
        a[mask] = np.nan
    return a


def _assert_dispatched_exact(a, q, axis, use_kwarg=False):
    args = (a, q)
    kwargs = {}
    if use_kwarg:
        kwargs["axis"] = axis
    else:
        args = (a, q, axis)
    decision, reason = GEARBOX.decide("numpy.nanquantile", args, kwargs)
    assert decision == "nanquantile_masked", (decision, reason)
    got = np.nanquantile(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    _assert_exact(got, stock)
    return got


def _assert_refused_equal(args, kwargs, equal_nan=False):
    decision, reason = GEARBOX.decide("numpy.nanquantile", args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.nanquantile(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert got.dtype == stock.dtype
    assert np.array_equal(got, stock, equal_nan=equal_nan)


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide("numpy.nanquantile", args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.nanquantile(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + exact equality
# ---------------------------------------------------------------------------

DISPATCH_CASES = []
for shape, axis in [
    ((27, 100), 0), ((27, 100), 1), ((27, 100), -1),
    ((50, 40, 30), 0), ((50, 40, 30), 1), ((50, 40, 30), 2),
    ((500, 500), 0),
]:
    for q in (0.0, 0.25, 0.8, 1.0):
        for nan_frac in (0.0, 0.1, 0.5):
            DISPATCH_CASES.append(
                pytest.param(shape, axis, q, nan_frac, id=f"{shape}-ax{axis}-q{q}-nan{nan_frac}")
            )


@pytest.mark.parametrize("shape, axis, q, nan_frac", DISPATCH_CASES)
def test_dispatch_exact_equality(shape, axis, q, nan_frac):
    a = _make(shape, nan_frac)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _assert_dispatched_exact(a, q, axis)


def test_dispatch_q_as_python_int_zero():
    a = _make((27, 100))
    _assert_dispatched_exact(a, 0, 0)


def test_dispatch_q_as_python_int_one():
    a = _make((27, 100))
    _assert_dispatched_exact(a, 1, 0)


def test_dispatch_axis_positional():
    a = _make((50, 40, 30))
    _assert_dispatched_exact(a, 0.5, 1, use_kwarg=False)


def test_dispatch_axis_kwarg():
    a = _make((50, 40, 30))
    _assert_dispatched_exact(a, 0.5, 1, use_kwarg=True)


# ---------------------------------------------------------------------------
# 2. all-NaN-slice case
# ---------------------------------------------------------------------------

def test_all_nan_slice_matches_stock_and_both_warn():
    a = _make((300, 10), nan_frac=0.0, seed=7)
    a[:, 3] = np.nan
    decision, reason = GEARBOX.decide("numpy.nanquantile", (a, 0.5, 0), {})
    assert decision == "nanquantile_masked", (decision, reason)
    with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
        got = np.nanquantile(a, 0.5, 0)
    with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
        stock = _stock(a, 0.5, 0)
    _assert_exact(got, stock)


# ---------------------------------------------------------------------------
# 3. refusal routes
# ---------------------------------------------------------------------------

def test_refusal_axis_omitted_full_reduction():
    a = _make((27, 100))
    _assert_refused_equal((a, 0.5), {})


def test_refusal_axis_none_kwarg():
    a = _make((27, 100))
    _assert_refused_equal((a, 0.5), {"axis": None})


def test_refusal_q_as_list():
    a = _make((27, 100))
    _assert_refused_equal((a, [0.5, 0.8], 0), {})


def test_refusal_q_above_one_raises():
    a = _make((27, 100))
    _assert_refused_raises((a, 1.5, 0), {})


def test_refusal_q_below_zero_raises():
    a = _make((27, 100))
    _assert_refused_raises((a, -0.1, 0), {})


def test_refusal_float32_input():
    a = _make((27, 100)).astype(np.float32)
    _assert_refused_equal((a, 0.5, 0), {})


def test_refusal_int64_input():
    a = RNG.integers(0, 1000, size=(27, 100)).astype(np.int64)
    _assert_refused_equal((a, 0.5, 0), {})


def test_refusal_1d_input():
    a = _make((500,))
    _assert_refused_equal((a, 0.5, 0), {})


def test_refusal_size_below_floor():
    a = _make((10, 20))
    _assert_refused_equal((a, 0.5, 0), {})


def test_refusal_keepdims_true():
    a = _make((27, 100))
    _assert_refused_equal((a, 0.5, 0), {"keepdims": True})


def test_refusal_out_provided():
    a = _make((27, 100))
    out = np.empty(100)
    decision, reason = GEARBOX.decide("numpy.nanquantile", (a, 0.5, 0), {"out": out})
    assert decision == "stock", (decision, reason)
    out_stock = np.empty(100)
    got = np.nanquantile(a, 0.5, 0, out=out)
    stock = _stock(a, 0.5, 0, out=out_stock)
    _assert_exact(got, stock)


def test_refusal_method_nearest():
    a = _make((27, 100))
    _assert_refused_equal((a, 0.5, 0), {"method": "nearest"})


def test_refusal_overwrite_input_true():
    a = _make((27, 100))
    a1 = a.copy()
    a2 = a.copy()
    _assert_refused_equal((a1, 0.5, 0), {"overwrite_input": True})
    decision, _ = GEARBOX.decide("numpy.nanquantile", (a2, 0.5, 0), {"overwrite_input": True})
    assert decision == "stock"


def test_refusal_weights_kwarg():
    a = _make((27, 100), nan_frac=0.0)
    w = np.ones_like(a)
    _assert_refused_equal((a, 0.5, 1), {"weights": w, "method": "inverted_cdf"})


def test_refusal_anti_regime_wide_short_axis0():
    # reduced = 10000 > 1000, size = 30000 < 10000**2: refused.
    a = _make((10000, 3))
    _assert_refused_equal((a, 0.5, 0), {})


def test_refusal_anti_regime_wide_short_axis1():
    # reduced = 10000 > 1000, size = 30000 < 10000**2: refused.
    a = _make((3, 10000))
    _assert_refused_equal((a, 0.5, 1), {})


def test_dispatch_regime_boundary_large_reduced_dispatches():
    # reduced = 1500 > 1000 but size = 2.25e6 >= 1500**2: dispatches.
    a = _make((1500, 1500), seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _assert_dispatched_exact(a, 0.5, 0)


def test_kill_switch_restores_stock_routing():
    a = _make((27, 100))
    decision, reason = GEARBOX.decide("numpy.nanquantile", (a, 0.5, 0), {})
    assert decision == "nanquantile_masked", (decision, reason)
    pyoverdrive.disable_path("nanquantile_masked")
    try:
        decision, reason = GEARBOX.decide("numpy.nanquantile", (a, 0.5, 0), {})
        assert decision == "stock", (decision, reason)
        got = np.nanquantile(a, 0.5, 0)
        stock = _stock(a, 0.5, 0)
        _assert_exact(got, stock)
    finally:
        pyoverdrive.enable_path("nanquantile_masked")


# ---------------------------------------------------------------------------
# 4. layout: Fortran-order and negative-strided views still dispatch
# ---------------------------------------------------------------------------

def test_fortran_order_input_matches_stock():
    a = np.asfortranarray(_make((50, 60), nan_frac=0.1, seed=13))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _assert_dispatched_exact(a, 0.4, 0)


def test_negative_strided_view_matches_stock():
    a = _make((50, 60), nan_frac=0.1, seed=17)[::-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _assert_dispatched_exact(a, 0.6, 1)
