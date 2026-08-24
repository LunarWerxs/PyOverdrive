"""Differential tests: isclose_fused fast path vs stock numpy.isclose.

Contract (src/pyoverdrive/fastpaths/isclose_fused.py): applies only to
isclose(a, b[, rtol, atol]) where rtol/atol (positional or keyword) are
FINITE, non-bool Python int/float and equal_nan is absent or False, and the
operands are either two non-bool Python int/float scalars (both finite) or
two plain same-shape same-dtype float64/float32 ndarrays, all-finite, below
the dtype size cap (1_000 float64, 10_000 float32). Dispatch computes the
fused expression abs(a - b) <= atol + rtol * abs(b), which for all-finite
operands and finite tolerances is exactly stock's within_tol arithmetic:
bit-identical output (np.bool_ for scalar input, a bool ndarray otherwise).
Comparison mode: bit-identical.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

OP = "numpy.isclose"
PATH = "isclose_fused"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _straddle(n, seed, lo=0.5, hi=2.0, dtype=np.float64):
    """a, b pair with the first half well inside the default tolerance and
    the second half clearly outside it (tol ~= 1e-8 + 1e-5 * |b|, which for
    b in [lo, hi] sits in roughly [1e-8, 2e-5])."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(lo, hi, size=n).astype(dtype)
    b = a.copy()
    half = n // 2
    b[:half] += dtype(1e-9)
    b[half:] += dtype(1e-4)
    return a, b


def _assert_dispatched_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.isclose(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    if isinstance(stock, np.ndarray):
        assert got.dtype == np.bool_
        assert got.shape == stock.shape
        assert isinstance(got, np.ndarray)
    assert np.array_equal(got, stock)
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.isclose(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert np.array_equal(got, stock)
    return got


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.isclose(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + bit-identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 10, 500, 1000])
def test_dispatch_float64_sizes(n):
    a, b = _straddle(n, seed=100 + n)
    got, stock = _assert_dispatched_equal((a, b), {})
    if n >= 500:
        assert got.any() and not got.all()


def test_dispatch_float32_n5000():
    a, b = _straddle(5000, seed=200, dtype=np.float32)
    got, stock = _assert_dispatched_equal((a, b), {})
    assert got.any() and not got.all()


def test_dispatch_scalar_close_floats():
    _assert_dispatched_equal((0.5, 0.50000001), {})


def test_dispatch_scalar_equal_ints():
    _assert_dispatched_equal((5, 5), {})


def test_dispatch_scalar_big_ints_float64_precision_parity():
    # 2**62 and 2**62 + 1 round to the same float64 (ulp >> 1 at this
    # magnitude); stock and the fast path both go through that rounding.
    _assert_dispatched_equal((2**62, 2**62 + 1), {})


def test_dispatch_custom_tolerances_positional():
    a, b = _straddle(200, seed=301)
    _assert_dispatched_equal((a, b, 1e-3, 1e-6), {})


def test_dispatch_custom_tolerances_kwarg():
    a, b = _straddle(200, seed=302)
    _assert_dispatched_equal((a, b), {"rtol": 1e-3, "atol": 1e-6})


def test_dispatch_zero_tolerances_exact_equality_edge():
    rng = np.random.default_rng(303)
    a = rng.uniform(0.1, 5.0, size=200)
    b = a.copy()
    b[::2] += 1e-12  # perturbed -> False under rtol=atol=0
    got, stock = _assert_dispatched_equal((a, b, 0, 0), {})
    assert got.any() and not got.all()


def test_dispatch_equal_nan_false_explicit():
    a, b = _straddle(50, seed=304)
    _assert_dispatched_equal((a, b), {"equal_nan": False})


def test_dispatch_negative_zero_pair():
    _assert_dispatched_equal((-0.0, 0.0), {})


# ---------------------------------------------------------------------------
# 2. refusal routes
# ---------------------------------------------------------------------------

def test_refusal_float64_n1001_just_over_cap():
    a, b = _straddle(1001, seed=401)
    _assert_refused_equal((a, b), {})


def test_refusal_float32_n10001_just_over_cap():
    a, b = _straddle(10_001, seed=402, dtype=np.float32)
    _assert_refused_equal((a, b), {})


def test_refusal_array_containing_nan():
    a, b = _straddle(50, seed=403)
    a[5] = np.nan
    _assert_refused_equal((a, b), {})


def test_refusal_array_containing_inf_preserves_isclose_inf_inf_true():
    a, b = _straddle(50, seed=404)
    a[10] = np.inf
    b[10] = np.inf
    got = _assert_refused_equal((a, b), {})
    assert bool(got[10]) is True


def test_refusal_equal_nan_true():
    a, b = _straddle(50, seed=405)
    _assert_refused_equal((a, b), {"equal_nan": True})


def test_refusal_atol_inf():
    a, b = _straddle(50, seed=406)
    _assert_refused_equal((a, b), {"atol": np.inf})


def test_refusal_rtol_nan():
    a, b = _straddle(50, seed=407)
    _assert_refused_equal((a, b), {"rtol": np.nan})


def test_atol_numpy_scalar_routes_and_matches_stock():
    # np.float64 subclasses Python float, so this may or may not clear the
    # predicate's isinstance/finite checks; assert stock agreement either
    # way rather than assuming the route.
    a, b = _straddle(50, seed=408)
    kwargs = {"atol": np.float64(1e-8)}
    decision, reason = GEARBOX.decide(OP, (a, b), kwargs)
    if decision == PATH:
        _assert_dispatched_equal((a, b), kwargs)
    else:
        _assert_refused_equal((a, b), kwargs)


def test_refusal_tolerance_as_bool():
    a, b = _straddle(50, seed=409)
    _assert_refused_equal((a, b), {"rtol": True})


def test_refusal_mixed_dtypes():
    a, _ = _straddle(50, seed=410, dtype=np.float64)
    b, _ = _straddle(50, seed=411, dtype=np.float32)
    _assert_refused_equal((a, b), {})


def test_refusal_mismatched_shapes_broadcast():
    x = np.array([[1.0], [2.0], [3.0]])  # (3, 1)
    y = np.array([1.0, 2.0, 3.0])  # (3,)
    _assert_refused_equal((x, y), {})


def test_refusal_ndarray_vs_scalar_mixed():
    a, _ = _straddle(50, seed=412)
    _assert_refused_equal((a, 0.5), {})


def test_refusal_int64_arrays():
    rng = np.random.default_rng(413)
    a = rng.integers(0, 100, size=50, dtype=np.int64)
    b = a.copy()
    _assert_refused_equal((a, b), {})


def test_refusal_python_bool_scalar_operand():
    _assert_refused_equal((True, 1.0), {})


def test_refusal_huge_python_int_predicate_error_raises_like_stock():
    # math.isfinite(10**400) raises OverflowError inside the predicate;
    # Gearbox.decide catches it and routes to stock, but the underlying
    # call itself still raises (stock hits the same OverflowError).
    _assert_refused_raises((10**400, 1.0), {})


def test_kill_switch_restores_stock_routing():
    a, b = _straddle(50, seed=414)
    decision, reason = GEARBOX.decide(OP, (a, b), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (a, b), {})
        assert decision == "stock", (decision, reason)
        got = np.isclose(a, b)
        stock = _stock(a, b)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)
