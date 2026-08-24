"""Differential tests: cholesky_small_batch fast path vs stock
numpy.linalg.cholesky.

Covers the Cholesky-Crout closed form on float64 (..., d, d) stacks,
d in {2, 3}, guarded by the PIVOT_RTOL positivity check in
cholesky_small_batch. Comparison mode is numeric (allclose rtol=1e-9,
atol=1e-12); refusals are checked against stock exactly (same
underlying call either way), including LinAlgError symmetry for
non-positive-definite input.
"""

import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.cholesky_small_batch import _WINDOWS, PIVOT_RTOL

_FLOORS = {d: w[0] for d, w in _WINDOWS.items()}

OP = "numpy.linalg.cholesky"
PATH = "cholesky_small_batch"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock():
    return GEARBOX.stock_fn(OP)


def _spd(batch_shape, d, seed, scale=0.3):
    rng = np.random.default_rng(seed)
    m = (rng.standard_normal(batch_shape + (d, d)) * scale).astype(np.float64)
    a = np.matmul(m, np.swapaxes(m, -1, -2)) + np.eye(d, dtype=np.float64) * 3.0
    return a


def _corrupt_upper(a):
    a = a.copy()
    d = a.shape[-1]
    a[..., 0, 1] = 999.0
    if d == 3:
        a[..., 0, 2] = -888.0
        a[..., 1, 2] = 777.0
    return a


def _bad_singular_3x3():
    # row1 == 2 * row0 exactly on the lower triangle -> zero pivot
    return np.array(
        [[4.0, 0.0, 0.0], [2.0, 1.0, 0.0], [8.0, 4.0, 4.0]], dtype=np.float64
    )


def _bad_indefinite_2x2():
    # negative Schur complement: a11=4, a21=5, a22=1 -> p2 = 1 - 25/4 < 0
    return np.array([[4.0, 0.0], [5.0, 1.0]], dtype=np.float64)


def _bad_near_semidefinite_3x3(d):
    a = _spd((), d, seed=99)
    scale = np.abs(a).max()
    if d == 2:
        # drive p2 just under PIVOT_RTOL * scale
        a11 = a[0, 0]
        target = PIVOT_RTOL * scale * 0.1
        a21 = np.sqrt(a[1, 1] * a11 - target * a11)
        a[1, 0] = a21
    else:
        a11 = a[0, 0]
        p2 = a[1, 1] - (a[1, 0] ** 2) / a11
        target = PIVOT_RTOL * scale * 0.1
        rhs = a[2, 2] - (a[2, 0] ** 2) / a11 - target
        # (a32 - a31*a21/a11)^2 / p2 = a[2,2] - a31^2/a11 - target
        val = rhs * p2
        base = a[2, 0] * a[1, 0] / a11
        a[2, 1] = base + np.sqrt(max(val, 0.0))
    return a


def _call(fn, args, kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as e:  # noqa: BLE001 - symmetric probe, any exception type
        return ("err", type(e))


def _assert_dispatched(a):
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision == PATH, (decision, reason)
    got = np.linalg.cholesky(a)
    stock = _stock()(a)
    assert np.allclose(got, stock, rtol=1e-9, atol=1e-12)
    return got, stock


def _assert_refused(args, kwargs=None):
    kwargs = kwargs or {}
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (decision, reason)
    got_tag, got = _call(np.linalg.cholesky, args, kwargs)
    stock_tag, stock = _call(_stock(), args, kwargs)
    assert got_tag == stock_tag, (got_tag, got, stock_tag, stock)
    if got_tag == "err":
        assert got is stock, (got, stock)
    else:
        assert np.array_equal(got, stock, equal_nan=True)


def _assert_served_by_stock(a):
    # in-window but guard-refused: decision is still PATH, and _run's
    # mid-run fallback must reproduce stock's exact outcome with no
    # fast-path branding (no RuntimeWarning, no divergent result/raise)
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision == PATH, (decision, reason)
    stock_tag, stock = _call(_stock(), (a,), {})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got_tag, got = _call(np.linalg.cholesky, (a,), {})
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught), caught
    assert got_tag == stock_tag, (got_tag, got, stock_tag, stock)
    if got_tag == "err":
        assert got is stock, (got, stock)
    else:
        assert np.array_equal(got, stock, equal_nan=True)
    return got, stock


# --- 1. well-conditioned dispatch, at and above floor --------------------


def test_dispatch_2x2_at_floor():
    a = _spd((_FLOORS[2],), 2, seed=1)
    _assert_dispatched(a)


def test_dispatch_2x2_above_floor():
    a = _spd((_FLOORS[2] + 200,), 2, seed=2)
    _assert_dispatched(a)


def test_dispatch_3x3_at_floor():
    a = _spd((_FLOORS[3],), 3, seed=3)
    _assert_dispatched(a)


def test_dispatch_3x3_above_floor():
    a = _spd((_FLOORS[3] + 700,), 3, seed=4)
    _assert_dispatched(a)


def test_dispatch_multi_batch_dims():
    a = _spd((4, 300), 3, seed=5)  # batch = 1200, inside the 3x3 window
    _assert_dispatched(a)


def test_dispatch_3x3_has_no_cap():
    lo, hi = _WINDOWS[3]
    assert hi is None
    a = _spd((5000,), 3, seed=95)
    _assert_dispatched(a)


def test_dispatch_2x2_has_no_cap():
    lo, hi = _WINDOWS[2]
    assert hi is None
    a = _spd((5000,), 2, seed=96)
    _assert_dispatched(a)


# --- 2. lower-triangle-only reading ---------------------------------------


def test_asymmetric_upper_triangle_still_dispatches_and_matches():
    a = _spd((_FLOORS[3] + 50,), 3, seed=6)
    a = _corrupt_upper(a)
    got, stock = _assert_dispatched(a)
    d = a.shape[-1]
    for i in range(d):
        for j in range(i + 1, d):
            assert np.all(got[..., i, j] == 0.0)
            assert np.all(stock[..., i, j] == 0.0)


# --- 3. result dtype / contiguity match stock -----------------------------


def test_result_dtype_and_contiguity_match_stock():
    a = _spd((_FLOORS[3] + 10,), 3, seed=7)
    got = np.linalg.cholesky(a)
    stock = _stock()(a)
    assert got.dtype == stock.dtype == np.float64
    assert got.flags["C_CONTIGUOUS"]
    assert stock.flags["C_CONTIGUOUS"]


# --- 4. floor / shape / dtype refusals ------------------------------------


def test_refusal_below_floor_2x2():
    a = _spd((_FLOORS[2] - 50,), 2, seed=8)
    _assert_refused((a,))


def test_refusal_below_floor_3x3():
    a = _spd((_FLOORS[3] - 100,), 3, seed=9)
    _assert_refused((a,))


def test_refusal_single_matrix_ndim2():
    a = _spd((), 3, seed=10)  # shape (3, 3), ndim == 2
    _assert_refused((a,))


def test_refusal_float32_stack():
    a = _spd((_FLOORS[3] + 10,), 3, seed=11).astype(np.float32)
    _assert_refused((a,))


# --- 5. kwargs refusal ------------------------------------------------------


def test_refusal_kwargs_upper_true():
    a = _spd((_FLOORS[3] + 10,), 3, seed=12)
    _assert_refused((a,), {"upper": True})


def test_refusal_kwargs_upper_false():
    a = _spd((_FLOORS[3] + 10,), 3, seed=13)
    _assert_refused((a,), {"upper": False})


# --- 6. non-finite input: served by stock via mid-run fallback ------------


def test_inf_entry_served_by_stock():
    a = _spd((_FLOORS[3] + 10,), 3, seed=14)
    a[7, 1, 0] = np.inf
    _assert_served_by_stock(a)


def test_nan_entry_served_by_stock():
    a = _spd((_FLOORS[2] + 10,), 2, seed=15)
    a[3, 1, 1] = np.nan
    _assert_served_by_stock(a)


# --- 7. non-PD input: served by stock, with stock-raise symmetry ----------


def test_exactly_singular_3x3_served_by_stock():
    a = _spd((_FLOORS[3] + 10,), 3, seed=16)
    a[5] = _bad_singular_3x3()
    _assert_served_by_stock(a)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(a)
    with pytest.raises(np.linalg.LinAlgError):
        _stock()(a)


def test_indefinite_2x2_served_by_stock():
    a = _spd((_FLOORS[2] + 10,), 2, seed=17)
    a[6] = _bad_indefinite_2x2()
    _assert_served_by_stock(a)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(a)
    with pytest.raises(np.linalg.LinAlgError):
        _stock()(a)


def test_near_semidefinite_2x2_served_by_stock():
    a = _spd((_FLOORS[2] + 10,), 2, seed=18)
    bad = _bad_near_semidefinite_3x3(2)
    scale = np.abs(bad).max()
    p2 = bad[1, 1] - bad[1, 0] ** 2 / bad[0, 0]
    assert 0.0 < p2 < PIVOT_RTOL * scale, (p2, scale)
    a[9] = bad
    _assert_served_by_stock(a)


def test_near_semidefinite_3x3_served_by_stock():
    a = _spd((_FLOORS[3] + 10,), 3, seed=19)
    bad = _bad_near_semidefinite_3x3(3)
    a[11] = bad
    _assert_served_by_stock(a)
