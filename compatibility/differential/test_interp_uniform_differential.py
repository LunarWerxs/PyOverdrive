"""Differential tests: interp_uniform_grid fast path vs stock numpy.interp.

Covers the direct-index linear interpolation in interp_uniform_grid for
interp(x, xp, fp) with all three args plain 1-D float64, xp uniformly
spaced (within UNIFORM_RTOL) and strictly increasing, x.size >= NQ_MIN,
and no left/right/period kwargs. Comparison mode is numeric (allclose
rtol=1e-9, atol=1e-9, per the module docstring's measured 1e-14..1e-12
relative drift); refusals are checked against stock exactly (same
underlying call either way), including ValueError parity for a
fp/xp length mismatch.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.interp_uniform_grid import NQ_MIN, UNIFORM_RTOL

OP = "numpy.interp"
PATH = "interp_uniform_grid"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock():
    return GEARBOX.stock_fn(OP)


def _linspace_grid(lo, hi, n):
    return np.linspace(lo, hi, n, dtype=np.float64)


def _arange_grid(start, stop, step):
    return np.arange(start, stop, step, dtype=np.float64)


def _queries(xp, n, seed, span_pad=0.2, grid_hits=50):
    """Random queries covering both out-of-range sides, unsorted, with a
    handful of entries snapped exactly onto grid points."""
    rng = np.random.default_rng(seed)
    lo, hi = float(xp[0]), float(xp[-1])
    span = hi - lo
    x = rng.uniform(lo - span_pad * span, hi + span_pad * span, size=n).astype(
        np.float64
    )
    k = min(grid_hits, xp.size, n)
    if k > 0:
        idx = rng.choice(xp.size, size=k, replace=False)
        pos = rng.choice(n, size=k, replace=False)
        x[pos] = xp[idx]
    return x


def _call(fn, args, kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as e:  # noqa: BLE001 - symmetric probe, any exception type
        return ("err", type(e))


def _assert_dispatched(x, xp, fp):
    decision, reason = GEARBOX.decide(OP, (x, xp, fp), {})
    assert decision == PATH, (decision, reason)
    got = np.interp(x, xp, fp)
    stock = _stock()(x, xp, fp)
    assert np.allclose(got, stock, rtol=1e-9, atol=1e-9)
    return got, stock


def _assert_refused(args, kwargs=None):
    kwargs = kwargs or {}
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (decision, reason)
    got_tag, got = _call(np.interp, args, kwargs)
    stock_tag, stock = _call(_stock(), args, kwargs)
    assert got_tag == stock_tag, (got_tag, got, stock_tag, stock)
    if got_tag == "err":
        assert got is stock, (got, stock)
    else:
        assert np.array_equal(got, stock, equal_nan=True)


# --- 1. dispatch + numeric match, linspace and arange grids ----------------


def test_dispatch_linspace_grid_numeric_match():
    xp = _linspace_grid(-5.0, 25.0, 733)
    fp = np.sin(xp) + 0.3 * xp
    x = _queries(xp, NQ_MIN + 500, seed=1)
    _assert_dispatched(x, xp, fp)


def test_dispatch_arange_grid_numeric_match():
    xp = _arange_grid(-1000.0, 3000.0, 1.5)
    fp = np.cos(0.01 * xp) * 7.0 - 2.0
    x = _queries(xp, NQ_MIN + 500, seed=2)
    _assert_dispatched(x, xp, fp)


# --- 2. out-of-range both sides, on-grid, unsorted --------------------------


def test_dispatch_out_of_range_both_sides():
    xp = _linspace_grid(0.0, 10.0, 500)
    fp = np.linspace(100.0, 200.0, 500)
    rng = np.random.default_rng(3)
    below = rng.uniform(-500.0, -0.001, size=NQ_MIN // 2).astype(np.float64)
    above = rng.uniform(10.001, 500.0, size=NQ_MIN // 2).astype(np.float64)
    x = np.concatenate([below, above])
    got, stock = _assert_dispatched(x, xp, fp)
    assert np.all(got[: below.size] == fp[0])
    assert np.all(got[below.size :] == fp[-1])
    assert np.array_equal(got, stock)


def test_dispatch_queries_exactly_on_grid_points():
    xp = _linspace_grid(-2.0, 2.0, 200)
    fp = xp**2 - xp
    reps = -(-NQ_MIN // xp.size)  # ceil
    x = np.tile(xp, reps)[:NQ_MIN].astype(np.float64)
    got, stock = _assert_dispatched(x, xp, fp)
    assert np.allclose(got, np.tile(fp, reps)[:NQ_MIN], rtol=1e-9, atol=1e-9)


def test_dispatch_unsorted_queries():
    xp = _linspace_grid(0.0, 50.0, 400)
    fp = np.sin(xp * 0.2)
    x = _queries(xp, NQ_MIN, seed=4)
    assert not np.all(np.diff(x) >= 0)  # genuinely unsorted
    _assert_dispatched(x, xp, fp)


# --- 3. constant fp, single-bin grid ----------------------------------------


def test_dispatch_constant_fp():
    xp = _linspace_grid(-10.0, 10.0, 1000)
    fp = np.full(xp.size, 42.5, dtype=np.float64)
    x = _queries(xp, NQ_MIN, seed=5)
    got, stock = _assert_dispatched(x, xp, fp)
    assert np.all(got == 42.5)


def test_dispatch_single_bin_grid():
    xp = np.array([0.0, 1.0], dtype=np.float64)
    fp = np.array([3.0, 9.0], dtype=np.float64)
    x = _queries(xp, NQ_MIN, seed=6)
    _assert_dispatched(x, xp, fp)


# --- 4. refusal: below NQ_MIN, non-uniform, descending, xp size 1 ----------


def test_refusal_below_nq_min():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 100)
    x = _queries(xp, NQ_MIN - 1, seed=7)
    _assert_refused((x, xp, fp))


def test_refusal_non_uniform_xp():
    xp = _linspace_grid(0.0, 1.0, 100)
    dx = xp[1] - xp[0]
    xp = xp.copy()
    xp[50] += dx * UNIFORM_RTOL * 1e6  # far past the tolerance
    fp = np.linspace(0.0, 1.0, 100)
    x = _queries(xp, NQ_MIN, seed=8)
    _assert_refused((x, xp, fp))


def test_refusal_descending_xp():
    xp = _linspace_grid(0.0, 1.0, 100)[::-1].copy()
    fp = np.linspace(0.0, 1.0, 100)
    x = _queries(xp[::-1], NQ_MIN, seed=9)
    _assert_refused((x, xp, fp))


def test_refusal_xp_size_one():
    xp = np.array([2.0], dtype=np.float64)
    fp = np.array([7.0], dtype=np.float64)
    x = np.linspace(-5.0, 5.0, NQ_MIN, dtype=np.float64)
    _assert_refused((x, xp, fp))


# --- 5. refusal: non-finite --------------------------------------------------


def test_refusal_nan_in_x():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 100)
    x = _queries(xp, NQ_MIN, seed=10)
    x[123] = np.nan
    _assert_refused((x, xp, fp))


def test_refusal_nan_in_fp():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 100)
    fp = fp.copy()
    fp[7] = np.nan
    x = _queries(xp, NQ_MIN, seed=11)
    _assert_refused((x, xp, fp))


def test_refusal_inf_in_fp():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 100)
    fp = fp.copy()
    fp[7] = np.inf
    x = _queries(xp, NQ_MIN, seed=12)
    _assert_refused((x, xp, fp))


# --- 6. refusal: kwargs -------------------------------------------------


def test_refusal_left_kwarg():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 100)
    x = _queries(xp, NQ_MIN, seed=13)
    _assert_refused((x, xp, fp), {"left": -1.0})


def test_refusal_right_kwarg():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 100)
    x = _queries(xp, NQ_MIN, seed=14)
    _assert_refused((x, xp, fp), {"right": 99.0})


def test_refusal_period_kwarg():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 100)
    x = _queries(xp, NQ_MIN, seed=15)
    _assert_refused((x, xp, fp), {"period": 1.0})


# --- 7. refusal: dtype / shape ----------------------------------------------


def test_refusal_non_f64_x():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 100)
    x = _queries(xp, NQ_MIN, seed=16).astype(np.float32)
    _assert_refused((x, xp, fp))


def test_refusal_non_f64_xp():
    xp = _linspace_grid(0.0, 1.0, 100).astype(np.float32)
    fp = np.linspace(0.0, 1.0, 100)
    x = _queries(xp.astype(np.float64), NQ_MIN, seed=17)
    _assert_refused((x, xp, fp))


def test_refusal_2d_x():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 100)
    x = _queries(xp, NQ_MIN, seed=18).reshape(-1, 1)
    _assert_refused((x, xp, fp))


def test_refusal_fp_length_mismatch():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 99)
    x = _queries(xp, NQ_MIN, seed=19)
    _assert_refused((x, xp, fp))
    with pytest.raises(ValueError):
        np.interp(x, xp, fp)
    with pytest.raises(ValueError):
        _stock()(x, xp, fp)


def test_refusal_x_scalar():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 100)
    _assert_refused((0.5, xp, fp))


def test_refusal_x_0d_array():
    xp = _linspace_grid(0.0, 1.0, 100)
    fp = np.linspace(0.0, 1.0, 100)
    x = np.array(0.5, dtype=np.float64)
    _assert_refused((x, xp, fp))
