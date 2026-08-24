"""Differential tests: quantile_dense_sort fast path vs stock numpy.quantile.

Contract (src/pyoverdrive/fastpaths/quantile_dense_sort.py): applies only to
quantile(a, q[, axis]) where a is a plain float64 ndarray, 1-D (axis absent,
None, 0, or -1 all mean the same reduction) or 2-D reduced along its LAST
axis (axis -1 or 1); q is a plain 1-D float64 ndarray with 4 <= q.size <=
16384, every q in [0, 1] (stock raises outside); the reduced length (last
axis size) is in [512, 65536]; method is absent or explicitly 'linear'.
out/keepdims/weights/other axes/scalar q/3-D+ all stay on stock. The route
replicates numpy's own virtual-index + lerp arithmetic bit-for-bit (spec
section 9, comparison mode: bit-identical), including the above-bounds -1
substitution, the gamma >= 0.5 stability rewrite, and NaN-last-after-sort
propagation, so dispatch cases assert exact equality (np.array_equal with
equal_nan=True), not allclose.
"""

import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

OP = "numpy.quantile"
PATH = "quantile_dense_sort"

RNG = np.random.default_rng(20260823)

_OMIT = object()


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _make(shape, seed=None):
    rng = RNG if seed is None else np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float64)


def _args_kwargs(a, q, axis):
    if axis is _OMIT:
        return (a, q), {}
    return (a, q, axis), {}


def _assert_exact(got, stock):
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.array_equal(got, stock, equal_nan=True)


def _assert_dispatched_exact(a, q, axis=_OMIT, extra_kwargs=None):
    args, kwargs = _args_kwargs(a, q, axis)
    if extra_kwargs:
        kwargs = {**kwargs, **extra_kwargs}
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.quantile(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    _assert_exact(got, stock)
    return got


def _assert_refused_equal(args, kwargs, equal_nan=False):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.quantile(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert got.dtype == stock.dtype
    assert np.array_equal(got, stock, equal_nan=equal_nan)
    return got


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.quantile(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + bit-identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "axis", [None, 0, -1, _OMIT], ids=["axisNone", "axis0", "axisNeg1", "axisOmit"]
)
def test_dispatch_1d_axis_spellings(axis):
    a = _make((2048,), seed=1)
    q = np.sort(np.random.default_rng(1001).random(128).astype(np.float64))
    _assert_dispatched_exact(a, q, axis)


@pytest.mark.parametrize("nq", [4, 128, 1000])
def test_dispatch_1d_q_sizes(nq):
    a = _make((2048,), seed=2)
    q = np.sort(np.random.default_rng(2000 + nq).random(nq).astype(np.float64))
    _assert_dispatched_exact(a, q, _OMIT)


@pytest.mark.parametrize("axis", [-1, 1])
def test_dispatch_2d_axis(axis):
    a = _make((40, 1024), seed=3)
    q = np.sort(np.random.default_rng(3000 + axis).random(64).astype(np.float64))
    _assert_dispatched_exact(a, q, axis)


def test_dispatch_q_endpoints_included():
    a = _make((2048,), seed=4)
    q = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float64)
    _assert_dispatched_exact(a, q, _OMIT)


def test_dispatch_q_unsorted_with_duplicates():
    a = _make((2048,), seed=5)
    q = np.array([0.5, 0.1, 0.5, 0.9, 0.1, 0.0, 1.0, 0.3], dtype=np.float64)
    _assert_dispatched_exact(a, q, _OMIT)


def test_dispatch_1d_single_nan():
    a = _make((2048,), seed=6)
    a[500] = np.nan
    q = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _assert_dispatched_exact(a, q, _OMIT)


def test_dispatch_1d_all_nan():
    a = np.full((2048,), np.nan, dtype=np.float64)
    q = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _assert_dispatched_exact(a, q, _OMIT)


def test_dispatch_2d_single_nan_one_slice():
    a = _make((20, 600), seed=7)
    a[3, 50] = np.nan
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _assert_dispatched_exact(a, q, -1)


def test_dispatch_2d_one_all_nan_slice():
    a = _make((20, 600), seed=8)
    a[5, :] = np.nan
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _assert_dispatched_exact(a, q, -1)


def test_dispatch_2d_all_slices_nan():
    a = _make((20, 600), seed=9)
    a[:, 0] = np.nan  # every slice (row) picks up a NaN via column 0
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _assert_dispatched_exact(a, q, -1)


def test_dispatch_inf_and_neg_inf_salted():
    a = _make((2048,), seed=10)
    a[10] = np.inf
    a[20] = -np.inf
    a[30] = np.inf
    q = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _assert_dispatched_exact(a, q, _OMIT)


def test_dispatch_method_linear_explicit():
    a = _make((2048,), seed=11)
    q = np.sort(np.random.default_rng(1100).random(64).astype(np.float64))
    _assert_dispatched_exact(a, q, _OMIT, extra_kwargs={"method": "linear"})


def test_dispatch_reduced_length_lower_boundary():
    a = _make((512,), seed=12)
    q = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    _assert_dispatched_exact(a, q, _OMIT)


def test_dispatch_reduced_length_upper_boundary():
    a = _make((65536,), seed=13)
    q = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    _assert_dispatched_exact(a, q, _OMIT)


def test_dispatch_q_size_upper_boundary():
    a = _make((600,), seed=14)
    q = np.sort(np.random.default_rng(14).random(16384).astype(np.float64))
    _assert_dispatched_exact(a, q, _OMIT)


# ---------------------------------------------------------------------------
# 2. refusal
# ---------------------------------------------------------------------------

def test_refusal_scalar_q():
    a = _make((600,), seed=20)
    _assert_refused_equal((a, 0.5), {})


def test_refusal_q_as_python_list():
    a = _make((600,), seed=21)
    _assert_refused_equal((a, [0.1, 0.3, 0.5, 0.9]), {})


def test_refusal_q_size_below_floor():
    a = _make((600,), seed=22)
    q = np.array([0.1, 0.5, 0.9], dtype=np.float64)  # size 3, floor is 4
    _assert_refused_equal((a, q), {})


def test_refusal_q_size_above_ceiling():
    a = _make((600,), seed=23)
    q = np.sort(np.random.default_rng(23).random(16385).astype(np.float64))
    _assert_refused_equal((a, q), {})


def test_refusal_q_out_of_range_raises():
    a = _make((600,), seed=24)
    q = np.array([0.1, 0.5, 1.0000001, 0.9], dtype=np.float64)
    _assert_refused_raises((a, q), {})


def test_refusal_q_float32():
    a = _make((600,), seed=25)
    q = np.array([0.1, 0.5, 0.9, 0.2], dtype=np.float32)
    _assert_refused_equal((a, q), {})


def test_refusal_reduced_length_below_floor():
    a = _make((511,), seed=26)
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    _assert_refused_equal((a, q), {})


def test_refusal_reduced_length_above_ceiling():
    a = _make((65537,), seed=27)
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    _assert_refused_equal((a, q), {})


def test_refusal_2d_axis0():
    a = _make((600, 20), seed=28)
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    _assert_refused_equal((a, q, 0), {})


def test_refusal_3d_input():
    a = _make((10, 20, 600), seed=29)
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    _assert_refused_equal((a, q, -1), {})


def test_refusal_float32_a():
    a = _make((600,), seed=30).astype(np.float32)
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    _assert_refused_equal((a, q), {})


def test_refusal_int64_a():
    a = np.random.default_rng(31).integers(0, 1000, size=600).astype(np.int64)
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    _assert_refused_equal((a, q), {})


def test_refusal_method_nearest():
    a = _make((600,), seed=32)
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    _assert_refused_equal((a, q), {"method": "nearest"})


def test_refusal_out_kwarg():
    a = _make((600,), seed=33)
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    out_got = np.empty(4, dtype=np.float64)
    out_stock = np.empty(4, dtype=np.float64)
    decision, reason = GEARBOX.decide(OP, (a, q), {"out": out_got})
    assert decision == "stock", (decision, reason)
    got = np.quantile(a, q, out=out_got)
    stock = _stock(a, q, out=out_stock)
    assert got is out_got
    _assert_exact(got, stock)


def test_refusal_keepdims_true():
    a = _make((600,), seed=34)
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    _assert_refused_equal((a, q), {"keepdims": True})


def test_refusal_weights_kwarg_raises():
    # linear (the default/only method this path accepts) does not support
    # weights at all; stock raises ValueError and this predicate refuses on
    # the extra unknown kwarg before ever reaching that check.
    a = _make((600,), seed=35)
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    w = np.ones(600, dtype=np.float64)
    _assert_refused_raises((a, q), {"weights": w})


def test_kill_switch_restores_stock_routing():
    a = _make((600,), seed=36)
    q = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    decision, reason = GEARBOX.decide(OP, (a, q), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (a, q), {})
        assert decision == "stock", (decision, reason)
        got = np.quantile(a, q)
        stock = _stock(a, q)
        _assert_exact(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)


# ---------------------------------------------------------------------------
# 3. consistency: quantile (no NaNs) must agree with nanquantile
# ---------------------------------------------------------------------------

def test_consistency_quantile_matches_nanquantile_when_no_nans():
    a = _make((30, 800), seed=40)
    q = np.sort(np.random.default_rng(41).random(50).astype(np.float64))
    decision, reason = GEARBOX.decide(OP, (a, q, -1), {})
    assert decision == PATH, (decision, reason)
    got_quantile = np.quantile(a, q, axis=-1)
    got_nanquantile = np.nanquantile(a, q, axis=-1)
    assert got_quantile.dtype == got_nanquantile.dtype
    assert got_quantile.shape == got_nanquantile.shape
    assert np.array_equal(got_quantile, got_nanquantile, equal_nan=True)
