"""Differential tests: nan_to_num_where fast path vs stock numpy.nan_to_num.

Contract (see src/pyoverdrive/fastpaths/nan_to_num_where.py docstring):
exactly nan_to_num(x): plain float64 ndarray, size >= SIZE_FLOOR, no kwargs
at all. Bit-identical to stock, always a fresh copy like stock's copy=True.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.nan_to_num_where import SIZE_FLOOR

OP = "numpy.nan_to_num"
PATH = "nan_to_num_where"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _mk_1d(n, nan_frac=0.0, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n).astype(np.float64)
    if nan_frac:
        k = max(1, int(n * nan_frac))
        idx = rng.choice(n, size=k, replace=False)
        a[idx] = np.nan
    return a


def _assert_dispatched_exact(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.nan_to_num(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert got.dtype == stock.dtype and got.shape == stock.shape
    assert np.array_equal(got, stock)
    return got, stock


def _assert_refused_exact(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (decision, reason)
    got = np.nan_to_num(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert np.array_equal(got, stock)
    return got, stock


def test_dispatch_1d_nan_percent_bit_identical_and_copy():
    a = _mk_1d(20000, nan_frac=0.01, seed=1)
    a_orig = a.copy()
    got, stock = _assert_dispatched_exact((a,), {})
    assert not np.isnan(got).any()
    assert got is not a
    got[0] = 12345.0
    assert np.array_equal(a, a_orig, equal_nan=True)


def test_dispatch_nan_inf_mix_bit_identical():
    a = _mk_1d(15000, nan_frac=0.01, seed=2)
    rng = np.random.default_rng(2)
    idx = rng.choice(15000, size=50, replace=False)
    a[idx[:25]] = np.inf
    a[idx[25:]] = -np.inf
    got, stock = _assert_dispatched_exact((a,), {})
    info = np.finfo(np.float64)
    assert (got == info.max).sum() == 25
    assert (got == info.min).sum() == 25


def test_dispatch_clean_array_bit_identical_and_copy():
    a = _mk_1d(12000, nan_frac=0.0, seed=3)
    a_orig = a.copy()
    got, stock = _assert_dispatched_exact((a,), {})
    assert got is not a
    got[0] = -999.0
    assert np.array_equal(a, a_orig)


def test_dispatch_all_inf_bit_identical():
    a = np.empty(10000, dtype=np.float64)
    a[::2] = np.inf
    a[1::2] = -np.inf
    _assert_dispatched_exact((a,), {})


def test_dispatch_2d_with_nans_bit_identical():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((200, 100)).astype(np.float64)
    idx0 = rng.integers(0, 200, size=100)
    idx1 = rng.integers(0, 100, size=100)
    a[idx0, idx1] = np.nan
    _assert_dispatched_exact((a,), {})


def test_nan_kwarg_scalar_override_now_dispatched():
    # Historically a refusal; the module now accepts a plain-float nan=
    # override (see kwargs-override section below for full coverage).
    a = _mk_1d(20000, nan_frac=0.01, seed=5)
    _assert_dispatched_exact((a,), {"nan": 5.0})


def test_posinf_kwarg_scalar_override_now_dispatched():
    a = _mk_1d(20000, nan_frac=0.0, seed=6)
    a[100] = np.inf
    _assert_dispatched_exact((a,), {"posinf": 1e300})


def test_neginf_kwarg_scalar_override_now_dispatched():
    a = _mk_1d(20000, nan_frac=0.0, seed=7)
    a[100] = -np.inf
    _assert_dispatched_exact((a,), {"neginf": -1.0})


def test_refusal_copy_false_kwarg_inplace_semantics():
    a = _mk_1d(20000, nan_frac=0.01, seed=8)
    a_for_stock = a.copy()

    decision, reason = GEARBOX.decide(OP, (a,), {"copy": False})
    assert decision == "stock", (decision, reason)

    result = np.nan_to_num(a, copy=False)
    assert result is a
    assert not np.isnan(a).any()

    stock_result = _stock(a_for_stock, copy=False)
    assert stock_result is a_for_stock
    assert np.array_equal(result, stock_result)


def test_refusal_float32():
    a = _mk_1d(20000, nan_frac=0.01, seed=9).astype(np.float32)
    _assert_refused_exact((a,), {})


def test_refusal_complex128():
    real = _mk_1d(20000, nan_frac=0.01, seed=10)
    imag = _mk_1d(20000, nan_frac=0.01, seed=11)
    a = (real + 1j * imag).astype(np.complex128)
    _assert_refused_exact((a,), {})


def test_refusal_int64():
    rng = np.random.default_rng(12)
    a = rng.integers(-1000, 1000, size=20000).astype(np.int64)
    _assert_refused_exact((a,), {})


def test_refusal_size_below_floor():
    a = _mk_1d(SIZE_FLOOR - 1, nan_frac=0.01, seed=13)
    _assert_refused_exact((a,), {})


def test_refusal_python_list_input():
    a_list = [1.0, float("nan"), 3.0, float("inf"), -float("inf")] * 4000
    decision, reason = GEARBOX.decide(OP, (a_list,), {})
    assert decision == "stock", (decision, reason)
    got = np.nan_to_num(a_list)
    stock = _stock(a_list)
    assert np.array_equal(got, stock)


def test_kill_switch_restores_stock_routing():
    a = _mk_1d(20000, nan_frac=0.01, seed=14)
    decision, _ = GEARBOX.decide(OP, (a,), {})
    assert decision == PATH
    pyoverdrive.disable_path(PATH)
    try:
        decision, _ = GEARBOX.decide(OP, (a,), {})
        assert decision == "stock"
        assert np.array_equal(np.nan_to_num(a), _stock(a))
    finally:
        pyoverdrive.enable_path(PATH)


# ---------------------------------------------------------------------------
# kwargs-override coverage: nan=/posinf=/neginf= scalar overrides (module
# docstring "Correctness contract" second bullet). Every case below runs on
# both a NaN/+inf/-inf-salted array and a clean array.
# ---------------------------------------------------------------------------


def _mk_salted(n, seed=100):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n).astype(np.float64)
    idx = rng.choice(n, size=max(30, n // 200), replace=False)
    third = len(idx) // 3
    a[idx[:third]] = np.nan
    a[idx[third : 2 * third]] = np.inf
    a[idx[2 * third :]] = -np.inf
    return a


def _kwarg_array(kind, seed):
    if kind == "salted":
        return _mk_salted(20000, seed=seed)
    return _mk_1d(20000, nan_frac=0.0, seed=seed)


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_nan_override_bit_identical(kind):
    a = _kwarg_array(kind, seed=20)
    _assert_dispatched_exact((a,), {"nan": 1.5})


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_nan_and_posinf_override_bit_identical(kind):
    a = _kwarg_array(kind, seed=21)
    _assert_dispatched_exact((a,), {"nan": -2.5, "posinf": 999.0})


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_posinf_and_neginf_override_bit_identical(kind):
    a = _kwarg_array(kind, seed=22)
    _assert_dispatched_exact((a,), {"posinf": 100.0, "neginf": -100.0})


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_nan_equals_inf_mask_ordering_trap(kind):
    # nan=np.inf: former-NaN slots must hold +inf (the override value),
    # never get re-swept by the isinf-based posinf replacement that runs
    # afterwards - both masks are computed from the ORIGINAL values, per
    # the module docstring's "Correctness contract".
    a = _kwarg_array(kind, seed=23)
    had_nan = np.isnan(a)
    got, stock = _assert_dispatched_exact((a,), {"nan": np.inf})
    if had_nan.any():
        assert np.isposinf(got[had_nan]).all()
    assert np.array_equal(got, stock)


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_nan_python_int(kind):
    a = _kwarg_array(kind, seed=24)
    _assert_dispatched_exact((a,), {"nan": 7})


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_nan_python_bool(kind):
    a = _kwarg_array(kind, seed=25)
    _assert_dispatched_exact((a,), {"nan": True})


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_posinf_none_explicit(kind):
    a = _kwarg_array(kind, seed=26)
    _assert_dispatched_exact((a,), {"posinf": None, "nan": 3.0})


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_np_float64_value_accepted(kind):
    a = _kwarg_array(kind, seed=27)
    _assert_dispatched_exact((a,), {"nan": np.float64(4.25)})


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_np_float32_value_refused(kind):
    a = _kwarg_array(kind, seed=28)
    _assert_refused_exact((a,), {"nan": np.float32(4.25)})


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_array_like_value_refused(kind):
    a = _kwarg_array(kind, seed=29)
    _assert_refused_exact((a,), {"nan": np.array(1.5)})


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_copy_true_refused(kind):
    a = _kwarg_array(kind, seed=30)
    _assert_refused_exact((a,), {"copy": True})


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_copy_false_refused_with_override(kind):
    # copy=False alongside a value override: still refused, and stock's
    # in-place semantics must be preserved on the direct (non-dispatched)
    # path, same as the existing bare copy=False refusal test above.
    a = _kwarg_array(kind, seed=31)
    a_for_stock = a.copy()

    decision, reason = GEARBOX.decide(OP, (a,), {"nan": 2.0, "copy": False})
    assert decision == "stock", (decision, reason)

    result = np.nan_to_num(a, nan=2.0, copy=False)
    assert result is a

    stock_result = _stock(a_for_stock, nan=2.0, copy=False)
    assert stock_result is a_for_stock
    assert np.array_equal(result, stock_result)


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_unknown_kwarg_refused(kind):
    # A kwarg stock itself doesn't accept at all: the dispatch decision
    # must still be "stock", and both paths must fail the same way.
    a = _kwarg_array(kind, seed=32)
    decision, reason = GEARBOX.decide(OP, (a,), {"foo": 1})
    assert decision == "stock", (decision, reason)
    with pytest.raises(TypeError):
        np.nan_to_num(a, foo=1)
    with pytest.raises(TypeError):
        _stock(a, foo=1)


@pytest.mark.parametrize("kind", ["salted", "clean"])
def test_kwarg_below_size_floor_refused(kind):
    n = SIZE_FLOOR - 1
    a = _mk_salted(n, seed=33) if kind == "salted" else _mk_1d(n, nan_frac=0.0, seed=33)
    _assert_refused_exact((a,), {"nan": 1.5})
