"""Differential tests: matmul_split_complex fast path vs stock numpy.matmul.

Contract (src/pyoverdrive/fastpaths/matmul_split_complex.py): applies only to
matmul(C, R) with no kwargs, where C is a plain 2-D complex ndarray and R a
plain 2-D real ndarray of the paired dtype (complex128 with float64, or
complex64 with float32), inner dimensions matching, m <= M_MAX, n >= N_MIN,
q >= Q_MIN, and both operands all-finite. Dispatch runs two real GEMMs
(out.real = C.real @ R, out.imag = C.imag @ R) instead of stock's upcast
complex GEMM, so results agree to BLAS-rounding scale (numeric mode) rather
than bit-identity.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.matmul_split_complex import M_MAX, N_MIN, Q_MIN

OP = "numpy.matmul"
PATH = "matmul_split_complex"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _mats(m, n, q, seed, c_dtype=np.complex128, r_dtype=np.float64, order="C", zero_imag=False):
    rng = np.random.default_rng(seed)
    if zero_imag:
        c = rng.uniform(0.5, 1.5, size=(m, n)).astype(c_dtype)
    else:
        c = (rng.uniform(0.5, 1.5, size=(m, n)) + 1j * rng.uniform(0.5, 1.5, size=(m, n))).astype(c_dtype)
    r = rng.uniform(0.5, 1.5, size=(n, q)).astype(r_dtype)
    if order == "F":
        c = np.asfortranarray(c)
        r = np.asfortranarray(r)
    return c, r


def _tol(stock, complex64=False):
    s = max(1.0, float(np.abs(stock).max()))
    if complex64:
        return 1e-5, 1e-5 * s
    return 1e-12, 1e-12 * s


def _assert_close(got, stock, complex64=False):
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    rtol, atol = _tol(stock, complex64=complex64)
    assert np.allclose(got, stock, rtol=rtol, atol=atol)


def _assert_dispatched_close(c, r, complex64=False):
    decision, reason = GEARBOX.decide(OP, (c, r), {})
    assert decision == PATH, (decision, reason)
    got = np.matmul(c, r)
    stock = _stock(c, r)
    _assert_close(got, stock, complex64=complex64)
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.matmul(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert np.array_equal(got, stock)
    return got


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.matmul(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + numeric equality
# ---------------------------------------------------------------------------

def test_dispatch_large_complex128():
    c, r = _mats(64, 1200, 600, 1)
    got, stock = _assert_dispatched_close(c, r)
    assert got.dtype == np.complex128
    assert got.shape == (64, 600)


def test_dispatch_large_complex64():
    c, r = _mats(64, 1200, 600, 2, c_dtype=np.complex64, r_dtype=np.float32)
    got, stock = _assert_dispatched_close(c, r, complex64=True)
    assert got.dtype == np.complex64
    assert got.shape == (64, 600)


def test_dispatch_m_equals_max_dispatches():
    c, r = _mats(M_MAX, N_MIN, Q_MIN, 3)
    _assert_dispatched_close(c, r)


def test_refusal_m_max_plus_one():
    c, r = _mats(M_MAX + 1, N_MIN, Q_MIN, 4)
    _assert_refused_equal((c, r), {})


def test_dispatch_n_equals_min_dispatches():
    c, r = _mats(M_MAX, N_MIN, Q_MIN, 5)
    _assert_dispatched_close(c, r)


def test_refusal_n_min_minus_one():
    c, r = _mats(M_MAX, N_MIN - 1, Q_MIN, 6)
    _assert_refused_equal((c, r), {})


def test_dispatch_q_equals_min_dispatches():
    c, r = _mats(M_MAX, N_MIN, Q_MIN, 7)
    _assert_dispatched_close(c, r)


def test_refusal_q_min_minus_one():
    c, r = _mats(M_MAX, N_MIN, Q_MIN - 1, 8)
    _assert_refused_equal((c, r), {})


def test_dispatch_f_order_operands():
    # predicate does not require contiguity; F-order views are strided
    # either way and still dispatch and match.
    c, r = _mats(64, 1200, 600, 9, order="F")
    assert c.flags.f_contiguous and not c.flags.c_contiguous
    assert r.flags.f_contiguous and not r.flags.c_contiguous
    _assert_dispatched_close(c, r)


def test_dispatch_zero_imaginary_c():
    c, r = _mats(64, 1200, 600, 10, zero_imag=True)
    assert c.dtype == np.complex128
    assert np.all(c.imag == 0.0)
    _assert_dispatched_close(c, r)


# ---------------------------------------------------------------------------
# 2. refusal routes
# ---------------------------------------------------------------------------

def test_refusal_reverse_direction_real_at_complex():
    # reverse: real 2-D @ complex 2-D -- the split path only handles C @ R.
    rng = np.random.default_rng(11)
    a_real = rng.uniform(0.5, 1.5, size=(64, 1200)).astype(np.float64)
    b_complex = (rng.uniform(0.5, 1.5, size=(1200, 600)) + 1j * rng.uniform(0.5, 1.5, size=(1200, 600))).astype(
        np.complex128
    )
    _assert_refused_equal((a_real, b_complex), {})


def test_refusal_complex128_c_with_float32_r():
    c, _ = _mats(64, 1200, 600, 13)
    r32 = np.random.default_rng(14).uniform(0.5, 1.5, size=(1200, 600)).astype(np.float32)
    _assert_refused_equal((c, r32), {})


def test_refusal_complex64_c_with_float64_r():
    c, _ = _mats(64, 1200, 600, 15, c_dtype=np.complex64, r_dtype=np.float32)
    r64 = np.random.default_rng(16).uniform(0.5, 1.5, size=(1200, 600)).astype(np.float64)
    _assert_refused_equal((c, r64), {})


def test_refusal_both_complex_operands():
    c, r = _mats(64, 1200, 600, 17)
    r_complex = (r + 1j * np.random.default_rng(18).uniform(0.5, 1.5, size=r.shape)).astype(np.complex128)
    _assert_refused_equal((c, r_complex), {})


def test_refusal_both_real_operands():
    rng = np.random.default_rng(19)
    a = rng.uniform(0.5, 1.5, size=(64, 1200)).astype(np.float64)
    b = rng.uniform(0.5, 1.5, size=(1200, 600)).astype(np.float64)
    decision, reason = GEARBOX.decide(OP, (a, b), {})
    assert decision == "stock", (decision, reason)
    got = np.matmul(a, b)
    stock = _stock(a, b)
    assert type(got) is type(stock)
    assert got.dtype == stock.dtype
    assert np.array_equal(got, stock)


def test_refusal_1d_r_vector():
    c, _ = _mats(64, 1200, 600, 20)
    r_vec = np.random.default_rng(21).uniform(0.5, 1.5, size=1200).astype(np.float64)
    _assert_refused_equal((c, r_vec), {})


def test_refusal_3d_batched_c():
    rng = np.random.default_rng(22)
    c3 = (rng.uniform(0.5, 1.5, size=(4, 64, 1200)) + 1j * rng.uniform(0.5, 1.5, size=(4, 64, 1200))).astype(
        np.complex128
    )
    r = rng.uniform(0.5, 1.5, size=(1200, 600)).astype(np.float64)
    _assert_refused_equal((c3, r), {})


def test_refusal_inner_dimension_mismatch_raises_same_exception():
    c, _ = _mats(64, 1200, 600, 23)
    r_wrong = np.random.default_rng(24).uniform(0.5, 1.5, size=(999, 600)).astype(np.float64)
    _assert_refused_raises((c, r_wrong), {})
    with pytest.raises(ValueError):
        _stock(c, r_wrong)


def test_refusal_out_kwarg():
    c, r = _mats(64, 1200, 600, 25)
    out_got = np.empty((64, 600), dtype=np.complex128)
    out_stock = np.empty((64, 600), dtype=np.complex128)
    decision, reason = GEARBOX.decide(OP, (c, r), {"out": out_got})
    assert decision == "stock", (decision, reason)
    got = np.matmul(c, r, out=out_got)
    stock = _stock(c, r, out=out_stock)
    assert got is out_got
    assert np.array_equal(got, stock)


def test_refusal_c_containing_nan():
    c, r = _mats(64, 1200, 600, 26)
    c[3, 3] = complex(np.nan, 0.0)
    decision, reason = GEARBOX.decide(OP, (c, r), {})
    assert decision == "stock", (decision, reason)
    got = np.matmul(c, r)
    stock = _stock(c, r)
    assert np.array_equal(np.isnan(got), np.isnan(stock))
    finite = ~np.isnan(got) & ~np.isnan(stock)
    assert np.array_equal(got[finite], stock[finite])


def test_refusal_r_containing_inf():
    c, r = _mats(64, 1200, 600, 27)
    r[5, 5] = np.inf
    decision, reason = GEARBOX.decide(OP, (c, r), {})
    assert decision == "stock", (decision, reason)
    got = np.matmul(c, r)
    stock = _stock(c, r)
    assert np.array_equal(np.isnan(got), np.isnan(stock))
    finite = ~np.isnan(got) & ~np.isnan(stock)
    assert np.array_equal(got[finite], stock[finite])


def test_refusal_python_list_operands():
    rng = np.random.default_rng(28)
    c = (rng.uniform(0.5, 1.5, size=(64, 1200)) + 1j * rng.uniform(0.5, 1.5, size=(64, 1200))).tolist()
    r = rng.uniform(0.5, 1.5, size=(1200, 600)).tolist()
    _assert_refused_equal((c, r), {})


# ---------------------------------------------------------------------------
# 3. kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_restores_stock_routing():
    c, r = _mats(64, 1200, 600, 29)
    decision, reason = GEARBOX.decide(OP, (c, r), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (c, r), {})
        assert decision == "stock", (decision, reason)
        got = np.matmul(c, r)
        stock = _stock(c, r)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)
