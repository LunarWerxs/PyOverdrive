"""Differential tests: dot_mixed_view fast path vs stock numpy.dot.

Contract (src/pyoverdrive/fastpaths/dot_mixed_view.py): applies only to
dot(A, b) with no kwargs, where A is a plain 2-D float64 ndarray, b a plain
1-D C-contiguous complex128 ndarray (the view needs contiguity), and
A.shape[1] == b.size, above a size floor of A.size >= 20_000. Dispatch views
b as an (m, 2) float64 matrix, runs one real matmul through the `@`
operator, and views the (n, 2) result back as complex128 -- exactly the two
real accumulations stock's complex GEMV performs, so results agree to BLAS
rounding (numeric mode, tight scaled tolerance) rather than bit-identity.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

OP = "numpy.dot"
PATH = "dot_mixed_view"
FLOOR = 20_000


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _mats(shape_a, seed, b_kind="mixed", order="C"):
    rng = np.random.default_rng(seed)
    m, k = shape_a
    a = rng.uniform(0.5, 1.5, size=shape_a).astype(np.float64)
    if order == "F":
        a = np.asfortranarray(a)
    if b_kind == "mixed":
        b = (rng.uniform(0.5, 1.5, size=k) + 1j * rng.uniform(0.5, 1.5, size=k)).astype(np.complex128)
    elif b_kind == "imag":
        b = (1j * rng.uniform(0.5, 1.5, size=k)).astype(np.complex128)
    elif b_kind == "zeros":
        b = (rng.uniform(0.5, 1.5, size=k) + 1j * rng.uniform(0.5, 1.5, size=k)).astype(np.complex128)
        b[::5] = 0.0
    elif b_kind == "wide":
        mag = 10.0 ** rng.uniform(-3, 3, size=k)
        sign_r = rng.choice([-1.0, 1.0], size=k)
        sign_i = rng.choice([-1.0, 1.0], size=k)
        re = mag * sign_r * rng.uniform(0.1, 1.0, size=k)
        im = mag * sign_i * rng.uniform(0.1, 1.0, size=k)
        b = (re + 1j * im).astype(np.complex128)
    else:
        raise ValueError(b_kind)
    return a, b


def _tol(stock):
    s = max(1.0, float(np.abs(stock).max()))
    return 1e-12, 1e-12 * s


def _assert_close(got, stock):
    assert got.dtype == np.complex128
    assert got.shape == stock.shape
    rtol, atol = _tol(stock)
    assert np.allclose(got, stock, rtol=rtol, atol=atol)


def _assert_dispatched_close(a, b):
    decision, reason = GEARBOX.decide(OP, (a, b), {})
    assert decision == PATH, (decision, reason)
    got = np.dot(a, b)
    stock = _stock(a, b)
    _assert_close(got, stock)
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.dot(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert np.array_equal(got, stock)
    return got


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.dot(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + numeric equality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shape, seed",
    [((200, 100), 1), ((1000, 500), 2), ((100, 200), 3)],
    ids=["tall-200x100", "tall-1000x500", "wide-100x200"],
)
def test_dispatch_shapes_numeric_equality(shape, seed):
    a, b = _mats(shape, seed, b_kind="mixed")
    got, stock = _assert_dispatched_close(a, b)
    assert got.dtype == np.complex128
    assert got.shape == (shape[0],)


def test_dispatch_f_order_a_still_dispatches():
    # only b needs contiguity; F-order A is not excluded by the predicate.
    a, b = _mats((300, 150), 4, b_kind="mixed", order="F")
    assert a.flags.f_contiguous and not a.flags.c_contiguous
    _assert_dispatched_close(a, b)


def test_dispatch_wide_magnitude_values():
    a, b = _mats((250, 120), 5, b_kind="wide")
    _assert_dispatched_close(a, b)


def test_dispatch_pure_imaginary_b():
    a, b = _mats((200, 100), 6, b_kind="imag")
    _assert_dispatched_close(a, b)


def test_dispatch_b_containing_zeros():
    a, b = _mats((200, 100), 7, b_kind="zeros")
    assert np.any(b == 0)
    _assert_dispatched_close(a, b)


# ---------------------------------------------------------------------------
# 2. special values: inf / nan
# ---------------------------------------------------------------------------

def test_special_values_inf_nan_in_a():
    # non-finite A is refused (BLAS complex multiply mixes components, so
    # stock turns lone infs into NaN the clean real route would not);
    # parity is stock-vs-stock and exact
    a, b = _mats((200, 100), 8, b_kind="mixed")
    a[3, 3] = np.inf
    a[7, 9] = np.nan
    decision, reason = GEARBOX.decide(OP, (a, b), {})
    assert decision == "stock", (decision, reason)
    got = np.dot(a, b)
    stock = _stock(a, b)
    assert np.array_equal(np.isnan(got), np.isnan(stock))
    finite = ~np.isnan(got) & ~np.isnan(stock)
    rtol, atol = _tol(np.where(np.isfinite(stock), stock, 0.0))
    assert np.allclose(got[finite], stock[finite], rtol=rtol, atol=atol)


def test_special_values_inf_nan_in_b_are_refused():
    # Non-finite entries in b DIVERGE if dispatched: stock's complex GEMV
    # multiplies by A's zero imaginary part, so 0 * inf breeds NaN in both
    # components where the view route's two clean real accumulations
    # produce none (this test found it). The predicate now refuses
    # non-finite b, so stock answers and parity is exact.
    a, b = _mats((200, 100), 9, b_kind="mixed")
    b[2] = complex(np.inf, 0.0)
    b[5] = complex(0.0, np.nan)
    decision, reason = GEARBOX.decide(OP, (a, b), {})
    assert decision == "stock", (decision, reason)
    got = np.dot(a, b)
    stock = _stock(a, b)
    assert np.array_equal(np.isnan(got), np.isnan(stock))
    finite = ~np.isnan(got) & ~np.isnan(stock)
    assert np.array_equal(got[finite], stock[finite])
    # non-finite entries in A are refused too: BLAS's complex multiply
    # mixes components internally, so a single inf became NaN on stock
    # where the clean real route kept inf (measured by this test's first
    # version)
    a2, b2 = _mats((200, 100), 10, b_kind="mixed")
    a2[3, 7] = np.inf
    a2[8, 2] = np.nan
    decision2, _ = GEARBOX.decide(OP, (a2, b2), {})
    assert decision2 == "stock"
    got2 = np.dot(a2, b2)
    stock2 = _stock(a2, b2)
    assert np.array_equal(np.isnan(got2), np.isnan(stock2))


# ---------------------------------------------------------------------------
# 3. refusal routes
# ---------------------------------------------------------------------------

def test_refusal_just_below_floor():
    a, b = _mats((199, 100), 10, b_kind="mixed")  # size 19_900 < 20_000
    assert a.size < FLOOR
    _assert_refused_equal((a, b), {})


def test_dispatch_exact_floor_boundary():
    a, b = _mats((200, 100), 11, b_kind="mixed")  # size == 20_000 exactly
    assert a.size == FLOOR
    _assert_dispatched_close(a, b)


def test_refusal_noncontiguous_b_slice_view():
    a, _ = _mats((200, 200), 12, b_kind="mixed")
    rng = np.random.default_rng(120)
    b_full = (rng.uniform(0.5, 1.5, size=400) + 1j * rng.uniform(0.5, 1.5, size=400)).astype(np.complex128)
    b = b_full[::2]  # length 200, matches a.shape[1], but not C-contiguous
    assert b.size == a.shape[1]
    assert not b.flags.c_contiguous
    _assert_refused_equal((a, b), {})


def test_refusal_complex_a_real_b():
    a, b_seed = _mats((200, 100), 13, b_kind="mixed")
    a_complex = a.astype(np.complex128)
    b_real = b_seed.real.copy()
    _assert_refused_equal((a_complex, b_real), {})


def test_refusal_float32_a():
    a, b = _mats((200, 100), 14, b_kind="mixed")
    a32 = a.astype(np.float32)
    _assert_refused_equal((a32, b), {})


def test_refusal_complex64_b():
    a, b = _mats((200, 100), 15, b_kind="mixed")
    b64 = b.astype(np.complex64)
    _assert_refused_equal((a, b64), {})


def test_refusal_1d_a():
    rng = np.random.default_rng(16)
    a = rng.uniform(0.5, 1.5, size=20_000).astype(np.float64)
    b = (rng.uniform(0.5, 1.5, size=20_000) + 1j * rng.uniform(0.5, 1.5, size=20_000)).astype(np.complex128)
    _assert_refused_equal((a, b), {})


def test_refusal_3d_a():
    rng = np.random.default_rng(17)
    a = rng.uniform(0.5, 1.5, size=(50, 50, 100)).astype(np.float64)
    b = (rng.uniform(0.5, 1.5, size=100) + 1j * rng.uniform(0.5, 1.5, size=100)).astype(np.complex128)
    _assert_refused_equal((a, b), {})


def test_refusal_length_mismatch_raises_valueerror_both_routes():
    a, _ = _mats((200, 100), 18, b_kind="mixed")
    rng = np.random.default_rng(19)
    b_wrong = (rng.uniform(0.5, 1.5, size=50) + 1j * rng.uniform(0.5, 1.5, size=50)).astype(np.complex128)
    _assert_refused_raises((a, b_wrong), {})
    with pytest.raises(ValueError):
        _stock(a, b_wrong)


def test_refusal_out_kwarg():
    a, b = _mats((200, 100), 20, b_kind="mixed")
    out_got = np.empty(200, dtype=np.complex128)
    out_stock = np.empty(200, dtype=np.complex128)
    decision, reason = GEARBOX.decide(OP, (a, b), {"out": out_got})
    assert decision == "stock", (decision, reason)
    got = np.dot(a, b, out=out_got)
    stock = _stock(a, b, out=out_stock)
    assert got is out_got
    assert np.array_equal(got, stock)


def test_refusal_python_list_operands():
    rng = np.random.default_rng(21)
    a = rng.uniform(0.5, 1.5, size=(200, 100)).tolist()
    b = (rng.uniform(0.5, 1.5, size=100) + 1j * rng.uniform(0.5, 1.5, size=100)).tolist()
    _assert_refused_equal((a, b), {})


def test_kill_switch_restores_stock_routing():
    a, b = _mats((200, 100), 22, b_kind="mixed")
    decision, reason = GEARBOX.decide(OP, (a, b), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (a, b), {})
        assert decision == "stock", (decision, reason)
        got = np.dot(a, b)
        stock = _stock(a, b)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)
