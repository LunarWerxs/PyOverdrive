"""Differential tests: matmul_int_blas / dot_int_blas fast paths vs stock numpy.

Covers numpy.matmul (path matmul_int_blas) and numpy.dot (path dot_int_blas):
same predicate/run (cast int64/int32 to float64, BLAS, cast back), exact
under k * max|A| * max|B| < 2**53 (int64) / 2**31 (int32), min(m,k,n) >= 50.

numpy.matmul is ALSO patched by matmul_split_complex, so every dispatched
assertion below pins the decision to the int-blas path name specifically,
not just "not stock".
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

MATMUL_OP = "numpy.matmul"
DOT_OP = "numpy.dot"
MATMUL_PATH = "matmul_int_blas"
DOT_PATH = "dot_int_blas"

CASES = [
    (MATMUL_OP, MATMUL_PATH, np.matmul),
    (DOT_OP, DOT_PATH, np.dot),
]

CASES_OP_FN = [
    (MATMUL_OP, np.matmul),
    (DOT_OP, np.dot),
]


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([MATMUL_OP, DOT_OP])
    yield
    pyoverdrive.disable()


def _stock(op, *args, **kwargs):
    return GEARBOX.stock_fn(op)(*args, **kwargs)


def _assert_dispatched_exact(op, path, fn, args, kwargs):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == path, (op, decision, reason)
    got = fn(*args, **kwargs)
    stock = _stock(op, *args, **kwargs)
    assert got.dtype == stock.dtype and got.shape == stock.shape
    assert np.array_equal(got, stock)


def _assert_refused_exact(op, fn, args, kwargs):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == "stock", (op, decision, reason)
    got = fn(*args, **kwargs)
    stock = _stock(op, *args, **kwargs)
    assert got.dtype == stock.dtype and got.shape == stock.shape
    assert np.array_equal(got, stock)


@pytest.mark.parametrize("op,path,fn", CASES)
def test_dispatch_int64_square_bit_identical(op, path, fn):
    rng = np.random.default_rng(1)
    a = rng.integers(-1000, 1000, size=(100, 100), dtype=np.int64)
    b = rng.integers(-1000, 1000, size=(100, 100), dtype=np.int64)
    _assert_dispatched_exact(op, path, fn, (a, b), {})


@pytest.mark.parametrize("op,path,fn", CASES)
def test_dispatch_int32_nonsquare_bit_identical(op, path, fn):
    rng = np.random.default_rng(2)
    a = rng.integers(-1000, 1000, size=(60, 80), dtype=np.int32)
    b = rng.integers(-1000, 1000, size=(80, 70), dtype=np.int32)
    _assert_dispatched_exact(op, path, fn, (a, b), {})


@pytest.mark.parametrize("op,path,fn", CASES)
def test_dispatch_negative_heavy_bit_identical(op, path, fn):
    rng = np.random.default_rng(3)
    a = rng.integers(-1000, -1, size=(70, 60), dtype=np.int64)
    b = rng.integers(-1000, -1, size=(60, 90), dtype=np.int64)
    _assert_dispatched_exact(op, path, fn, (a, b), {})


@pytest.mark.parametrize("op,path,fn", CASES)
def test_dispatch_near_bound_admit_int64(op, path, fn):
    k = 50
    m = k
    n = k
    bound = 2**53
    big = int((bound / k) ** 0.5) - 1
    assert k * big * big < bound
    rng = np.random.default_rng(4)
    a = rng.integers(-5, 5, size=(m, k), dtype=np.int64)
    b = rng.integers(-5, 5, size=(k, n), dtype=np.int64)
    a[0, 0] = big
    a[1, 1] = -big
    b[0, 0] = big
    b[1, 1] = -big
    _assert_dispatched_exact(op, path, fn, (a, b), {})


@pytest.mark.parametrize("op,fn", CASES_OP_FN)
def test_refusal_over_bound_int64(op, fn):
    k = 50
    m = k
    n = k
    bound = 2**53
    big = int((bound / k) ** 0.5) + 2
    assert k * big * big >= bound
    rng = np.random.default_rng(5)
    a = rng.integers(-5, 5, size=(m, k), dtype=np.int64)
    b = rng.integers(-5, 5, size=(k, n), dtype=np.int64)
    a[0, 0] = big
    b[0, 0] = big
    _assert_refused_exact(op, fn, (a, b), {})


@pytest.mark.parametrize("op,fn", CASES_OP_FN)
def test_refusal_over_bound_int32(op, fn):
    k = 50
    m = k
    n = k
    rng = np.random.default_rng(6)
    a = rng.integers(-40000, 40000, size=(m, k), dtype=np.int32)
    b = rng.integers(-40000, 40000, size=(k, n), dtype=np.int32)
    bound = 2**31
    assert k * 40000 * 40000 >= bound
    _assert_refused_exact(op, fn, (a, b), {})


@pytest.mark.parametrize("op,fn", CASES_OP_FN)
def test_int64_min_present_safe(op, fn):
    k = 60
    rng = np.random.default_rng(7)
    a = rng.integers(-100, 100, size=(k, k), dtype=np.int64)
    b = rng.integers(-100, 100, size=(k, k), dtype=np.int64)
    a[0, 0] = np.iinfo(np.int64).min
    decision, reason = GEARBOX.decide(op, (a, b), {})
    assert decision in ("stock", MATMUL_PATH, DOT_PATH), (decision, reason)
    got = fn(a, b)
    stock = _stock(op, a, b)
    assert got.dtype == stock.dtype and got.shape == stock.shape
    assert np.array_equal(got, stock)


@pytest.mark.parametrize("op,fn", CASES_OP_FN)
def test_refusal_mixed_dtype(op, fn):
    rng = np.random.default_rng(8)
    a = rng.integers(-1000, 1000, size=(60, 60), dtype=np.int64)
    b = rng.integers(-1000, 1000, size=(60, 60), dtype=np.int32)
    _assert_refused_exact(op, fn, (a, b), {})


@pytest.mark.parametrize("op,fn", CASES_OP_FN)
def test_refusal_float_operands(op, fn):
    rng = np.random.default_rng(9)
    a = rng.standard_normal((60, 60))
    b = rng.standard_normal((60, 60))
    _assert_refused_exact(op, fn, (a, b), {})


@pytest.mark.parametrize("op,fn", CASES_OP_FN)
def test_refusal_1d_vector(op, fn):
    rng = np.random.default_rng(10)
    a = rng.integers(-1000, 1000, size=(60, 60), dtype=np.int64)
    v = rng.integers(-1000, 1000, size=(60,), dtype=np.int64)
    _assert_refused_exact(op, fn, (a, v), {})


@pytest.mark.parametrize("op,fn", CASES_OP_FN)
def test_refusal_3d_stack(op, fn):
    rng = np.random.default_rng(11)
    a = rng.integers(-1000, 1000, size=(4, 60, 60), dtype=np.int64)
    b = rng.integers(-1000, 1000, size=(4, 60, 60), dtype=np.int64)
    _assert_refused_exact(op, fn, (a, b), {})


@pytest.mark.parametrize("op,fn", CASES_OP_FN)
def test_refusal_below_min_dim(op, fn):
    rng = np.random.default_rng(12)
    a = rng.integers(-1000, 1000, size=(30, 30), dtype=np.int64)
    b = rng.integers(-1000, 1000, size=(30, 30), dtype=np.int64)
    _assert_refused_exact(op, fn, (a, b), {})


@pytest.mark.parametrize("op,fn", CASES_OP_FN)
def test_refusal_out_kwarg(op, fn):
    rng = np.random.default_rng(13)
    a = rng.integers(-1000, 1000, size=(60, 60), dtype=np.int64)
    b = rng.integers(-1000, 1000, size=(60, 60), dtype=np.int64)
    out_got = np.empty((60, 60), dtype=np.int64)
    out_stock = np.empty((60, 60), dtype=np.int64)
    decision, reason = GEARBOX.decide(op, (a, b), {"out": out_got})
    assert decision == "stock", (op, decision, reason)
    got = fn(a, b, out=out_got)
    stock = _stock(op, a, b, out=out_stock)
    assert got.dtype == stock.dtype and got.shape == stock.shape
    assert np.array_equal(got, stock)
    assert np.array_equal(out_got, out_stock)


def test_kill_switch_restores_stock_routing():
    rng = np.random.default_rng(14)
    a = rng.integers(-1000, 1000, size=(60, 60), dtype=np.int64)
    b = rng.integers(-1000, 1000, size=(60, 60), dtype=np.int64)

    decision, _ = GEARBOX.decide(MATMUL_OP, (a, b), {})
    assert decision == MATMUL_PATH
    decision, _ = GEARBOX.decide(DOT_OP, (a, b), {})
    assert decision == DOT_PATH

    pyoverdrive.disable_path(MATMUL_PATH)
    pyoverdrive.disable_path(DOT_PATH)
    try:
        decision, _ = GEARBOX.decide(MATMUL_OP, (a, b), {})
        assert decision == "stock"
        decision, _ = GEARBOX.decide(DOT_OP, (a, b), {})
        assert decision == "stock"
        assert np.array_equal(np.matmul(a, b), _stock(MATMUL_OP, a, b))
        assert np.array_equal(np.dot(a, b), _stock(DOT_OP, a, b))
    finally:
        pyoverdrive.enable_path(MATMUL_PATH)
        pyoverdrive.enable_path(DOT_PATH)
