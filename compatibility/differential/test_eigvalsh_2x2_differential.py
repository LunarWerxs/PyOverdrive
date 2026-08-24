"""Differential tests: eigvalsh_2x2_closed fast path vs stock numpy.linalg.eigvalsh.

Contract (src/pyoverdrive/fastpaths/eigvalsh_2x2.py): applies only to
eigvalsh(a) / eigvalsh(a, UPLO='L') where a is a plain float64/float32
ndarray shaped (..., 2, 2) with ndim >= 3, at least BATCH_MIN matrices, and
every element finite. UPLO='U', 2-D single matrices, complex Hermitian
input, other dtypes, and non-finite values all stay on stock. Both stock
(UPLO='L') and the fast path read only the lower triangle, so an
asymmetric upper triangle must not change the result. Comparison mode is
numeric: per-matrix scaled tolerance against the maximum absolute
eigenvalue in that matrix.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.eigvalsh_2x2 import BATCH_MIN

OP = "numpy.linalg.eigvalsh"
PATH = "eigvalsh_2x2_closed"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _symmetric_batch(batch_shape, seed, dtype=np.float64):
    """Build a batch of symmetric 2x2 matrices: a @ swapaxes(a) + 0.1*eye(2)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(*batch_shape, 2, 2)).astype(dtype)
    sym = a @ np.swapaxes(a, -1, -2) + (0.1 * np.eye(2, dtype=dtype))
    return sym.astype(dtype)


def _tol(dtype):
    return 1e-9 if np.dtype(dtype) == np.float64 else 1e-3


def _assert_close(got, stock, dtype):
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    rtol = _tol(dtype)
    max_abs = np.max(np.abs(stock), axis=-1)
    scale = np.maximum(1.0, max_abs)
    diff = np.abs(got - stock)
    assert np.all(diff <= rtol * scale[..., None])


def _assert_dispatched_close(a, *extra_args, kwargs=None):
    kwargs = kwargs or {}
    args = (a, *extra_args)
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.linalg.eigvalsh(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    _assert_close(got, stock, a.dtype)
    return got, stock


def _assert_refused_close(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.linalg.eigvalsh(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert np.array_equal(got, stock)
    return got


def _assert_refused_raise_parity(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)

    def _call(fn):
        try:
            return ("ok", fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - parity capture, not handling
            return ("raised", exc)

    got_kind, got_val = _call(lambda *a, **k: np.linalg.eigvalsh(*a, **k))
    stock_kind, stock_val = _call(_stock)
    assert got_kind == stock_kind, (got_kind, got_val, stock_kind, stock_val)
    if got_kind == "raised":
        assert type(got_val) is type(stock_val)
    else:
        assert np.array_equal(np.isnan(got_val), np.isnan(stock_val))
        finite = ~np.isnan(got_val)
        assert np.array_equal(got_val[finite], stock_val[finite])


# ---------------------------------------------------------------------------
# 1. dispatch + numeric equality
# ---------------------------------------------------------------------------

def test_dispatch_float64_batch_500():
    a = _symmetric_batch((500,), 1, dtype=np.float64)
    got, stock = _assert_dispatched_close(a)
    assert got.dtype == np.float64
    assert got.shape == (500, 2)


def test_dispatch_float32_batch_500():
    a = _symmetric_batch((500,), 2, dtype=np.float32)
    got, stock = _assert_dispatched_close(a)
    assert got.dtype == np.float32
    assert got.shape == (500, 2)


def test_dispatch_batch_exactly_min_dispatches():
    a = _symmetric_batch((BATCH_MIN,), 3, dtype=np.float64)
    _assert_dispatched_close(a)


def test_refusal_batch_min_minus_one():
    a = _symmetric_batch((BATCH_MIN - 1,), 4, dtype=np.float64)
    _assert_refused_close((a,), {})


def test_dispatch_4d_leading_shape():
    a = _symmetric_batch((10, 50), 5, dtype=np.float64)
    assert a.shape == (10, 50, 2, 2)
    got, stock = _assert_dispatched_close(a)
    assert got.shape == (10, 50, 2)


def test_dispatch_explicit_uplo_l_kwarg():
    a = _symmetric_batch((500,), 6, dtype=np.float64)
    _assert_dispatched_close(a, kwargs={"UPLO": "L"})


def test_dispatch_explicit_uplo_l_positional():
    a = _symmetric_batch((500,), 7, dtype=np.float64)
    _assert_dispatched_close(a, "L")


def test_dispatch_exactly_diagonal_matrices():
    rng = np.random.default_rng(8)
    diag = rng.uniform(0.1, 5.0, size=(500, 2))
    a = np.zeros((500, 2, 2), dtype=np.float64)
    a[..., 0, 0] = diag[..., 0]
    a[..., 1, 1] = diag[..., 1]
    assert np.all(a[..., 0, 1] == 0.0)
    assert np.all(a[..., 1, 0] == 0.0)
    _assert_dispatched_close(a)


def test_dispatch_repeated_eigenvalue_matrices():
    rng = np.random.default_rng(9)
    scales = rng.uniform(0.1, 5.0, size=(500, 1, 1))
    a = scales * np.eye(2, dtype=np.float64)
    _assert_dispatched_close(a)


def test_dispatch_ascending_order():
    a = _symmetric_batch((500,), 10, dtype=np.float64)
    got, _ = _assert_dispatched_close(a)
    assert np.all(got[..., 0] <= got[..., 1])


def test_dispatch_asymmetric_input_lower_triangle_only():
    # Both stock (UPLO='L') and the fast path read only the lower triangle,
    # so garbage in the upper triangle must not change dispatch or result.
    a = _symmetric_batch((500,), 11, dtype=np.float64)
    rng = np.random.default_rng(111)
    a[..., 0, 1] = rng.uniform(-1000.0, 1000.0, size=500)
    assert not np.array_equal(a[..., 0, 1], a[..., 1, 0])
    _assert_dispatched_close(a)


# ---------------------------------------------------------------------------
# 2. refusals
# ---------------------------------------------------------------------------

def test_refusal_uplo_u():
    a = _symmetric_batch((500,), 12, dtype=np.float64)
    _assert_refused_close((a,), {"UPLO": "U"})


def test_refusal_single_2d_matrix():
    a = _symmetric_batch((), 13, dtype=np.float64)
    assert a.shape == (2, 2)
    _assert_refused_close((a,), {})


def test_batch_3x3_shapes_not_claimed_by_2x2_path():
    # Since batch 7, (..., 3, 3) stacks dispatch to the sibling
    # eigvalsh_3x3_trig path (its own differential suite covers it); this
    # suite only pins that the 2x2 path itself never claims them.
    rng = np.random.default_rng(14)
    a = rng.uniform(-1.0, 1.0, size=(500, 3, 3)).astype(np.float64)
    a = a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(3)
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision != PATH, (decision, reason)


def test_refusal_complex128_hermitian_batch():
    rng = np.random.default_rng(15)
    re = rng.uniform(-1.0, 1.0, size=(500, 2, 2))
    im = rng.uniform(-1.0, 1.0, size=(500, 2, 2))
    a = (re + 1j * im).astype(np.complex128)
    a = a @ np.conj(np.swapaxes(a, -1, -2)) + 0.1 * np.eye(2)
    a = a.astype(np.complex128)
    _assert_refused_close((a,), {})


def test_refusal_int64_batch():
    rng = np.random.default_rng(16)
    a = rng.integers(-10, 10, size=(500, 2, 2)).astype(np.int64)
    a = a + np.swapaxes(a, -1, -2)  # symmetric int64
    _assert_refused_close((a,), {})


def test_refusal_nan_in_batch():
    a = _symmetric_batch((500,), 17, dtype=np.float64)
    a[42, 0, 0] = np.nan
    _assert_refused_raise_parity((a,), {})


def test_refusal_inf_in_batch():
    a = _symmetric_batch((500,), 18, dtype=np.float64)
    a[42, 1, 0] = np.inf
    _assert_refused_raise_parity((a,), {})


def test_refusal_python_nested_list_input():
    rng = np.random.default_rng(19)
    a_arr = _symmetric_batch((500,), 19, dtype=np.float64)
    a_list = a_arr.tolist()
    _assert_refused_close((a_list,), {})


def test_refusal_batch_below_floor():
    a = _symmetric_batch((10,), 20, dtype=np.float64)
    assert a.shape[0] < BATCH_MIN
    _assert_refused_close((a,), {})


# ---------------------------------------------------------------------------
# 3. kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_restores_stock_routing():
    a = _symmetric_batch((500,), 21, dtype=np.float64)
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (a,), {})
        assert decision == "stock", (decision, reason)
        got = np.linalg.eigvalsh(a)
        stock = _stock(a)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)
