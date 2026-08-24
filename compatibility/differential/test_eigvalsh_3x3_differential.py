"""Differential tests: eigvalsh_3x3_trig fast path vs stock numpy.linalg.eigvalsh.

Contract (src/pyoverdrive/fastpaths/eigvalsh_3x3.py): applies only to
eigvalsh(a) / eigvalsh(a, UPLO='L') where a is a plain float64/float32
ndarray shaped (..., 3, 3) with ndim >= 3, at least BATCH_MIN matrices, and
every element finite. UPLO='U', 2-D single matrices, (...,2,2) shapes,
complex Hermitian input, other dtypes, and non-finite values all stay on
stock. Both stock (UPLO='L') and the fast path read only the lower
triangle, so an asymmetric upper triangle must not change the result.
Comparison mode is numeric: per-matrix scaled tolerance against the
maximum absolute eigenvalue in that matrix.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.eigvalsh_3x3 import BATCH_MIN

OP = "numpy.linalg.eigvalsh"
PATH = "eigvalsh_3x3_trig"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _symmetric_batch(batch_shape, seed, dtype=np.float64):
    """Build a batch of symmetric 3x3 matrices: a @ swapaxes(a) + 0.1*eye(3)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(*batch_shape, 3, 3)).astype(dtype)
    sym = a @ np.swapaxes(a, -1, -2) + (0.1 * np.eye(3, dtype=dtype))
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
    assert got.shape == (500, 3)


def test_dispatch_float32_batch_500():
    a = _symmetric_batch((500,), 2, dtype=np.float32)
    got, stock = _assert_dispatched_close(a)
    assert got.dtype == np.float32
    assert got.shape == (500, 3)


def test_dispatch_batch_exactly_min_dispatches():
    a = _symmetric_batch((BATCH_MIN,), 3, dtype=np.float64)
    _assert_dispatched_close(a)


def test_refusal_batch_min_minus_one():
    a = _symmetric_batch((BATCH_MIN - 1,), 4, dtype=np.float64)
    _assert_refused_close((a,), {})


def test_dispatch_4d_leading_shape():
    a = _symmetric_batch((10, 50), 5, dtype=np.float64)
    assert a.shape == (10, 50, 3, 3)
    got, stock = _assert_dispatched_close(a)
    assert got.shape == (10, 50, 3)


def test_dispatch_explicit_uplo_l_kwarg():
    a = _symmetric_batch((500,), 6, dtype=np.float64)
    _assert_dispatched_close(a, kwargs={"UPLO": "L"})


def test_dispatch_explicit_uplo_l_positional():
    a = _symmetric_batch((500,), 7, dtype=np.float64)
    _assert_dispatched_close(a, "L")


def test_dispatch_exact_multiple_of_identity():
    rng = np.random.default_rng(8)
    scales = rng.uniform(0.1, 5.0, size=(500, 1, 1))
    a = scales * np.eye(3, dtype=np.float64)
    got, stock = _assert_dispatched_close(a)
    assert np.allclose(got[..., 0], got[..., 1])
    assert np.allclose(got[..., 1], got[..., 2])


def test_dispatch_clustered_eigenvalues():
    # Two nearly-equal eigenvalues (near-degenerate p ~ small but nonzero):
    # base diagonal matrix with a tiny symmetric perturbation.
    rng = np.random.default_rng(9)
    base = np.zeros((500, 3, 3), dtype=np.float64)
    base[..., 0, 0] = 1.0
    base[..., 1, 1] = 1.0 + 1e-6
    base[..., 2, 2] = 7.0
    perturb = rng.uniform(-1e-7, 1e-7, size=(500, 3, 3))
    perturb = perturb + np.swapaxes(perturb, -1, -2)
    a = base + perturb
    got, _ = _assert_dispatched_close(a)
    assert np.all(got[..., :-1] <= got[..., 1:])


def test_dispatch_ascending_order():
    a = _symmetric_batch((500,), 10, dtype=np.float64)
    got, _ = _assert_dispatched_close(a)
    assert np.all(got[..., :-1] <= got[..., 1:])


def test_dispatch_asymmetric_input_lower_triangle_only():
    # Both stock (UPLO='L') and the fast path read only the lower triangle,
    # so garbage in the upper triangle must not change dispatch or result.
    a = _symmetric_batch((500,), 11, dtype=np.float64)
    rng = np.random.default_rng(111)
    a[..., 0, 1] = rng.uniform(-1000.0, 1000.0, size=500)
    a[..., 0, 2] = rng.uniform(-1000.0, 1000.0, size=500)
    a[..., 1, 2] = rng.uniform(-1000.0, 1000.0, size=500)
    assert not np.array_equal(a[..., 0, 1], a[..., 1, 0])
    _assert_dispatched_close(a)


# ---------------------------------------------------------------------------
# 2. refusals
# ---------------------------------------------------------------------------

def test_refusal_uplo_u():
    a = _symmetric_batch((500,), 12, dtype=np.float64)
    _assert_refused_close((a,), {"UPLO": "U"})


def test_refusal_duplicate_uplo_positional_and_kwarg():
    a = _symmetric_batch((500,), 121, dtype=np.float64)
    _assert_refused_raise_parity((a, "L"), {"UPLO": "L"})


def test_refusal_single_2d_matrix():
    a = _symmetric_batch((), 13, dtype=np.float64)
    assert a.shape == (3, 3)
    _assert_refused_close((a,), {})


def test_refusal_batch_2x2_shapes():
    # (..., 2, 2) belongs to the eigvalsh_2x2 path, not this one: the
    # eigvalsh_3x3_trig path must decline it. (It may still dispatch to
    # eigvalsh_2x2_closed, which shares this op and is also enabled by
    # this module's fixture; that path's own correctness is covered by
    # test_eigvalsh_2x2_differential.py.)
    rng = np.random.default_rng(14)
    a = rng.uniform(-1.0, 1.0, size=(500, 2, 2)).astype(np.float64)
    a = a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(2)
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision != PATH, (decision, reason)


def test_refusal_complex128_hermitian_batch():
    rng = np.random.default_rng(15)
    re = rng.uniform(-1.0, 1.0, size=(500, 3, 3))
    im = rng.uniform(-1.0, 1.0, size=(500, 3, 3))
    a = (re + 1j * im).astype(np.complex128)
    a = a @ np.conj(np.swapaxes(a, -1, -2)) + 0.1 * np.eye(3)
    a = a.astype(np.complex128)
    _assert_refused_close((a,), {})


def test_refusal_int64_batch():
    rng = np.random.default_rng(16)
    a = rng.integers(-10, 10, size=(500, 3, 3)).astype(np.int64)
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


# ---------------------------------------------------------------------------
# 4. near-degenerate split-and-recombine (batch 9)
# ---------------------------------------------------------------------------

_DEGEN = np.diag([1.0, 1.0, 5.0])  # exactly repeated pair: 1 - r^2 == 0


def _mixed_stack(n, frac, seed, dtype=np.float64):
    a = _symmetric_batch((n,), seed, dtype=dtype)
    rng = np.random.default_rng(seed + 1)
    nbad = int(round(frac * n))
    idx = rng.choice(n, nbad, replace=False)
    a[idx] = _DEGEN.astype(dtype)
    return a, idx


def test_split_degenerate_cells_served_at_stock_grade():
    # a few coalesced pairs no longer cost the stack its speedup, and the
    # degenerate cells get stock's eps-grade values, not the trig form's
    # sqrt(eps)-degraded ones
    a, idx = _mixed_stack(2000, 0.01, seed=31)
    got, stock = _assert_dispatched_close(a)
    scale = np.maximum(1.0, np.max(np.abs(stock[idx]), axis=-1))
    assert np.all(np.abs(got[idx] - stock[idx]) <= 1e-12 * scale[..., None])


def test_split_high_fraction_whole_stack_from_stock():
    # past DEGEN_FRAC_MAX the whole stack goes to stock in one batched
    # call, so the result is stock's exactly
    from pyoverdrive.fastpaths.eigvalsh_3x3 import DEGEN_FRAC_MAX

    frac = min(1.0, DEGEN_FRAC_MAX * 2)
    a, _ = _mixed_stack(2000, frac, seed=32)
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision == PATH, (decision, reason)
    got = np.linalg.eigvalsh(a)
    assert np.array_equal(got, _stock(a))


def test_split_float32_degenerate_cells_exact_stock_values():
    # the split serves float32 degenerate cells from stock's own float32
    # route; the values survive the f32->f64->f32 roundtrip exactly
    a, idx = _mixed_stack(2000, 0.01, seed=33, dtype=np.float32)
    got, stock = _assert_dispatched_close(a)
    assert np.array_equal(got[idx], stock[idx])


def test_split_multidim_batch():
    a, _ = _mixed_stack(3000, 0.01, seed=34)
    a = a.reshape(20, 150, 3, 3).copy()
    _assert_dispatched_close(a)
