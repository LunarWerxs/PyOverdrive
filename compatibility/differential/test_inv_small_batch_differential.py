"""Differential tests: inv_small_batch fast path vs stock numpy.linalg.inv.

Contract (src/pyoverdrive/fastpaths/inv_small_batch.py): applies only to
inv(a) where a is a plain ndarray, shape (..., d, d) with d in {2, 3},
ndim >= 3, dtype/batch in the measured table (_FLOORS: float64 batch >= 1000
for 2x2, >= 300 for 3x3; float32: 3x3 only, batch >= 10_000), every element
finite, and every |det| above the scale-relative floor (DET_RTOL * scale**d,
scale = that matrix's max |entry|, clamped away from zero). Any kwargs
refuse. Everything else - single matrices, other dtypes/sizes/shapes,
near-singular batches - stays on stock. Comparison mode is numeric: per-matrix
scaled tolerance against the maximum absolute entry of the stock inverse.
"""

import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.inv_small_batch import _scale, DET_RTOL, _FLOORS

OP = "numpy.linalg.inv"
PATH = "inv_small_batch"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _well_conditioned_batch(batch_shape, seed, d, dtype=np.float64):
    """Build a batch of well-conditioned dxd matrices: a @ swapaxes(a) + 0.1*eye(d)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(*batch_shape, d, d)).astype(dtype)
    wc = a @ np.swapaxes(a, -1, -2) + (0.1 * np.eye(d, dtype=dtype))
    return wc.astype(dtype)


def _tol(dtype):
    return 1e-9 if np.dtype(dtype) == np.float64 else 1e-3


def _assert_close(got, stock, dtype):
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    rtol = _tol(dtype)
    max_abs = np.max(np.abs(stock), axis=(-2, -1))
    scale = np.maximum(1.0, max_abs)
    diff = np.abs(got - stock)
    assert np.all(diff <= rtol * scale[..., None, None])


def _assert_dispatched_close(a, *extra_args, kwargs=None):
    kwargs = kwargs or {}
    args = (a, *extra_args)
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.linalg.inv(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    _assert_close(got, stock, a.dtype)
    return got, stock


def _assert_refused_close(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.linalg.inv(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert np.array_equal(got, stock)
    return got


def _assert_guarded_at_run(args, kwargs):
    """Parity for a refusal the guard now makes MID-RUN, not in the predicate.

    The predicate used to prove finiteness and non-singularity, which meant
    computing the determinant there and again in the run, plus a full
    isfinite scan of the stack. Measured at batch 4096 that guard cost
    128.5 us against 25.8 us for the entire 2x2 inverse it was protecting -
    the dispatched call ran at 5.0x where the work alone is far more. The
    guard now runs once, inside the run, against the determinant the run
    already computed, and hands the whole call to stock on refusal.

    So the DECISION is the path and the BEHAVIOUR is stock's. The second
    half is the part that matters and it is asserted exactly as strictly as
    it was when the predicate refused.
    """
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (args, kwargs, decision, reason)
    _assert_parity_with_stock(args, kwargs)


def _assert_refused_raise_parity(args, kwargs):
    """Refusal parity that tolerates either side raising (or not).

    Used for inputs the PREDICATE still refuses (shape, dtype, floor), where
    stock's behavior (raise vs. NaN/garbage output) is itself not part of
    the contract we assert on: only that the fast path stays out of the way
    and whatever stock does, the wrapped call does too.
    """
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    _assert_parity_with_stock(args, kwargs)


def _assert_parity_with_stock(args, kwargs):
    def _call(fn):
        try:
            return ("ok", fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - parity capture, not handling
            return ("raised", exc)

    got_kind, got_val = _call(lambda *a, **k: np.linalg.inv(*a, **k))
    stock_kind, stock_val = _call(_stock)
    assert got_kind == stock_kind, (got_kind, got_val, stock_kind, stock_val)
    if got_kind == "raised":
        assert type(got_val) is type(stock_val)
    else:
        assert np.array_equal(np.isnan(got_val), np.isnan(stock_val))
        finite_mask = np.isfinite(got_val) & np.isfinite(stock_val)
        assert np.array_equal(got_val[finite_mask], stock_val[finite_mask])


# ---------------------------------------------------------------------------
# 1. dispatch + numeric equality
# ---------------------------------------------------------------------------

def test_dispatch_3x3_f64_at_floor():
    floor = _FLOORS[(3, np.dtype(np.float64))]
    a = _well_conditioned_batch((floor,), 1, d=3, dtype=np.float64)
    got, stock = _assert_dispatched_close(a)
    assert got.dtype == np.float64
    assert got.shape == (floor, 3, 3)


def test_refusal_3x3_f64_floor_minus_one():
    floor = _FLOORS[(3, np.dtype(np.float64))]
    a = _well_conditioned_batch((floor - 1,), 2, d=3, dtype=np.float64)
    _assert_refused_close((a,), {})


def test_dispatch_2x2_f64_at_floor():
    floor = _FLOORS[(2, np.dtype(np.float64))]
    a = _well_conditioned_batch((floor,), 3, d=2, dtype=np.float64)
    got, stock = _assert_dispatched_close(a)
    assert got.dtype == np.float64
    assert got.shape == (floor, 2, 2)


def test_dispatch_3x3_f32_at_floor():
    floor = _FLOORS[(3, np.dtype(np.float32))]
    a = _well_conditioned_batch((floor,), 4, d=3, dtype=np.float32)
    got, stock = _assert_dispatched_close(a)
    assert got.dtype == np.float32
    assert got.shape == (floor, 3, 3)


def test_dispatch_4d_leading_shape():
    a = _well_conditioned_batch((10, 40), 5, d=3, dtype=np.float64)
    assert a.shape == (10, 40, 3, 3)
    got, stock = _assert_dispatched_close(a)
    assert got.shape == (10, 40, 3, 3)


def test_reconstruction_sanity():
    floor = _FLOORS[(3, np.dtype(np.float64))]
    a = _well_conditioned_batch((floor,), 6, d=3, dtype=np.float64)
    got, _ = _assert_dispatched_close(a)
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), a.shape)
    recon = a @ got
    assert np.all(np.abs(recon - eye) <= 1e-6)


# ---------------------------------------------------------------------------
# 2. refusals
# ---------------------------------------------------------------------------

def test_refusal_all_zero_matrix_in_batch():
    # Docstring: an all-zero matrix has det == 0 and scale == 0 (clamped to
    # 1e-100), so the predicate must refuse rather than divide by zero.
    # Stock raises LinAlgError for the singular matrix - exception-type parity.
    floor = _FLOORS[(3, np.dtype(np.float64))]
    a = _well_conditioned_batch((floor,), 7, d=3, dtype=np.float64)
    a[42] = 0.0
    _assert_guarded_at_run((a,), {})


def test_refusal_nan_element():
    floor = _FLOORS[(3, np.dtype(np.float64))]
    a = _well_conditioned_batch((floor,), 8, d=3, dtype=np.float64)
    a[42, 0, 0] = np.nan
    _assert_guarded_at_run((a,), {})


def test_refusal_inf_element():
    floor = _FLOORS[(3, np.dtype(np.float64))]
    a = _well_conditioned_batch((floor,), 9, d=3, dtype=np.float64)
    a[42, 1, 2] = np.inf
    _assert_guarded_at_run((a,), {})


def test_refusal_near_singular_matrix():
    # Third row = first row + 1e-12: det collapses far below DET_RTOL *
    # scale**3, so the predicate must refuse. Verify the ratio really is
    # under the contract's own threshold (imported, not hardcoded), then
    # confirm the refusal is a genuine parity - stock still answers or
    # raises, and the wrapped call matches it either way.
    floor = _FLOORS[(3, np.dtype(np.float64))]
    a = _well_conditioned_batch((floor,), 10, d=3, dtype=np.float64)
    a[42, 2, :] = a[42, 0, :] + 1e-12

    det = np.linalg.det(a[42])
    scale = np.max(np.abs(a[42]))
    assert abs(det) < DET_RTOL * max(scale, 1e-100) ** 3

    _assert_guarded_at_run((a,), {})


def test_refusal_4x4_batch():
    a = _well_conditioned_batch((50,), 11, d=4, dtype=np.float64)
    _assert_refused_close((a,), {})


def test_refusal_single_2d_matrix():
    a = _well_conditioned_batch((), 12, d=3, dtype=np.float64)
    assert a.shape == (3, 3)
    _assert_refused_close((a,), {})


def test_refusal_int64_batch():
    # Diagonally dominant so stock's cast-to-float64 inverse is well defined;
    # dtype alone (int64 is not in _FLOORS) forces refusal regardless of size.
    rng = np.random.default_rng(13)
    d = 3
    batch = 50
    a = rng.integers(-2, 3, size=(batch, d, d)).astype(np.int64)
    idx = np.arange(d)
    a[:, idx, idx] = 10 + rng.integers(0, 3, size=(batch, d))
    _assert_refused_close((a,), {})


def test_refusal_python_nested_list_input():
    floor = _FLOORS[(3, np.dtype(np.float64))]
    a_arr = _well_conditioned_batch((floor,), 14, d=3, dtype=np.float64)
    a_list = a_arr.tolist()
    _assert_refused_close((a_list,), {})


def test_refusal_unknown_kwarg():
    # No kwargs are valid for this path; stock raises TypeError for an
    # unrecognized keyword - exception-type parity.
    floor = _FLOORS[(3, np.dtype(np.float64))]
    a = _well_conditioned_batch((floor,), 15, d=3, dtype=np.float64)
    _assert_refused_raise_parity((a,), {"foo": 1})


# ---------------------------------------------------------------------------
# 3. kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_restores_stock_routing():
    floor = _FLOORS[(3, np.dtype(np.float64))]
    a = _well_conditioned_batch((floor,), 16, d=3, dtype=np.float64)
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (a,), {})
        assert decision == "stock", (decision, reason)
        got = np.linalg.inv(a)
        stock = _stock(a)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)


def test_refused_call_emits_no_warning_stock_would_not():
    """A call that ends up on stock must be indistinguishable from never
    having taken this path - warnings included.

    Fusing the guard into the run means non-finite input now reaches the
    closed form BEFORE the guard can refuse it, so inf-inf and inf*0 raise
    "invalid value encountered" on exactly the inputs about to be handed
    back to stock, which would not have warned. That is a real difference a
    caller can see (and -W error turns it into a crash), so the guard path
    suppresses it and this test is what keeps it suppressed.
    """
    floor = _FLOORS[(3, np.dtype(np.float64))]
    for seed, bad in ((21, np.inf), (22, -np.inf), (23, np.nan)):
        a = _well_conditioned_batch((floor,), seed, d=3, dtype=np.float64)
        a[7, 1, 2] = bad
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                np.linalg.inv(a)
            except np.linalg.LinAlgError:
                pass  # stock's own refusal is fine; a WARNING is not


def test_folded_scale_matches_the_reduction_it_replaced():
    """_scale folds max|entry| over the entry views instead of reducing over
    the last two axes, because the reduction cost 4.6-11x more. Same values,
    or the conditioning guard means something different than it did."""
    rng = np.random.default_rng(99)
    for d in (2, 3, 4):
        a = rng.standard_normal((257, d, d)) * rng.choice([1e-8, 1.0, 1e8], (257, 1, 1))
        assert np.array_equal(_scale(a), np.abs(a).max(axis=(-2, -1)))
