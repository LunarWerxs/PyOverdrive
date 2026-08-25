"""Differential tests: svd_small_batch fast paths vs stock NumPy.

Contract (src/pyoverdrive/fastpaths/svd_small_batch.py): three paths share
one closed-form gram-matrix core, registered from one module -

- norm2_small_batch  (numpy.linalg.norm(a, ord=2, axis=(-2,-1) or
  (ndim-2,ndim-1))): only sigma_max is needed, accurate to eps at ANY
  condition number, so it carries no conditioning guard at all.
- svdvals_small_batch (numpy.linalg.svd(a, compute_uv=False)): all
  singular values, judged by ABSOLUTE error against ||A|| (sigma_max),
  gated by a per-dimension SVDVALS_SIGMA_RATIO_MIN band.
- pinv_small_batch (numpy.linalg.pinv(a)): the pseudo-inverse via the
  adjugate formula, error grows linearly with condition number, gated by
  PINV_SIGMA_RATIO_MIN.

Shared applicability: plain float64 (..., d, d) ndarrays, d in {2, 3},
ndim >= 3, batch >= BATCH_MIN, all finite, default keyword arguments only.
Band-trippers are gathered, served by stock, and scattered back; past
BAD_FRAC_MAX of a stack, the whole call goes to stock in one batched pass.
Kill switches: pinv_small_batch, norm2_small_batch, svdvals_small_batch.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.svd_small_batch import (
    BAD_FRAC_MAX,
    BATCH_MIN,
    DEGENERACY_MIN,
    PINV_SIGMA_RATIO_MIN,
    SVDVALS_SIGMA_RATIO_MIN,
)

PINV_OP = "numpy.linalg.pinv"
NORM_OP = "numpy.linalg.norm"
SVD_OP = "numpy.linalg.svd"
OPS = [PINV_OP, NORM_OP, SVD_OP]

PINV_PATH = "pinv_small_batch"
NORM_PATH = "norm2_small_batch"
SVDVALS_PATH = "svdvals_small_batch"

# minimal kwargs that make each op otherwise-applicable, for generic
# shape/dtype refusal tests shared across all three paths
_OP_KWARGS = {
    PINV_OP: {},
    SVD_OP: {"compute_uv": False},
    NORM_OP: {"ord": 2, "axis": (-2, -1)},
}
_OP_PATH = {PINV_OP: PINV_PATH, SVD_OP: SVDVALS_PATH, NORM_OP: NORM_PATH}
_OP_FN = {PINV_OP: np.linalg.pinv, SVD_OP: np.linalg.svd, NORM_OP: np.linalg.norm}

# conditioning thresholds derived from the module's own constants
_PINV_COND_MAX = 1.0 / PINV_SIGMA_RATIO_MIN
_SVDVALS_COND_MAX = {d: 1.0 / SVDVALS_SIGMA_RATIO_MIN[d] for d in (2, 3)}
_BAD_COND_FACTOR = 100.0  # deliberately, comfortably past the band floor
_EDGE_FACTOR = 0.9  # deliberately, comfortably inside the band floor


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(OPS)
    yield
    pyoverdrive.disable()


def _dispatch_fn(op):
    return _OP_FN[op]


def _stock(op, *args, **kwargs):
    return GEARBOX.stock_fn(op)(*args, **kwargs)


def _same(got, stock):
    """NaN-aware structural equality for ndarray / SVDResult-like tuple."""
    if isinstance(stock, tuple):
        return type(got) is type(stock) and len(got) == len(stock) and all(
            np.array_equal(g, s, equal_nan=True) for g, s in zip(got, stock)
        )
    return type(got) is type(stock) and np.array_equal(got, stock, equal_nan=True)


def _assert_refused(op, args, kwargs):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == "stock", (op, args, kwargs, decision, reason)
    got = _dispatch_fn(op)(*args, **kwargs)
    stock = _stock(op, *args, **kwargs)
    assert _same(got, stock)
    return got


def _assert_refused_raise_parity(op, args, kwargs):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == "stock", (op, args, kwargs, decision, reason)

    def _call(fn):
        try:
            return ("ok", fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - parity capture, not handling
            return ("raised", exc)

    got_kind, got_val = _call(_dispatch_fn(op))
    stock_kind, stock_val = _call(lambda *a, **k: _stock(op, *a, **k))
    assert got_kind == stock_kind, (got_kind, got_val, stock_kind, stock_val)
    if got_kind == "raised":
        assert type(got_val) is type(stock_val)
    else:
        assert _same(got_val, stock_val)


def _assert_no_runtime_warning(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not rw, [str(w.message) for w in rw]
    return result


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------


def _random_batch(batch_shape, d, seed, dtype=np.float64):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(*batch_shape, d, d)).astype(dtype)


def _cond_batch(batch_shape, d, seed, cond, dtype=np.float64):
    """Exact-spectrum construction: random orthogonal U, V (via QR of a
    random matrix) and a fixed spectrum shape - sigma_max == 1.0 (and,
    for d=3, a middle value of 0.5, distinct from both neighbours) with
    only the smallest singular value driven down to 1/cond - so
    cond(A) == cond to rounding, with sigma_max staying O(1).

    Two other shapes were tried and rejected empirically:
    - a full geomspace spread (sigma_max == cond) pushes the gram's
      largest eigenvalue to cond**2, and the closed-form trig solve then
      loses the smallest eigenvalue to cancellation noise *larger* than
      the "bad" threshold itself - the classifier's own detector became
      unreliable at exactly the condition numbers it exists to catch
      (measured frac flagged bad ranging ~0.2-0.75 at d=3 across a wide
      span of factors past the floor, instead of a clean 0/1 split).
    - a two-tier spectrum (sigma_max == mid == 1.0 exactly) classifies
      cleanly, but the exact top-pair degeneracy trips the same
      clustered-eigenvalue error amplification documented for
      eigvalsh_3x3, holding svdvals error near ~4e-9 regardless of how
      small cond was pushed - i.e. it was probing degeneracy, not band
      conditioning.

    This shape keeps hi/mid/lo distinct and O(1) and was verified
    empirically to classify cleanly (0.0 below the floor, 1.0
    comfortably past it) AND hold accuracy to ~1e-10 or tighter at the
    band edge, for both pinv and svdvals, d in {2, 3}.
    """
    rng = np.random.default_rng(seed)
    shape = (*batch_shape, d, d)
    u, _ = np.linalg.qr(rng.standard_normal(shape))
    v, _ = np.linalg.qr(rng.standard_normal(shape))
    sv = np.array([1.0, 0.5, 1.0 / cond]) if d == 3 else np.array([1.0, 1.0 / cond])
    s = np.zeros(shape)
    idx = np.arange(d)
    s[..., idx, idx] = sv
    a = u @ s @ np.swapaxes(v, -1, -2)
    return a.astype(dtype)


def _mixed_cond_batch(n_good, n_bad, d, cond_bad, seed, dtype=np.float64):
    good = _random_batch((n_good,), d, seed=seed, dtype=dtype)
    bad = _cond_batch((n_bad,), d, seed=seed + 1, cond=cond_bad, dtype=dtype)
    a = np.concatenate([good, bad], axis=0)
    rng = np.random.default_rng(seed + 2)
    perm = rng.permutation(a.shape[0])
    a = a[perm]
    bad_idx = np.flatnonzero(perm >= n_good)
    return a, bad_idx


def _rank_deficient_batch(batch_shape, d, seed, dtype=np.float64):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=(*batch_shape, d, 1)).astype(dtype)
    v = rng.uniform(-1.0, 1.0, size=(*batch_shape, 1, d)).astype(dtype)
    return (u @ v).astype(dtype)


def _multidim_batch(d, seed, dtype=np.float64):
    n = math.ceil(BATCH_MIN / 10)
    a = _random_batch((10, n), d, seed=seed, dtype=dtype)
    assert a.shape == (10, n, d, d)
    assert 10 * n >= BATCH_MIN
    return a


# ---------------------------------------------------------------------------
# per-path agreement helpers, each enforcing the RIGHT accuracy standard
# ---------------------------------------------------------------------------


def _assert_pinv_agrees(a):
    decision, reason = GEARBOX.decide(PINV_OP, (a,), {})
    assert decision == PINV_PATH, (decision, reason)
    got = np.linalg.pinv(a)
    stock = _stock(PINV_OP, a)
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert type(got) is type(stock)
    scale = max(1.0, float(np.max(np.abs(stock))))
    assert np.max(np.abs(got - stock)) <= 1e-9 * scale
    return got, stock


def _assert_svdvals_agrees(a):
    kwargs = {"compute_uv": False}
    decision, reason = GEARBOX.decide(SVD_OP, (a,), kwargs)
    assert decision == SVDVALS_PATH, (decision, reason)
    got = np.linalg.svd(a, **kwargs)
    stock = _stock(SVD_OP, a, **kwargs)
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert type(got) is type(stock)
    sigma_max = stock[..., 0]
    err = np.abs(got - stock)
    assert np.all(err <= 1e-9 * sigma_max[..., None])
    return got, stock


def _assert_norm2_agrees(a, axis):
    kwargs = {"ord": 2, "axis": axis}
    decision, reason = GEARBOX.decide(NORM_OP, (a,), kwargs)
    assert decision == NORM_PATH, (decision, reason)
    got = np.linalg.norm(a, **kwargs)
    stock = _stock(NORM_OP, a, **kwargs)
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert type(got) is type(stock)
    rel = np.abs(got - stock) / np.maximum(1.0, np.abs(stock))
    assert np.all(rel <= 1e-9)
    return got, stock


# ---------------------------------------------------------------------------
# 1. agreement on well-conditioned random stacks, d=2/3, BATCH_MIN and larger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [BATCH_MIN, BATCH_MIN * 5])
def test_pinv_agreement_well_conditioned(d, n):
    a = _random_batch((n,), d, seed=100 + d * 10 + (n == BATCH_MIN * 5))
    got, _ = _assert_pinv_agrees(a)
    assert got.shape == (n, d, d)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [BATCH_MIN, BATCH_MIN * 5])
def test_svdvals_agreement_well_conditioned(d, n):
    a = _random_batch((n,), d, seed=200 + d * 10 + (n == BATCH_MIN * 5))
    got, _ = _assert_svdvals_agrees(a)
    assert got.shape == (n, d)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [BATCH_MIN, BATCH_MIN * 5])
def test_norm2_agreement_well_conditioned(d, n):
    a = _random_batch((n,), d, seed=300 + d * 10 + (n == BATCH_MIN * 5))
    got, _ = _assert_norm2_agrees(a, axis=(-2, -1))
    assert got.shape == (n,)


# ---------------------------------------------------------------------------
# 2. multi-dimensional batch shapes, plus both norm2 axis spellings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [2, 3])
def test_multidim_batch_pinv(d):
    a = _multidim_batch(d, seed=1000 + d)
    _assert_pinv_agrees(a)


@pytest.mark.parametrize("d", [2, 3])
def test_multidim_batch_svdvals(d):
    a = _multidim_batch(d, seed=1010 + d)
    _assert_svdvals_agrees(a)


@pytest.mark.parametrize("d", [2, 3])
def test_multidim_batch_norm2_negative_axis(d):
    a = _multidim_batch(d, seed=1020 + d)
    _assert_norm2_agrees(a, axis=(-2, -1))


@pytest.mark.parametrize("d", [2, 3])
def test_multidim_batch_norm2_positive_axis(d):
    a = _multidim_batch(d, seed=1030 + d)
    _assert_norm2_agrees(a, axis=(a.ndim - 2, a.ndim - 1))


# ---------------------------------------------------------------------------
# 3. the band + split behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [2, 3])
def test_band_split_few_ill_conditioned_scattered_pinv(d):
    cond_bad = _PINV_COND_MAX * _BAD_COND_FACTOR
    n_bad = 5
    n_good = BATCH_MIN * 4 - n_bad
    a, bad_idx = _mixed_cond_batch(n_good, n_bad, d, cond_bad, seed=600 + d)
    decision, reason = GEARBOX.decide(PINV_OP, (a,), {})
    assert decision == PINV_PATH, (decision, reason)
    got = np.linalg.pinv(a)
    stock = _stock(PINV_OP, a)
    assert np.array_equal(got[bad_idx], stock[bad_idx])
    scale = max(1.0, float(np.max(np.abs(stock))))
    assert np.max(np.abs(got - stock)) <= 1e-9 * scale


@pytest.mark.parametrize("d", [2, 3])
def test_band_split_few_ill_conditioned_scattered_svdvals(d):
    cond_bad = _SVDVALS_COND_MAX[d] * _BAD_COND_FACTOR
    n_bad = 5
    n_good = BATCH_MIN * 4 - n_bad
    a, bad_idx = _mixed_cond_batch(n_good, n_bad, d, cond_bad, seed=650 + d)
    kwargs = {"compute_uv": False}
    decision, reason = GEARBOX.decide(SVD_OP, (a,), kwargs)
    assert decision == SVDVALS_PATH, (decision, reason)
    got = np.linalg.svd(a, **kwargs)
    stock = _stock(SVD_OP, a, **kwargs)
    assert np.array_equal(got[bad_idx], stock[bad_idx])
    mask = np.ones(a.shape[0], dtype=bool)
    mask[bad_idx] = False
    sigma_max = stock[mask, 0]
    err = np.abs(got[mask] - stock[mask])
    assert np.all(err <= 1e-9 * sigma_max[:, None])


@pytest.mark.parametrize("d", [2, 3])
def test_band_split_mostly_ill_conditioned_falls_back_pinv(d):
    frac = min(1.0, BAD_FRAC_MAX * 2)
    n = BATCH_MIN * 4
    n_bad = int(n * frac)
    cond_bad = _PINV_COND_MAX * _BAD_COND_FACTOR
    a, _ = _mixed_cond_batch(n - n_bad, n_bad, d, cond_bad, seed=700 + d)
    decision, reason = GEARBOX.decide(PINV_OP, (a,), {})
    assert decision == PINV_PATH, (decision, reason)
    got = np.linalg.pinv(a)
    stock = _stock(PINV_OP, a)
    assert np.array_equal(got, stock)


@pytest.mark.parametrize("d", [2, 3])
def test_band_split_mostly_ill_conditioned_falls_back_svdvals(d):
    frac = min(1.0, BAD_FRAC_MAX * 2)
    n = BATCH_MIN * 4
    n_bad = int(n * frac)
    cond_bad = _SVDVALS_COND_MAX[d] * _BAD_COND_FACTOR
    a, _ = _mixed_cond_batch(n - n_bad, n_bad, d, cond_bad, seed=710 + d)
    kwargs = {"compute_uv": False}
    decision, reason = GEARBOX.decide(SVD_OP, (a,), kwargs)
    assert decision == SVDVALS_PATH, (decision, reason)
    got = np.linalg.svd(a, **kwargs)
    stock = _stock(SVD_OP, a, **kwargs)
    assert np.array_equal(got, stock)


@pytest.mark.parametrize("d", [2, 3])
def test_singular_matrices_scattered_pinv(d):
    n = BATCH_MIN * 4
    a = _random_batch((n,), d, seed=800 + d)
    rng = np.random.default_rng(801 + d)
    zero_idx = rng.choice(n, 3, replace=False)
    a[zero_idx] = 0.0
    remaining = np.setdiff1d(np.arange(n), zero_idx)
    rank1_idx = rng.choice(remaining, 3, replace=False)
    a[rank1_idx] = _rank_deficient_batch((3,), d, seed=802 + d)
    decision, reason = GEARBOX.decide(PINV_OP, (a,), {})
    assert decision == PINV_PATH, (decision, reason)
    got = np.linalg.pinv(a)
    stock = _stock(PINV_OP, a)
    bad_idx = np.concatenate([zero_idx, rank1_idx])
    assert np.array_equal(got[bad_idx], stock[bad_idx])


@pytest.mark.parametrize("d", [2, 3])
def test_singular_matrices_scattered_svdvals(d):
    n = BATCH_MIN * 4
    a = _random_batch((n,), d, seed=810 + d)
    rng = np.random.default_rng(811 + d)
    zero_idx = rng.choice(n, 3, replace=False)
    a[zero_idx] = 0.0
    remaining = np.setdiff1d(np.arange(n), zero_idx)
    rank1_idx = rng.choice(remaining, 3, replace=False)
    a[rank1_idx] = _rank_deficient_batch((3,), d, seed=812 + d)
    kwargs = {"compute_uv": False}
    decision, reason = GEARBOX.decide(SVD_OP, (a,), kwargs)
    assert decision == SVDVALS_PATH, (decision, reason)
    got = np.linalg.svd(a, **kwargs)
    stock = _stock(SVD_OP, a, **kwargs)
    bad_idx = np.concatenate([zero_idx, rank1_idx])
    assert np.array_equal(got[bad_idx], stock[bad_idx])


@pytest.mark.parametrize("d", [2, 3])
def test_singular_matrices_scattered_norm2(d):
    # norm2 has no conditioning guard at all: sigma_max is accurate to eps
    # at any condition number, including exactly-singular matrices.
    n = BATCH_MIN * 4
    a = _random_batch((n,), d, seed=820 + d)
    rng = np.random.default_rng(821 + d)
    zero_idx = rng.choice(n, 3, replace=False)
    a[zero_idx] = 0.0
    _assert_norm2_agrees(a, axis=(-2, -1))


@pytest.mark.parametrize("d", [2, 3])
def test_pinv_accuracy_at_band_edge(d):
    cond = _PINV_COND_MAX * _EDGE_FACTOR
    a = _cond_batch((BATCH_MIN * 2,), d, seed=900 + d, cond=cond)
    _assert_pinv_agrees(a)


@pytest.mark.parametrize("d", [2, 3])
def test_svdvals_accuracy_at_band_edge(d):
    cond = _SVDVALS_COND_MAX[d] * _EDGE_FACTOR
    a = _cond_batch((BATCH_MIN * 2,), d, seed=910 + d, cond=cond)
    _assert_svdvals_agrees(a)


# ---------------------------------------------------------------------------
# 4. refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", OPS)
def test_refusal_4x4_batch(op):
    a = _random_batch((BATCH_MIN,), 4, seed=1200)
    _assert_refused(op, (a,), _OP_KWARGS[op])


@pytest.mark.parametrize("op", OPS)
def test_refusal_rectangular_batch(op):
    rng = np.random.default_rng(1201)
    a = rng.uniform(-1.0, 1.0, size=(BATCH_MIN, 3, 2))
    _assert_refused(op, (a,), _OP_KWARGS[op])


@pytest.mark.parametrize("op", OPS)
def test_refusal_float32_input(op):
    a = _random_batch((BATCH_MIN,), 3, seed=1210, dtype=np.float32)
    _assert_refused(op, (a,), _OP_KWARGS[op])


@pytest.mark.parametrize("op", OPS)
def test_refusal_int_dtype(op):
    rng = np.random.default_rng(1211)
    a = rng.integers(-5, 5, size=(BATCH_MIN, 3, 3)).astype(np.int64)
    _assert_refused(op, (a,), _OP_KWARGS[op])


@pytest.mark.parametrize("op", OPS)
def test_refusal_single_2d_matrix(op):
    a = _random_batch((), 3, seed=1220)
    assert a.shape == (3, 3)
    _assert_refused(op, (a,), _OP_KWARGS[op])


@pytest.mark.parametrize("op", OPS)
def test_refusal_batch_below_min(op):
    a = _random_batch((BATCH_MIN - 1,), 3, seed=1230)
    _assert_refused(op, (a,), _OP_KWARGS[op])


@pytest.mark.parametrize("op", OPS)
def test_refusal_nan_entry(op):
    a = _random_batch((BATCH_MIN,), 3, seed=1240)
    a[7, 0, 0] = np.nan
    _assert_refused_raise_parity(op, (a,), _OP_KWARGS[op])


@pytest.mark.parametrize("op", OPS)
def test_refusal_inf_entry(op):
    """Infinite entry: full parity for svd/norm, DECISION ONLY for pinv.

    pinv is the one spelling that must not be executed here. On Linux,
    np.linalg.svd with compute_uv=True - which is what pinv calls - never
    returns for a matrix 3x3 or larger carrying an infinite entry on the
    DIAGONAL; the thread spins in LAPACK rather than blocking. Measured on
    GitHub's ubuntu runners, numpy 2.0.2/2.4.5/2.5.2, py3.12/3.13/3.14.
    The same call on Windows returns in well under a millisecond.
    compute_uv=False, 2x2, and an infinity off the diagonal are all
    unaffected, so SVD_OP (compute_uv=False) and NORM_OP (which reaches
    stock's own _multi_svd_norm, also compute_uv=False) keep their full
    raise-parity check. See docs/research/upstream-pinv-inf-hang.md.

    What PyOverdrive owns either way is the refusal itself: a non-finite
    batch must go to stock rather than through the closed form, and
    decide() reports that without running anything.
    """
    a = _random_batch((BATCH_MIN,), 3, seed=1240)
    a[7, 0, 0] = np.inf
    if op == PINV_OP:
        decision, reason = GEARBOX.decide(op, (a,), _OP_KWARGS[op])
        assert decision == "stock", (op, decision, reason)
        return
    _assert_refused_raise_parity(op, (a,), _OP_KWARGS[op])


def test_refusal_pinv_rcond_kwarg():
    a = _random_batch((BATCH_MIN,), 3, seed=1250)
    _assert_refused(PINV_OP, (a,), {"rcond": 1e-10})


def test_refusal_pinv_rtol_kwarg():
    a = _random_batch((BATCH_MIN,), 3, seed=1251)
    _assert_refused(PINV_OP, (a,), {"rtol": 1e-10})


def test_refusal_pinv_hermitian_kwarg():
    a = _random_batch((BATCH_MIN,), 3, seed=1252)
    a = a + np.swapaxes(a, -1, -2)
    _assert_refused(PINV_OP, (a,), {"hermitian": True})


def test_refusal_svd_default_wants_uv():
    a = _random_batch((BATCH_MIN,), 3, seed=1260)
    _assert_refused(SVD_OP, (a,), {})


def test_refusal_svd_full_matrices_false_wants_uv():
    a = _random_batch((BATCH_MIN,), 3, seed=1261)
    _assert_refused(SVD_OP, (a,), {"full_matrices": False})


def test_refusal_svd_hermitian_true():
    a = _random_batch((BATCH_MIN,), 3, seed=1262)
    a = a + np.swapaxes(a, -1, -2)
    _assert_refused(SVD_OP, (a,), {"compute_uv": False, "hermitian": True})


@pytest.mark.parametrize("ord_value", [None, "fro", 1, np.inf])
def test_refusal_norm2_other_ord(ord_value):
    a = _random_batch((BATCH_MIN,), 3, seed=1270)
    _assert_refused(NORM_OP, (a,), {"ord": ord_value, "axis": (-2, -1)})


def test_refusal_norm2_keepdims():
    a = _random_batch((BATCH_MIN,), 3, seed=1271)
    _assert_refused(NORM_OP, (a,), {"ord": 2, "axis": (-2, -1), "keepdims": True})


def test_refusal_norm2_axis_none():
    a = _random_batch((BATCH_MIN,), 3, seed=1272)
    _assert_refused_raise_parity(NORM_OP, (a,), {"ord": 2, "axis": None})


def test_refusal_norm2_non_trailing_axis_pair():
    a = _random_batch((BATCH_MIN,), 3, seed=1273)
    _assert_refused(NORM_OP, (a,), {"ord": 2, "axis": (0, 1)})


# ---------------------------------------------------------------------------
# 5. no RuntimeWarning on served calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [2, 3])
def test_no_runtime_warning_pinv(d):
    a = _random_batch((BATCH_MIN * 3,), d, seed=1100 + d)
    _assert_no_runtime_warning(lambda: np.linalg.pinv(a))


@pytest.mark.parametrize("d", [2, 3])
def test_no_runtime_warning_svdvals(d):
    a = _random_batch((BATCH_MIN * 3,), d, seed=1110 + d)
    _assert_no_runtime_warning(lambda: np.linalg.svd(a, compute_uv=False))


@pytest.mark.parametrize("d", [2, 3])
def test_no_runtime_warning_norm2(d):
    a = _random_batch((BATCH_MIN * 3,), d, seed=1120 + d)
    _assert_no_runtime_warning(lambda: np.linalg.norm(a, ord=2, axis=(-2, -1)))


# ---------------------------------------------------------------------------
# 6. kill switches
# ---------------------------------------------------------------------------


def test_kill_switch_pinv_restores_stock_routing():
    a = _random_batch((BATCH_MIN,), 3, seed=1300)
    decision, reason = GEARBOX.decide(PINV_OP, (a,), {})
    assert decision == PINV_PATH, (decision, reason)
    pyoverdrive.disable_path(PINV_PATH)
    try:
        decision, reason = GEARBOX.decide(PINV_OP, (a,), {})
        assert decision == "stock", (decision, reason)
        got = np.linalg.pinv(a)
        stock = _stock(PINV_OP, a)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PINV_PATH)


def test_kill_switch_svdvals_restores_stock_routing():
    a = _random_batch((BATCH_MIN,), 3, seed=1301)
    kwargs = {"compute_uv": False}
    decision, reason = GEARBOX.decide(SVD_OP, (a,), kwargs)
    assert decision == SVDVALS_PATH, (decision, reason)
    pyoverdrive.disable_path(SVDVALS_PATH)
    try:
        decision, reason = GEARBOX.decide(SVD_OP, (a,), kwargs)
        assert decision == "stock", (decision, reason)
        got = np.linalg.svd(a, **kwargs)
        stock = _stock(SVD_OP, a, **kwargs)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(SVDVALS_PATH)


def test_kill_switch_norm2_restores_stock_routing():
    a = _random_batch((BATCH_MIN,), 3, seed=1302)
    kwargs = {"ord": 2, "axis": (-2, -1)}
    decision, reason = GEARBOX.decide(NORM_OP, (a,), kwargs)
    assert decision == NORM_PATH, (decision, reason)
    pyoverdrive.disable_path(NORM_PATH)
    try:
        decision, reason = GEARBOX.decide(NORM_OP, (a,), kwargs)
        assert decision == "stock", (decision, reason)
        got = np.linalg.norm(a, **kwargs)
        stock = _stock(NORM_OP, a, **kwargs)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(NORM_PATH)


# ---------------------------------------------------------------------------
# 7. near-degenerate singular values: the SECOND d=3 hazard
#
# A coalescing PAIR of singular values wrecks the trigonometric solution's
# accuracy no matter how well conditioned the matrix is, so the sigma-ratio
# band cannot see it coming. Before DEGENERACY_MIN existed, every case below
# shipped 4.3e-9 to 1.3e-8 absolute error against ||A|| - four to thirteen
# times outside the contract - while the band happily passed them (their
# sigma_min/sigma_max is 0.3, i.e. perfectly healthy). These pin that shut.
# ---------------------------------------------------------------------------

def _from_singular_values(n, d, svals, seed):
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((n, d, d)))
    v, _ = np.linalg.qr(rng.standard_normal((n, d, d)))
    s = np.asarray(svals, dtype=np.float64)
    return np.ascontiguousarray(u @ (s[None, :, None] * np.swapaxes(v, -1, -2)))


_DEGENERATE_CASES = [
    pytest.param((1.0, 1.0, 0.3), id="top-pair-exactly-equal"),
    pytest.param((1.0, 1.0 - 1e-9, 0.3), id="top-pair-1e-9-apart"),
    pytest.param((1.0, 1.0 - 1e-13, 0.3), id="top-pair-1e-13-apart"),
    pytest.param((1.0, 0.3, 0.3), id="bottom-pair-exactly-equal"),
    pytest.param((1.0, 0.5, 0.5 - 1e-11), id="bottom-pair-near-equal"),
]


@pytest.mark.parametrize("svals", _DEGENERATE_CASES)
def test_degenerate_pair_svdvals_matches_stock(svals):
    a = _from_singular_values(400, 3, svals, seed=910)
    decision, reason = GEARBOX.decide(SVD_OP, (a,), {"compute_uv": False})
    assert decision == SVDVALS_PATH, (decision, reason)
    got = np.linalg.svd(a, compute_uv=False)
    stock = _stock(SVD_OP, a, compute_uv=False)
    # served BY stock through the split, so these agree exactly
    assert np.array_equal(got, stock)


@pytest.mark.parametrize("svals", _DEGENERATE_CASES)
def test_degenerate_pair_norm2_matches_stock(svals):
    a = _from_singular_values(400, 3, svals, seed=911)
    kwargs = {"ord": 2, "axis": (-2, -1)}
    decision, reason = GEARBOX.decide(NORM_OP, (a,), kwargs)
    assert decision == NORM_PATH, (decision, reason)
    got = np.linalg.norm(a, **kwargs)
    stock = _stock(NORM_OP, a, **kwargs)
    assert np.array_equal(got, stock)


def test_exact_identity_multiple_is_not_treated_as_degenerate():
    # all three singular values equal is the p == 0 branch, which is EXACT;
    # refusing it would be a needless loss, so it must still be served fast
    a = _from_singular_values(400, 3, (7.0, 7.0, 7.0), seed=912)
    got = np.linalg.svd(a, compute_uv=False)
    stock = _stock(SVD_OP, a, compute_uv=False)
    assert np.all(np.abs(got - stock) <= 1e-12 * stock[..., :1])


def test_degeneracy_split_leaves_most_of_a_random_stack_on_the_fast_path():
    # the guard must cost accuracy-free matrices nothing: a random stack
    # should divert only a tiny fraction, or the win is gone
    from pyoverdrive.fastpaths.svd_small_batch import _sv3_squared

    rng = np.random.default_rng(913)
    a = np.ascontiguousarray(rng.standard_normal((20_000, 3, 3)))
    hi, _mid, lo, margin = _sv3_squared(a)
    diverted = ((lo <= SVDVALS_SIGMA_RATIO_MIN[3] ** 2 * hi)
                | (margin < DEGENERACY_MIN))
    assert diverted.mean() < 0.02, diverted.mean()


@pytest.mark.parametrize(
    "svals",
    [pytest.param((1.0, 1.0), id="pair-exactly-equal"),
     pytest.param((1.0, 1.0 - 1e-10), id="pair-1e-10-apart"),
     pytest.param((2.5, 2.5), id="pair-equal-scaled")],
)
def test_d2_degenerate_pair_matches_stock(svals):
    # 2x2 has no arccos, but it has the same hazard by a DIFFERENT
    # mechanism: the discriminant t^2 - 4*det cancels to zero exactly when
    # the pair coalesces, and sqrt of a cancelling difference carries
    # ~sqrt(eps) error. Measured 7.5e-9 before DEGENERACY_MIN covered d=2 -
    # this test is what caught the wrong "2x2 needs no guard" assumption.
    a = _from_singular_values(400, 2, svals, seed=914)
    decision, reason = GEARBOX.decide(SVD_OP, (a,), {"compute_uv": False})
    assert decision == SVDVALS_PATH, (decision, reason)
    got = np.linalg.svd(a, compute_uv=False)
    stock = _stock(SVD_OP, a, compute_uv=False)
    assert np.array_equal(got, stock)
    kwargs = {"ord": 2, "axis": (-2, -1)}
    assert np.array_equal(np.linalg.norm(a, **kwargs), _stock(NORM_OP, a, **kwargs))
