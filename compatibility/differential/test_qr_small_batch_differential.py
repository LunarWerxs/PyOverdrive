"""Differential tests: qr_small_batch fast path vs stock numpy.linalg.qr.

Contract (src/pyoverdrive/fastpaths/qr_small_batch.py): applies only to
qr(a) / qr(a, mode) / qr(a, mode=...) with mode in {'reduced', 'complete',
'r'}, where a is a plain float64 ndarray shaped (..., d, d), d in {2, 3},
ndim >= 3, at least BATCH_MIN matrices, and every element finite. 'raw'
mode, non-finite input, wrong dtype/shape/ndim, duplicate mode arguments,
and unknown kwargs all stay on stock. Q/R modes return stock's QRResult
namedtuple; 'r' returns the plain R ndarray, exactly like stock. Signs are
exact by construction (LAPACK's dlarfg convention); numeric agreement is
checked at a scaled tolerance.
"""

import math

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.qr_small_batch import _FLOORS, BATCH_MIN

OP = "numpy.linalg.qr"
PATH = "qr_small_batch"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _random_batch(batch_shape, d, seed, dtype=np.float64):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(*batch_shape, d, d)).astype(dtype)


def _upper_triangular_batch(batch_shape, d, seed, dtype=np.float64):
    a = _random_batch(batch_shape, d, seed, dtype)
    il = np.tril_indices(d, k=-1)
    a[..., il[0], il[1]] = 0.0
    return a


def _negative_leading_batch(batch_shape, d, seed, dtype=np.float64):
    a = _random_batch(batch_shape, d, seed, dtype)
    a[..., 0, 0] = -np.abs(a[..., 0, 0]) - 0.1
    return a


def _zero_first_column_batch(batch_shape, d, seed, dtype=np.float64):
    a = _random_batch(batch_shape, d, seed, dtype)
    a[..., :, 0] = 0.0
    return a


def _zero_batch(batch_shape, d, dtype=np.float64):
    return np.zeros((*batch_shape, d, d), dtype=dtype)


def _rank1_batch(batch_shape, d, seed, dtype=np.float64):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=(*batch_shape, d, 1)).astype(dtype)
    v = rng.uniform(-1.0, 1.0, size=(*batch_shape, 1, d)).astype(dtype)
    return (u @ v).astype(dtype)


def _same(got, stock):
    """NaN-aware structural equality for QRResult / plain ndarray / raw tuple."""
    if isinstance(stock, np.ndarray):
        return np.array_equal(got, stock, equal_nan=True)
    if isinstance(stock, tuple):
        return type(got) is type(stock) and len(got) == len(stock) and all(
            np.array_equal(g, s, equal_nan=True) for g, s in zip(got, stock)
        )
    return got == stock


def _assert_dispatched(a, *extra_args, kwargs=None):
    kwargs = kwargs or {}
    args = (a, *extra_args)
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.linalg.qr(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    d = a.shape[-1]
    il = np.tril_indices(d, k=-1)
    if isinstance(stock, np.ndarray):
        tol = 1e-9 * max(1.0, float(np.max(np.abs(stock))))
        assert np.all(np.abs(got - stock) <= tol)
        assert np.all(got[..., il[0], il[1]] == 0.0)
    else:
        gq, gr = got
        sq, sr = stock
        tol = 1e-9 * max(1.0, float(np.max(np.abs(sr))))
        assert np.all(np.abs(gq - sq) <= tol)
        assert np.all(np.abs(gr - sr) <= tol)
        assert np.all(gr[..., il[0], il[1]] == 0.0)
    return got, stock


def _assert_refused(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.linalg.qr(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert _same(got, stock)
    return got


def _assert_refused_raise_parity(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)

    def _call(fn):
        try:
            return ("ok", fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - parity capture, not handling
            return ("raised", exc)

    got_kind, got_val = _call(lambda *a, **k: np.linalg.qr(*a, **k))
    stock_kind, stock_val = _call(_stock)
    assert got_kind == stock_kind, (got_kind, got_val, stock_kind, stock_val)
    if got_kind == "raised":
        assert type(got_val) is type(stock_val)
    else:
        assert type(got_val) is type(stock_val)
        assert _same(got_val, stock_val)


# ---------------------------------------------------------------------------
# 1. dispatch + numeric agreement, all mode spellings, d=2 and d=3
# ---------------------------------------------------------------------------

_MODE_CASES = [
    ("default", (), {}, 101),
    ("reduced_kwarg", (), {"mode": "reduced"}, 102),
    ("complete_positional", ("complete",), {}, 103),
    ("complete_kwarg", (), {"mode": "complete"}, 104),
    ("r_positional", ("r",), {}, 105),
    ("r_kwarg", (), {"mode": "r"}, 106),
]


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("label,extra_args,kwargs,seed", _MODE_CASES)
def test_dispatch_modes(d, label, extra_args, kwargs, seed):
    a = _random_batch((BATCH_MIN,), d, seed=seed + d)
    got, stock = _assert_dispatched(a, *extra_args, kwargs=kwargs)
    is_r = kwargs.get("mode") == "r" or (extra_args and extra_args[0] == "r")
    if is_r:
        assert type(got) is np.ndarray
    else:
        assert isinstance(got, tuple) and hasattr(got, "_fields")


# ---------------------------------------------------------------------------
# 2. sign exactness classes - the path's whole contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [2, 3])
def test_sign_exact_upper_triangular_identity_reflector(d):
    a = _upper_triangular_batch((BATCH_MIN,), d, seed=200 + d)
    _assert_dispatched(a)


@pytest.mark.parametrize("d", [2, 3])
def test_sign_exact_negative_leading_entries(d):
    a = _negative_leading_batch((BATCH_MIN,), d, seed=210 + d)
    _assert_dispatched(a)


@pytest.mark.parametrize("d", [2, 3])
def test_sign_exact_zero_first_column(d):
    a = _zero_first_column_batch((BATCH_MIN,), d, seed=220 + d)
    _assert_dispatched(a)


@pytest.mark.parametrize("d", [2, 3])
def test_sign_exact_fully_singular_zero_matrix(d):
    a = _zero_batch((BATCH_MIN,), d)
    _assert_dispatched(a)


def test_sign_exact_rank1_d2():
    # d=2 has a single reflector: well-conditioned even for rank-1 input.
    a = _rank1_batch((BATCH_MIN,), 2, seed=232)
    _assert_dispatched(a)


def test_sign_exact_rank1_d3():
    # d=3's second reflector pivots on the trailing 2x2 block, which is
    # mathematically exactly zero for a rank-1 input - noise grade, where
    # dlarfg's discontinuities can make two valid factorizations disagree
    # at full scale. The QR_RTOL determinism band detects this mid-run and
    # serves the whole call BY stock, so the result must equal stock's
    # EXACTLY (decision still reads PATH; the fallback is inside _run).
    a = _rank1_batch((BATCH_MIN,), 3, seed=233)
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision == PATH, (decision, reason)
    got = np.linalg.qr(a)
    stock = _stock(a)
    assert type(got) is type(stock)
    gq, gr = got
    sq, sr = stock
    assert np.array_equal(gq, sq)
    assert np.array_equal(gr, sr)


# ---------------------------------------------------------------------------
# 3. multi-dim leading batch shape
# ---------------------------------------------------------------------------

def test_dispatch_multidim_leading_batch_shape():
    n = math.ceil(BATCH_MIN / 20)
    a = _random_batch((20, n), 3, seed=301)
    assert a.shape == (20, n, 3, 3)
    assert 20 * n >= BATCH_MIN
    _assert_dispatched(a)


# ---------------------------------------------------------------------------
# 4. refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [2, 3])
def test_refusal_batch_below_min(d):
    a = _random_batch((_FLOORS[d] - 1,), d, seed=401 + d)
    _assert_refused((a,), {})


def test_refusal_float32_input():
    a = _random_batch((BATCH_MIN,), 3, seed=402, dtype=np.float32)
    _assert_refused((a,), {})


def test_refusal_nan_entry():
    a = _random_batch((BATCH_MIN,), 3, seed=403)
    a[7, 0, 0] = np.nan
    _assert_refused_raise_parity((a,), {})


def test_refusal_inf_entry():
    a = _random_batch((BATCH_MIN,), 3, seed=404)
    a[7, 1, 0] = np.inf
    _assert_refused_raise_parity((a,), {})


def test_refusal_4x4_batch():
    a = _random_batch((BATCH_MIN,), 4, seed=405)
    _assert_refused((a,), {})


def test_refusal_rectangular_batch():
    rng = np.random.default_rng(406)
    a = rng.uniform(-1.0, 1.0, size=(BATCH_MIN, 3, 2))
    _assert_refused((a,), {})


def test_refusal_single_2d_matrix():
    a = _random_batch((), 3, seed=407)
    assert a.shape == (3, 3)
    _assert_refused((a,), {})


def test_refusal_mode_raw():
    a = _random_batch((BATCH_MIN,), 3, seed=408)
    _assert_refused((a,), {"mode": "raw"})


def test_refusal_duplicate_mode_positional_and_kwarg():
    a = _random_batch((BATCH_MIN,), 3, seed=409)
    _assert_refused_raise_parity((a, "reduced"), {"mode": "complete"})


def test_refusal_unknown_kwarg():
    a = _random_batch((BATCH_MIN,), 3, seed=410)
    _assert_refused_raise_parity((a,), {"unknown": True})


# ---------------------------------------------------------------------------
# 5. kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_restores_stock_routing():
    a = _random_batch((BATCH_MIN,), 3, seed=501)
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (a,), {})
        assert decision == "stock", (decision, reason)
        got = np.linalg.qr(a)
        stock = _stock(a)
        assert _same(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)


# ---------------------------------------------------------------------------
# 6. reflector-2 band split-and-recombine (d=3 only)
# ---------------------------------------------------------------------------

def test_band_trippers_split_out_and_served_by_stock():
    # a few rank-deficient matrices scattered in a random stack: the fast
    # route serves the rest, the trippers get stock's EXACT values
    rng = np.random.default_rng(431)
    a = _random_batch((BATCH_MIN * 4,), 3, seed=431)
    deficient = _rank1_batch((5,), 3, seed=433)
    planted = rng.choice(a.shape[0], 5, replace=False)
    a[planted] = deficient
    decision, reason = GEARBOX.decide(OP, (a,), {})
    assert decision == PATH, (decision, reason)
    got = np.linalg.qr(a)
    stock = _stock(a)
    assert np.array_equal(got.Q[planted], stock.Q[planted])
    assert np.array_equal(got.R[planted], stock.R[planted])
    tol = 1e-9 * max(1.0, float(np.max(np.abs(stock.R))))
    assert np.all(np.abs(got.R - stock.R) <= tol)
    assert np.all(np.abs(got.Q - stock.Q) <= 1e-9)


def test_band_trippers_split_mode_r():
    a = _random_batch((BATCH_MIN * 4,), 3, seed=435)
    a[7] = _rank1_batch((1,), 3, seed=437)[0]
    got = np.linalg.qr(a, mode="r")
    stock = _stock(a, mode="r")
    assert np.array_equal(got[7], stock[7])
    tol = 1e-9 * max(1.0, float(np.max(np.abs(stock))))
    assert np.all(np.abs(got - stock) <= tol)
