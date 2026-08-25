"""Differential tests: linalg_small_batch fast paths vs stock numpy.linalg.

Covers det_small_batch (numpy.linalg.det), slogdet_small_batch
(numpy.linalg.slogdet), solve_small_batch (numpy.linalg.solve): closed
forms on float64 (..., d, d) stacks, d in {2, 3}, guarded by the shared
DET_RTOL conditioning check from inv_small_batch. Comparison mode is
numeric (allclose rtol=1e-9, atol=1e-12), except slogdet signs, which
must be exactly equal, and slogdet's result type, which must match
stock's namedtuple type. Refusals are checked against stock exactly
(same underlying call either way).
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.linalg_small_batch import _WINDOWS

OP_DET = "numpy.linalg.det"
OP_SLOGDET = "numpy.linalg.slogdet"
OP_SOLVE = "numpy.linalg.solve"
PATH_DET = "det_small_batch"
PATH_SLOGDET = "slogdet_small_batch"
PATH_SOLVE = "solve_small_batch"

# Derived from the module, never a second copy: a hardcoded mirror of the
# windows drifts the moment they are re-measured, and these were re-measured
# once already when the old floors turned out to sit below break-even.
DET_FLOOR = {d: lo for (kind, d), (lo, _hi) in _WINDOWS.items() if kind == "det"}
SLOGDET_FLOOR = {d: lo for (kind, d), (lo, _hi) in _WINDOWS.items() if kind == "slogdet"}
SOLVE_FLOOR = {d: lo for (kind, d), (lo, _hi) in _WINDOWS.items() if kind == "solve"}
DET_CAP = {d: hi for (kind, d), (_lo, hi) in _WINDOWS.items() if kind == "det"}

_DISPATCH_FN = {
    OP_DET: lambda: np.linalg.det,
    OP_SLOGDET: lambda: np.linalg.slogdet,
    OP_SOLVE: lambda: np.linalg.solve,
}


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP_DET, OP_SLOGDET, OP_SOLVE])
    yield
    pyoverdrive.disable()


def _stock(op):
    return GEARBOX.stock_fn(op)


def _well_conditioned(batch_shape, d, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(batch_shape + (d, d)) * 0.1
    a = a + np.eye(d, dtype=np.float64) * 3.0
    return a.astype(np.float64)


def _rhs(batch_shape, d, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(batch_shape + (d, 1)).astype(np.float64)


def _bad_singular_3x3():
    # row1 == 2 * row0 exactly -> det == 0 exactly
    return np.array(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]], dtype=np.float64
    )


def _bad_near_singular_3x3():
    m = _bad_singular_3x3()
    m[1, 2] += 1e-12
    det = np.linalg.det(m)
    scale = np.abs(m).max()
    assert abs(det) / scale**3 < 1e-10, (det, scale)
    return m


def _call(fn, args, kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as e:  # noqa: BLE001 - symmetric probe, any exception type
        return ("err", type(e))


def _assert_dispatched_det(a):
    decision, reason = GEARBOX.decide(OP_DET, (a,), {})
    assert decision == PATH_DET, (decision, reason)
    got = np.linalg.det(a)
    stock = _stock(OP_DET)(a)
    assert np.allclose(got, stock, rtol=1e-9, atol=1e-12)


def _assert_dispatched_slogdet(a):
    decision, reason = GEARBOX.decide(OP_SLOGDET, (a,), {})
    assert decision == PATH_SLOGDET, (decision, reason)
    got = np.linalg.slogdet(a)
    stock = _stock(OP_SLOGDET)(a)
    assert type(got) is type(stock)
    assert np.array_equal(got.sign, stock.sign)
    assert np.allclose(got.logabsdet, stock.logabsdet, rtol=1e-9, atol=1e-12)


def _assert_dispatched_solve(a, b):
    decision, reason = GEARBOX.decide(OP_SOLVE, (a, b), {})
    assert decision == PATH_SOLVE, (decision, reason)
    got = np.linalg.solve(a, b)
    stock = _stock(OP_SOLVE)(a, b)
    assert np.allclose(got, stock, rtol=1e-9, atol=1e-12)


def _assert_guarded_at_run(op, args, kwargs=None):
    """For det/slogdet, whose guard was FUSED INTO THE RUN.

    The predicate deliberately accepts these calls now: asking it to prove
    finiteness and non-singularity meant computing the determinant in the
    predicate and again in the run, which cost the entire margin (the 2x2
    path measured 0.70x end to end at its own floor - slower than stock
    while still dispatching). The guard runs once, mid-run, and hands the
    whole call to stock on refusal.

    So the DECISION is the path and the BEHAVIOUR is stock's. That second
    half is the part that matters, and it is asserted exactly as strictly
    as before.
    """
    kwargs = kwargs or {}
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision != "stock", (decision, reason)
    dispatched_fn = _DISPATCH_FN[op]()
    got_tag, got = _call(dispatched_fn, args, kwargs)
    stock_tag, stock = _call(_stock(op), args, kwargs)
    assert got_tag == stock_tag, (got_tag, got, stock_tag, stock)
    if got_tag == "err":
        assert type(got) is type(stock) and str(got) == str(stock), (got, stock)
        return
    if op == OP_SLOGDET:
        assert type(got) is type(stock)
        for g, s in zip(got, stock):
            assert g.dtype == s.dtype and g.shape == s.shape
            assert np.array_equal(g, s, equal_nan=True)
        return
    assert got.dtype == stock.dtype and got.shape == stock.shape
    assert np.array_equal(got, stock, equal_nan=True)


def _assert_refused(op, args, kwargs=None):
    kwargs = kwargs or {}
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == "stock", (decision, reason)
    dispatched_fn = _DISPATCH_FN[op]()
    got_tag, got = _call(dispatched_fn, args, kwargs)
    stock_tag, stock = _call(_stock(op), args, kwargs)
    assert got_tag == stock_tag, (got_tag, got, stock_tag, stock)
    if got_tag == "err":
        assert got is stock, (got, stock)
        return
    if op == OP_SLOGDET:
        assert type(got) is type(stock)
        assert np.array_equal(got.sign, stock.sign, equal_nan=True)
        assert np.array_equal(got.logabsdet, stock.logabsdet, equal_nan=True)
    else:
        assert np.array_equal(got, stock, equal_nan=True)


# --- 1. well-conditioned dispatch -------------------------------------


def test_dispatch_det_3x3():
    a = _well_conditioned((1000,), 3, seed=1)
    _assert_dispatched_det(a)


def test_dispatch_det_2x2():
    a = _well_conditioned((500,), 2, seed=2)
    _assert_dispatched_det(a)


def test_dispatch_slogdet_3x3():
    a = _well_conditioned((1000,), 3, seed=3)
    _assert_dispatched_slogdet(a)


def test_dispatch_slogdet_2x2():
    a = _well_conditioned((500,), 2, seed=4)
    _assert_dispatched_slogdet(a)


def test_dispatch_solve_3x3():
    a = _well_conditioned((2000,), 3, seed=5)
    b = _rhs((2000,), 3, seed=6)
    _assert_dispatched_solve(a, b)


def test_dispatch_solve_2x2():
    a = _well_conditioned((500,), 2, seed=7)
    b = _rhs((500,), 2, seed=8)
    _assert_dispatched_solve(a, b)


# --- 2. multi-dim leading batch shape -----------------------------------


def test_dispatch_det_multidim_batch():
    a = _well_conditioned((20, 60), 3, seed=9)  # batch = 1200 >= 300
    _assert_dispatched_det(a)


# --- 3. guard refusals: singular / near-singular / non-finite -----------


def test_refusal_exactly_singular_det():
    a = _well_conditioned((300,), 3, seed=10)
    a[150] = _bad_singular_3x3()
    _assert_guarded_at_run(OP_DET, (a,))


def test_refusal_exactly_singular_solve():
    a = _well_conditioned((1000,), 3, seed=11)
    a[500] = _bad_singular_3x3()
    b = _rhs((1000,), 3, seed=12)
    _assert_refused(OP_SOLVE, (a, b))


def test_refusal_near_singular_det():
    a = _well_conditioned((300,), 3, seed=13)
    a[100] = _bad_near_singular_3x3()
    _assert_guarded_at_run(OP_DET, (a,))


def test_refusal_near_singular_solve():
    a = _well_conditioned((1000,), 3, seed=14)
    a[700] = _bad_near_singular_3x3()
    b = _rhs((1000,), 3, seed=15)
    _assert_refused(OP_SOLVE, (a, b))


def test_refusal_inf_entry_det():
    a = _well_conditioned((300,), 3, seed=16)
    a[42, 0, 0] = np.inf
    _assert_guarded_at_run(OP_DET, (a,))


def test_refusal_nan_entry_solve():
    a = _well_conditioned((1000,), 3, seed=17)
    a[900, 1, 2] = np.nan
    b = _rhs((1000,), 3, seed=18)
    _assert_refused(OP_SOLVE, (a, b))


# --- 4. shape / floor refusals ------------------------------------------


def test_refusal_single_matrix_ndim2():
    a = _well_conditioned((), 3, seed=19)  # shape (3, 3), ndim == 2
    _assert_refused(OP_DET, (a,))


def test_agreement_4x4_stack():
    """4x4 is served now, by Laplace expansion on complementary 2x2 minors."""
    a = _well_conditioned((DET_FLOOR[4],), 4, seed=20)
    decision, reason = GEARBOX.decide(OP_DET, (a,), {})
    assert decision == PATH_DET, (decision, reason)
    got, stock = np.linalg.det(a), _stock(OP_DET)(a)
    assert got.dtype == stock.dtype and got.shape == stock.shape
    assert np.allclose(got, stock, rtol=1e-9, atol=0.0)


def test_refusal_4x4_above_its_cap():
    """The 4x4 margin decays to ~1.09x by 100_000, so the window has a cap."""
    cap = DET_CAP[4]
    assert cap is not None
    a = _well_conditioned((cap + 1,), 4, seed=201)
    _assert_refused(OP_DET, (a,))


def test_refusal_5x5_stack():
    a = _well_conditioned((1000,), 5, seed=202)
    _assert_refused(OP_DET, (a,))


def test_refusal_below_floor_det_2x2():
    a = _well_conditioned((DET_FLOOR[2] - 50,), 2, seed=21)  # below the 2x2 floor
    _assert_refused(OP_DET, (a,))


def test_refusal_below_floor_det_3x3():
    a = _well_conditioned((DET_FLOOR[3] - 100,), 3, seed=22)  # below the 3x3 floor
    _assert_refused(OP_DET, (a,))


def test_refusal_below_floor_solve_3x3():
    n = SOLVE_FLOOR[3] - 500  # below the 3x3 solve floor
    a = _well_conditioned((n,), 3, seed=23)
    b = _rhs((n,), 3, seed=24)
    _assert_refused(OP_SOLVE, (a, b))


def test_refusal_float32_stack():
    a = _well_conditioned((500,), 3, seed=25).astype(np.float32)
    _assert_refused(OP_DET, (a,))


def test_refusal_solve_b_shape_batch_d():
    # stock raises on b shaped (N, d) in numpy 2.x; both must raise identically
    a = _well_conditioned((1500,), 3, seed=26)
    b_bad = _rhs((1500,), 3, seed=27)[..., 0]  # shape (1500, 3)
    _assert_refused(OP_SOLVE, (a, b_bad))


def test_refusal_solve_b_shape_batch_d_k():
    # b shaped (N, d, 2) is a valid multi-RHS stock call, just not the
    # fast path's supported column form -> refused, results equal
    a = _well_conditioned((1500,), 3, seed=28)
    rng = np.random.default_rng(29)
    b_bad = rng.standard_normal((1500, 3, 2)).astype(np.float64)
    _assert_refused(OP_SOLVE, (a, b_bad))


def test_refusal_kwargs():
    a = _well_conditioned((500,), 3, seed=30)
    b = _rhs((500,), 3, seed=31)
    _assert_refused(OP_DET, (a,), {"foo": 1})
    _assert_refused(OP_SLOGDET, (a,), {"foo": 1})
    _assert_refused(OP_SOLVE, (a, b), {"foo": 1})


# --- 5. kill switch -------------------------------------------------------


def test_kill_switch_restores_stock_routing():
    a_det = _well_conditioned((600,), 3, seed=40)  # clears det AND slogdet 3x3
    a_solve = _well_conditioned((1100,), 3, seed=41)
    b_solve = _rhs((1100,), 3, seed=42)

    cases = [
        (PATH_DET, OP_DET, (a_det,)),
        (PATH_SLOGDET, OP_SLOGDET, (a_det,)),
        (PATH_SOLVE, OP_SOLVE, (a_solve, b_solve)),
    ]
    for path, op, args in cases:
        decision, reason = GEARBOX.decide(op, args, {})
        assert decision == path, (path, decision, reason)
        pyoverdrive.disable_path(path)
        try:
            decision, reason = GEARBOX.decide(op, args, {})
            assert decision == "stock", (path, decision, reason)
            fn = _DISPATCH_FN[op]()
            got = fn(*args)
            stock = _stock(op)(*args)
            if op == OP_SLOGDET:
                assert np.array_equal(got.sign, stock.sign)
                assert np.array_equal(got.logabsdet, stock.logabsdet)
            else:
                assert np.array_equal(got, stock)
        finally:
            pyoverdrive.enable_path(path)
