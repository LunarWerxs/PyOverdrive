"""Differential tests: einsum_optimize fast path vs stock numpy.einsum.

Contract (src/pyoverdrive/fastpaths/einsum_optimize.py): applies only to the
subscripts-string form einsum(subs, a, b) with exactly two positional
operands, no keyword arguments, both plain ndarrays of the same
float64/float32 dtype, subscripts that parse cleanly (alphabetic labels, no
ellipsis, exactly one comma, per-operand label count equal to that
operand's ndim), and a size gate on min(a.size, b.size): >= 10_000
(PROJECTED_FLOOR) for subscripts whose output keeps at least one index, or
>= 1_000_000 (SCALAR_FLOOR) for subscripts that reduce to a scalar (explicit
'->' with an empty right-hand side, or an implicit form where every label
repeats). Dispatch routes to stock np.einsum(..., optimize=True) and
normalizes its 0-d array back to a numpy scalar for scalar-output
subscripts to match stock's own return type. Results are numerically equal,
not bit-identical (optimize=True sums in a different order), so equality
checks use an absolute-scaled tolerance rather than a purely relative one.
"""

import math
import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.einsum_optimize import (
    CHAIN_ELLIPSIS_VOLUME_FLOOR,
    CHAIN_SCALAR_VOLUME_FLOOR,
    CHAIN_VOLUME_FLOOR,
    PROJECTED_FLOOR,
)

RNG = np.random.default_rng(20260823)


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(["numpy.einsum"])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn("numpy.einsum")(*args, **kwargs)


def _pos(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 1.5, size=shape).astype(dtype)


def _tol(stock, dtype):
    s = max(1.0, float(np.abs(stock).max()))
    if np.dtype(dtype) == np.dtype(np.float64):
        return 1e-6, 1e-9 * s
    return 1e-3, 1e-4 * s


def _assert_close(got, stock, dtype):
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    rtol, atol = _tol(stock, dtype)
    assert np.allclose(got, stock, rtol=rtol, atol=atol)


def _assert_dispatched_close(subs, a, b, scalar=False):
    decision, reason = GEARBOX.decide("numpy.einsum", (subs, a, b), {})
    assert decision == "einsum_optimize", (decision, reason)
    got = np.einsum(subs, a, b)
    stock = _stock(subs, a, b)
    if scalar:
        assert type(got) is type(stock)
    _assert_close(got, stock, a.dtype)
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide("numpy.einsum", args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.einsum(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert np.array_equal(got, stock)
    return got


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide("numpy.einsum", args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.einsum(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


def _assert_dispatched_raises_same(args, kwargs):
    decision, reason = GEARBOX.decide("numpy.einsum", args, kwargs)
    assert decision == "einsum_optimize", (decision, reason)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(Exception) as got_exc:
            np.einsum(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + numeric equality (projected output)
# ---------------------------------------------------------------------------

PROJECTED_CASES = [
    pytest.param("ij,jk->ik", (120, 120), (120, 120), np.float64, id="matmul-f64"),
    pytest.param("ij,jk->ik", (120, 120), (120, 120), np.float32, id="matmul-f32"),
    pytest.param("thd,Thd->thT", (120, 1, 120), (120, 1, 120), np.float64, id="batched-thd"),
    pytest.param("ijk,ikl->ijl", (8, 40, 40), (8, 40, 40), np.float64, id="threeidx"),
    pytest.param("ij,jk", (120, 120), (120, 120), np.float64, id="implicit-projected"),
]


@pytest.mark.parametrize("subs, shape_a, shape_b, dtype", PROJECTED_CASES)
def test_dispatch_projected_numeric_equality(subs, shape_a, shape_b, dtype):
    a = _pos(shape_a, dtype, seed=1)
    b = _pos(shape_b, dtype, seed=2)
    _assert_dispatched_close(subs, a, b, scalar=False)


SCALAR_CASES = [
    pytest.param("i,i->", (1_000_000,), (1_000_000,), id="scalar-1d-i-i"),
    pytest.param("ij,ij->", (1000, 1000), (1000, 1000), id="scalar-2d-ij-ij"),
]


@pytest.mark.parametrize("subs, shape_a, shape_b", SCALAR_CASES)
def test_dispatch_scalar_numeric_equality(subs, shape_a, shape_b):
    a = _pos(shape_a, np.float64, seed=3)
    b = _pos(shape_b, np.float64, seed=4)
    _assert_dispatched_close(subs, a, b, scalar=True)


# ---------------------------------------------------------------------------
# 2. refusal routes
# ---------------------------------------------------------------------------

def test_refusal_below_projected_floor_matmul():
    a = _pos((30, 30), np.float64, seed=5)
    b = _pos((30, 30), np.float64, seed=6)
    _assert_refused_equal(("ij,jk->ik", a, b), {})


def test_refusal_len_100_below_both_floors():
    a = _pos((100,), np.float64, seed=7)
    b = _pos((100,), np.float64, seed=8)
    _assert_refused_equal(("i,i->", a, b), {})


def test_refusal_len_100000_above_projected_below_scalar_regime_guard():
    # size 100_000 clears PROJECTED_FLOOR but the output is scalar, which
    # needs SCALAR_FLOOR (1_000_000); this is the regime-mismatch case.
    a = _pos((100_000,), np.float64, seed=9)
    b = _pos((100_000,), np.float64, seed=10)
    _assert_refused_equal(("i,i->", a, b), {})


@pytest.mark.parametrize(
    "kwargs_fn",
    [
        lambda a, b: {"optimize": True},
        lambda a, b: {"optimize": False},
        lambda a, b: {"out": np.empty((120, 120), dtype=a.dtype)},
        lambda a, b: {"dtype": a.dtype},
        lambda a, b: {"casting": "safe"},
        lambda a, b: {"order": "C"},
    ],
    ids=["optimize-true", "optimize-false", "out", "dtype", "casting", "order"],
)
def test_refusal_any_kwarg(kwargs_fn):
    a = _pos((120, 120), np.float64, seed=11)
    b = _pos((120, 120), np.float64, seed=12)
    kwargs = kwargs_fn(a, b)
    _assert_refused_equal(("ij,jk->ik", a, b), kwargs)


def test_refusal_ellipsis():
    a = _pos((20_000,), np.float64, seed=13)
    b = _pos((20_000,), np.float64, seed=14)
    _assert_refused_equal(("...i,...i->...", a, b), {})


def test_refusal_interleaved_sublist_form():
    a = _pos((150, 150), np.float64, seed=15)
    b = _pos((150, 150), np.float64, seed=16)
    _assert_refused_equal((a, [0, 1], b, [1, 2]), {})


def test_refusal_three_operand():
    # Below CHAIN_VOLUME_FLOOR: with the chain path now registered for this
    # op (section 5 below), a 3-operand call is refused on volume, not on
    # operand count alone -- see test_chain_three_operand_matmul_above_floor
    # for the case where an equivalent shape above the floor dispatches.
    n = max(2, math.floor((CHAIN_VOLUME_FLOOR * 0.1) ** (1.0 / 4)))
    a = _pos((n, n), np.float64, seed=17)
    b = _pos((n, n), np.float64, seed=18)
    c = _pos((n, n), np.float64, seed=19)
    _assert_refused_equal(("ij,jk,kl->il", a, b, c), {})


def test_refusal_single_operand_transpose():
    a = _pos((150, 150), np.float64, seed=20)
    _assert_refused_equal(("ij->ji", a), {})


def test_refusal_single_operand_trace():
    a = _pos((150, 150), np.float64, seed=21)
    _assert_refused_equal(("ii->", a), {})


def test_refusal_mixed_dtypes():
    a = _pos((120, 120), np.float64, seed=22)
    b = _pos((120, 120), np.float32, seed=23)
    _assert_refused_equal(("ij,jk->ik", a, b), {})


def test_refusal_int64_operands():
    a = RNG.integers(1, 100, size=(120, 120)).astype(np.int64)
    b = RNG.integers(1, 100, size=(120, 120)).astype(np.int64)
    _assert_refused_equal(("ij,jk->ik", a, b), {})


def test_refusal_ndim_mismatch_raises_like_stock():
    a = _pos((200, 200), np.float64, seed=24)
    b = _pos((200, 200), np.float64, seed=25)
    _assert_refused_raises(("ijk,jk->ik", a, b), {})


def test_refusal_non_ndarray_operands():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    _assert_refused_equal(("ij,jk->ik", a, b), {})


def test_kill_switch_restores_stock_routing():
    a = _pos((120, 120), np.float64, seed=26)
    b = _pos((120, 120), np.float64, seed=27)
    decision, reason = GEARBOX.decide("numpy.einsum", ("ij,jk->ik", a, b), {})
    assert decision == "einsum_optimize", (decision, reason)
    pyoverdrive.disable_path("einsum_optimize")
    try:
        decision, reason = GEARBOX.decide("numpy.einsum", ("ij,jk->ik", a, b), {})
        assert decision == "stock", (decision, reason)
        got = np.einsum("ij,jk->ik", a, b)
        stock = _stock("ij,jk->ik", a, b)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path("einsum_optimize")


# ---------------------------------------------------------------------------
# 3. diag-style repeated label: assert consistency, not a predicted route
# ---------------------------------------------------------------------------

def test_repeated_label_probe_matches_stock_whichever_way_it_routes():
    a = _pos((150, 150), np.float64, seed=28)  # 'ii' needs a.ndim == len('ii')
    b = _pos((150, 150), np.float64, seed=29)
    decision, reason = GEARBOX.decide("numpy.einsum", ("ii,ij->j", a, b), {})
    got = np.einsum("ii,ij->j", a, b)
    stock = _stock("ii,ij->j", a, b)
    _assert_close(got, stock, a.dtype)


# ---------------------------------------------------------------------------
# 4. error parity: dispatches (above floor) but the contraction itself
#    raises; the dispatcher falls back to stock, which raises the same way.
# ---------------------------------------------------------------------------

def test_incompatible_inner_dims_above_floor_raises_valueerror_both_routes():
    a = _pos((100, 200), np.float64, seed=30)
    b = _pos((300, 100), np.float64, seed=31)
    decision, reason = GEARBOX.decide("numpy.einsum", ("ij,jk->ik", a, b), {})
    assert decision == "einsum_optimize", (decision, reason)
    _assert_dispatched_raises_same(("ij,jk->ik", a, b), {})
    with pytest.raises(ValueError):
        _stock("ij,jk->ik", a, b)


# ---------------------------------------------------------------------------
# 5. chain regime (einsum_optimize_chain): >=3 operands, gated on the naive
#    loop volume (product of distinct label extents), not any single
#    operand's size. Dims below are derived from CHAIN_VOLUME_FLOOR itself
#    (never hardcoded) so a later calibration update cannot strand these
#    thresholds above/below where they need to land.
# ---------------------------------------------------------------------------

def _n_above(distinct_labels, margin=3.0):
    """Per-label extent giving naive volume ~= margin * CHAIN_VOLUME_FLOOR
    for `distinct_labels` equal-extent labels (comfortably above the gate)."""
    return math.ceil((CHAIN_VOLUME_FLOOR * margin) ** (1.0 / distinct_labels))


def _n_below(distinct_labels, margin=0.25):
    """Per-label extent giving naive volume ~= margin * CHAIN_VOLUME_FLOOR
    (comfortably below the gate)."""
    return max(2, math.floor((CHAIN_VOLUME_FLOOR * margin) ** (1.0 / distinct_labels)))


_N3_ABOVE = _n_above(4)  # ij,jk,kl chain: 4 distinct labels (i,j,k,l)
_N3_BELOW = _n_below(4)
_N4_ABOVE = _n_above(5)  # ij,jk,kl,lm chain: 5 distinct labels (i,j,k,l,m)
# scalar-output chains gate on their own, higher floor
_N3_SCALAR_ABOVE = math.ceil((CHAIN_SCALAR_VOLUME_FLOOR * 3.0) ** 0.25)
_N3_SCALAR_BELOW = max(
    2, math.floor((CHAIN_SCALAR_VOLUME_FLOOR * 0.25) ** 0.25)
)


def _assert_chain_dispatched_close(subs, operands, scalar=False):
    args = (subs, *operands)
    decision, reason = GEARBOX.decide("numpy.einsum", args, {})
    assert decision == "einsum_optimize_chain", (decision, reason)
    got = np.einsum(*args)
    stock = _stock(*args)
    if scalar:
        assert type(got) is type(stock)
    _assert_close(got, stock, operands[0].dtype)
    return got, stock


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
def test_chain_three_operand_matmul_above_floor(dtype):
    n = _N3_ABOVE
    a = _pos((n, n), dtype, seed=40)
    b = _pos((n, n), dtype, seed=41)
    c = _pos((n, n), dtype, seed=42)
    _assert_chain_dispatched_close("ij,jk,kl->il", (a, b, c))


def test_chain_four_operand_above_floor():
    n = _N4_ABOVE
    a = _pos((n, n), np.float64, seed=43)
    b = _pos((n, n), np.float64, seed=44)
    c = _pos((n, n), np.float64, seed=45)
    d = _pos((n, n), np.float64, seed=46)
    _assert_chain_dispatched_close("ij,jk,kl,lm->im", (a, b, c, d))


def test_chain_scalar_output_returns_stock_scalar_type():
    n = _N3_SCALAR_ABOVE
    a = _pos((n, n), np.float64, seed=47)
    b = _pos((n, n), np.float64, seed=48)
    c = _pos((n, n), np.float64, seed=49)
    _assert_chain_dispatched_close("ij,jk,kl->", (a, b, c), scalar=True)


def test_chain_scalar_below_scalar_floor_refuses():
    # volume between the projected floor and the scalar floor: a projected
    # chain would dispatch here, a scalar one must not
    n = _N3_SCALAR_BELOW
    if n**4 < CHAIN_VOLUME_FLOOR:
        pytest.skip("floors too close to separate on this grid")
    a = _pos((n, n), np.float64, seed=57)
    b = _pos((n, n), np.float64, seed=58)
    c = _pos((n, n), np.float64, seed=59)
    decision, reason = GEARBOX.decide("numpy.einsum", ("ij,jk,kl->", a, b, c), {})
    assert decision != "einsum_optimize_chain", (decision, reason)


def test_chain_implicit_output_form():
    n = _N3_ABOVE
    a = _pos((n, n), np.float64, seed=50)
    b = _pos((n, n), np.float64, seed=51)
    c = _pos((n, n), np.float64, seed=52)
    _assert_chain_dispatched_close("ij,jk,kl", (a, b, c))


def test_chain_below_volume_floor_refusal():
    n = _N3_BELOW
    a = _pos((n, n), np.float64, seed=53)
    b = _pos((n, n), np.float64, seed=54)
    c = _pos((n, n), np.float64, seed=55)
    _assert_refused_equal(("ij,jk,kl->il", a, b, c), {})


def test_chain_two_operand_call_never_claimed_by_chain_path():
    a = _pos((120, 120), np.float64, seed=56)
    b = _pos((120, 120), np.float64, seed=57)
    decision, reason = GEARBOX.decide("numpy.einsum", ("ij,jk->ik", a, b), {})
    assert decision in ("einsum_optimize", "stock"), (decision, reason)
    assert decision != "einsum_optimize_chain", (decision, reason)


def test_chain_mixed_dtype_refusal():
    n = _N3_ABOVE
    a = _pos((n, n), np.float64, seed=58)
    b = _pos((n, n), np.float64, seed=59)
    c = _pos((n, n), np.float32, seed=60)
    _assert_refused_equal(("ij,jk,kl->il", a, b, c), {})


CHAIN_ELLIPSIS_REFUSAL_CASES = [
    pytest.param(
        "...ij,jk,kl->...il",
        (3, _N3_ABOVE, _N3_ABOVE),
        (_N3_ABOVE, _N3_ABOVE),
        (_N3_ABOVE, _N3_ABOVE),
        id="mixed-operands",
    ),
    pytest.param(
        "...ij,...jk,...kl->...il",
        (3, _N3_ABOVE, _N3_ABOVE),
        (1, _N3_ABOVE, _N3_ABOVE),
        (3, _N3_ABOVE, _N3_ABOVE),
        id="unequal-ellipsis-shapes",
    ),
    pytest.param(
        "...ij,...jk,...kl->...il",
        (_N3_ABOVE, _N3_ABOVE),
        (_N3_ABOVE, _N3_ABOVE),
        (_N3_ABOVE, _N3_ABOVE),
        id="zero-ellipsis-dims",
    ),
]


@pytest.mark.parametrize("subs, shape_a, shape_b, shape_c", CHAIN_ELLIPSIS_REFUSAL_CASES)
def test_chain_ellipsis_refusal(subs, shape_a, shape_b, shape_c):
    # Ellipsis chains ARE admitted now (batch 9) when every operand carries
    # '...' with equal, non-empty ellipsis shapes and an ellipsis-carrying
    # (or implicit) output. These four shapes each violate exactly one of
    # those conditions, so they must still refuse -- not "any ellipsis
    # refuses" any more.
    a = _pos(shape_a, np.float64, seed=61)
    b = _pos(shape_b, np.float64, seed=62)
    c = _pos(shape_c, np.float64, seed=63)
    _assert_refused_equal((subs, a, b, c), {})


def test_chain_ellipsis_output_lacks_ellipsis_raises_valueerror_both_routes():
    # Explicit '->' output omitting '...' while every input carries leftover
    # (unsummed) batch dims is invalid numpy, not implicit summation: both
    # routes raise the same ValueError.
    n = _N3_ABOVE
    batch = (3,)
    a = _pos(batch + (n, n), np.float64, seed=84)
    b = _pos(batch + (n, n), np.float64, seed=85)
    c = _pos(batch + (n, n), np.float64, seed=86)
    _assert_refused_raises(("...ij,...jk,...kl->il", a, b, c), {})


def test_chain_ellipsis_extent_mismatch_raises_valueerror_both_routes():
    n = _N3_ABOVE
    batch = (3,)
    a = _pos(batch + (n, n), np.float64, seed=73)  # i,j = n,n
    b = _pos(batch + (n + 5, n), np.float64, seed=74)  # j,k = n+5,n -- conflicts with a's j
    c = _pos(batch + (n, n), np.float64, seed=75)
    _assert_refused_raises(("...ij,...jk,...kl->...il", a, b, c), {})


# ---------------------------------------------------------------------------
# 6. ellipsis admission (batch 9): '...ij,...jk->...ik' class, two-operand
#    and chain. Sizes are derived from the imported floor constants so a
#    later calibration change cannot silently strand these above/below
#    where they need to land.
# ---------------------------------------------------------------------------

_ELL2_SHAPE = (2,)
_N_ELL2 = math.ceil((PROJECTED_FLOOR * 1.5 / math.prod(_ELL2_SHAPE)) ** 0.5)
# ell (2,) * n^2 per operand ~= 1.5 * PROJECTED_FLOOR: comfortably >= the floor.

TWO_OPERAND_ELLIPSIS_CASES = [
    pytest.param("...ij,...jk->...ik", id="explicit-output"),
    pytest.param("...ij,...jk", id="implicit-output"),
]


@pytest.mark.parametrize("subs", TWO_OPERAND_ELLIPSIS_CASES)
def test_two_operand_ellipsis_dispatch_numeric_equality(subs):
    n = _N_ELL2
    a = _pos(_ELL2_SHAPE + (n, n), np.float64, seed=82)
    b = _pos(_ELL2_SHAPE + (n, n), np.float64, seed=83)
    _assert_dispatched_close(subs, a, b, scalar=False)


_ELL_CHAIN_SHAPE_ABOVE = (4,)
_N_ELL_CHAIN_ABOVE = math.ceil(
    (CHAIN_ELLIPSIS_VOLUME_FLOOR * 4.0 / math.prod(_ELL_CHAIN_SHAPE_ABOVE)) ** (1.0 / 4)
)  # ell (4,) * n^4 across i,j,k,l ~= 4 * the ellipsis chain's own floor

_ELL_CHAIN_SHAPE_BELOW = (1,)
_N_ELL_CHAIN_BELOW = max(
    2,
    math.floor(
        (CHAIN_ELLIPSIS_VOLUME_FLOOR * 0.3 / math.prod(_ELL_CHAIN_SHAPE_BELOW))
        ** (1.0 / 4)
    ),
)  # ell (1,) * n^4 comfortably below the ellipsis chain's own floor


def test_chain_ellipsis_admitted_above_floor():
    n = _N_ELL_CHAIN_ABOVE
    a = _pos(_ELL_CHAIN_SHAPE_ABOVE + (n, n), np.float64, seed=76)
    b = _pos(_ELL_CHAIN_SHAPE_ABOVE + (n, n), np.float64, seed=77)
    c = _pos(_ELL_CHAIN_SHAPE_ABOVE + (n, n), np.float64, seed=78)
    _assert_chain_dispatched_close("...ij,...jk,...kl->...il", (a, b, c))


def test_chain_ellipsis_below_floor_refusal():
    n = _N_ELL_CHAIN_BELOW
    a = _pos(_ELL_CHAIN_SHAPE_BELOW + (n, n), np.float64, seed=79)
    b = _pos(_ELL_CHAIN_SHAPE_BELOW + (n, n), np.float64, seed=80)
    c = _pos(_ELL_CHAIN_SHAPE_BELOW + (n, n), np.float64, seed=81)
    _assert_refused_equal(("...ij,...jk,...kl->...il", a, b, c), {})


def test_chain_extent_mismatch_raises_valueerror_both_routes():
    n = _N3_ABOVE
    a = _pos((n, n), np.float64, seed=64)  # i,j = n,n
    b = _pos((n + 5, n), np.float64, seed=65)  # j,k = n+5,n -- conflicts with a's j
    c = _pos((n, n), np.float64, seed=66)  # k,l = n,n
    _assert_refused_raises(("ij,jk,kl->il", a, b, c), {})


def test_chain_malformed_repeated_output_label_raises_valueerror_both_routes():
    n = _N3_ABOVE
    a = _pos((n, n), np.float64, seed=67)
    b = _pos((n, n), np.float64, seed=68)
    c = _pos((n, n), np.float64, seed=69)
    _assert_refused_raises(("ij,jk,kl->iil", a, b, c), {})
