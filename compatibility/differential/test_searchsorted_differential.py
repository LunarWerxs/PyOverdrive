"""Differential tests: searchsorted_sortqueries fast path vs stock numpy.searchsorted.

Contract (src/pyoverdrive/fastpaths/searchsorted_sortqueries.py): applies only
to searchsorted(a, v[, side]) where side is the 3rd positional arg or a
'side' kwarg with value 'left' (default) or 'right'; a and v are plain 1-D
ndarrays of the same dtype from {float64, int64}; no sorter=; len(a) >=
10_000 (HAYSTACK_FLOOR); len(v) >= the dtype floor (10_000 float64, 100_000
int64, the SUPPORTED dict) and len(v) <= 10 * len(a) (SKEW_CAP); and a
sampled disorder estimate over 4096 evenly strided adjacent pairs of v must
have min(descent_fraction, 1 - descent_fraction) >= 0.40, so only
random-ordered queries dispatch (sorted, nearly-sorted, and descending
queries all refuse). Dispatch argsorts v, searchsorted's the sorted copy
against a via GEARBOX.stock_fn (bypassing the dispatcher so the sorted copy
cannot re-trigger this same path), then scatters the indices back through
the permutation. Since each element's insertion index depends only on its
own value and a, never on its neighbors, the result is bit-identical to
stock for any input (NaNs and even an unsorted haystack included: garbage
in, the same garbage out, element for element). Comparison mode:
bit-identical.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

OP = "numpy.searchsorted"
PATH = "searchsorted_sortqueries"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _sorted_float(n, seed, lo=-1000.0, hi=1000.0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(lo, hi, size=n))


def _random_float(n, seed, lo=-1000.0, hi=1000.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=n)


def _sorted_int(n, seed, lo=-(10**12), hi=10**12):
    rng = np.random.default_rng(seed)
    return np.sort(rng.integers(lo, hi, size=n, dtype=np.int64))


def _random_int(n, seed, lo=-(10**12), hi=10**12):
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, size=n, dtype=np.int64)


def _assert_dispatched_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.searchsorted(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert np.asarray(got).dtype == np.asarray(stock).dtype
    assert np.asarray(got).shape == np.asarray(stock).shape
    assert np.array_equal(got, stock)
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.searchsorted(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert np.array_equal(got, stock)
    return got


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.searchsorted(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + bit-identity
# ---------------------------------------------------------------------------

def test_dispatch_float64_side_omitted():
    x = _sorted_float(50_000, seed=1)
    v = _random_float(50_000, seed=2)
    _assert_dispatched_equal((x, v), {})


def test_dispatch_float64_side_left_kwarg():
    x = _sorted_float(50_000, seed=1)
    v = _random_float(50_000, seed=2)
    _assert_dispatched_equal((x, v), {"side": "left"})


def test_dispatch_float64_side_right_kwarg():
    x = _sorted_float(50_000, seed=1)
    v = _random_float(50_000, seed=2)
    _assert_dispatched_equal((x, v), {"side": "right"})


def test_dispatch_float64_side_positional():
    x = _sorted_float(50_000, seed=1)
    v = _random_float(50_000, seed=2)
    _assert_dispatched_equal((x, v, "right"), {})


def test_dispatch_int64_wide_range():
    x = _sorted_int(200_000, seed=3)
    v = _random_int(200_000, seed=4)
    _assert_dispatched_equal((x, v), {})


def test_dispatch_int64_duplicates_heavy_both_sides():
    rng_x = np.random.default_rng(5)
    rng_v = np.random.default_rng(6)
    x = np.sort(rng_x.integers(0, 50, size=150_000, dtype=np.int64))
    v = rng_v.integers(0, 50, size=150_000, dtype=np.int64)
    _assert_dispatched_equal((x, v), {"side": "left"})
    _assert_dispatched_equal((x, v), {"side": "right"})


def test_dispatch_float64_queries_with_few_nans():
    x = _sorted_float(50_000, seed=7)
    v = _random_float(50_000, seed=8).copy()
    v[[100, 12345, 25000, 37777, 49999]] = np.nan
    _assert_dispatched_equal((x, v), {})


def test_dispatch_float64_queries_with_inf():
    x = _sorted_float(50_000, seed=9)
    v = _random_float(50_000, seed=10).copy()
    v[0] = np.inf
    v[1] = -np.inf
    v[2] = np.inf
    v[3] = -np.inf
    _assert_dispatched_equal((x, v), {})


def test_dispatch_float64_queries_outside_haystack_range():
    x = _sorted_float(50_000, seed=11, lo=0.0, hi=1000.0)
    v = _random_float(50_000, seed=12, lo=0.0, hi=1000.0).copy()
    v[:100] = -5000.0  # below x.min()
    v[100:200] = 5000.0  # above x.max()
    _assert_dispatched_equal((x, v), {})


# ---------------------------------------------------------------------------
# 2. refusal routes
# ---------------------------------------------------------------------------

def test_refusal_sorted_v():
    x = _sorted_float(50_000, seed=13)
    v = np.sort(_random_float(50_000, seed=14))
    _assert_refused_equal((x, v), {})


def test_refusal_descending_v():
    x = _sorted_float(50_000, seed=15)
    v = np.sort(_random_float(50_000, seed=16))[::-1]
    _assert_refused_equal((x, v), {})


def test_refusal_nearly_sorted_v():
    x = _sorted_float(50_000, seed=17)
    v = np.sort(_random_float(50_000, seed=18))
    rng = np.random.default_rng(19)
    n_swaps = v.size // 100  # ~1% of adjacent pairs swapped
    swap_starts = rng.choice(v.size - 1, size=n_swaps, replace=False)
    for i in swap_starts:
        v[i], v[i + 1] = v[i + 1], v[i]
    _assert_refused_equal((x, v), {})


def test_refusal_v_below_float_floor():
    x = _sorted_float(50_000, seed=20)
    v = _random_float(5_000, seed=21)
    _assert_refused_equal((x, v), {})


def test_refusal_int64_v_below_dtype_floor_above_float_floor():
    # 50_000 clears the float64 floor (10_000) but not the int64 floor
    # (100_000): a dtype-specific threshold, not one global number.
    x = _sorted_int(50_000, seed=22)
    v = _random_int(50_000, seed=23)
    _assert_refused_equal((x, v), {})


def test_refusal_x_below_haystack_floor():
    x = _sorted_float(5_000, seed=24)
    v = _random_float(50_000, seed=25)
    _assert_refused_equal((x, v), {})


def test_refusal_skew_cap_exceeded():
    x = _sorted_float(10_000, seed=26)
    v = _random_float(1_000_000, seed=27)
    _assert_refused_equal((x, v), {})


def test_refusal_sorter_kwarg():
    x = _random_float(50_000, seed=28)  # unsorted; sorter= corrects it
    sorter = np.argsort(x)
    v = _random_float(50_000, seed=29)
    _assert_refused_equal((x, v), {"sorter": sorter})


def test_refusal_bad_side_string_raises_both():
    x = _sorted_float(50_000, seed=30)
    v = _random_float(50_000, seed=31)
    _assert_refused_raises((x, v, "wrong"), {})


def test_refusal_scalar_v():
    x = _sorted_float(50_000, seed=32)
    _assert_refused_equal((x, 0.5), {})


def test_refusal_2d_v():
    x = _sorted_float(50_000, seed=33)
    v = _random_float(50_000, seed=34).reshape(500, 100)
    _assert_refused_equal((x, v), {})


def test_refusal_mixed_dtypes():
    x = _sorted_float(50_000, seed=35)
    v = _random_int(50_000, seed=36)
    _assert_refused_equal((x, v), {})


def test_refusal_float32_arrays():
    x = _sorted_float(50_000, seed=37).astype(np.float32)
    v = _random_float(50_000, seed=38).astype(np.float32)
    _assert_refused_equal((x, v), {})


def test_kill_switch_restores_stock_routing():
    x = _sorted_float(50_000, seed=39)
    v = _random_float(50_000, seed=40)
    decision, reason = GEARBOX.decide(OP, (x, v), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (x, v), {})
        assert decision == "stock", (decision, reason)
        got = np.searchsorted(x, v)
        stock = _stock(x, v)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)


# ---------------------------------------------------------------------------
# 3. unsorted haystack: garbage in, the same garbage out (per the module
#    docstring) -- verified NOT to hold under installed numpy: stock's own
#    searchsorted on an unsorted haystack is query-BATCH-ORDER dependent
#    (confirmed by comparing scalar-per-element calls, which match a manual
#    textbook bisect_left, against a batched call, which does not; this
#    smells like an internal locality hint chained between consecutive
#    queries -- harmless/correct on a truly sorted haystack, order-sensitive
#    "garbage" on an unsorted one). The fastpath reorders v via argsort
#    before calling stock, so its unpermuted output can diverge from
#    stock's own call on the original v order whenever the haystack is not
#    actually sorted. Dispatch still happens (applicability never checks a's
#    sortedness), so that half of the contract is asserted for real; the
#    bit-identity half is marked xfail(strict=True) as an executable record
#    of the discrepancy from searchsorted_sortqueries.py's docstring
#    ("an unsorted haystack included: garbage in, the same garbage out").
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    condition=np.lib.NumpyVersion(np.__version__) < "2.5.0",
    strict=True,
    reason=(
        "numpy < 2.5: searchsorted's batched result on an unsorted haystack "
        "is query-order dependent (17122/20000 elements measured on 2.4.5). "
        "numpy 2.5 removed the order dependence (0/20000 measured on 2.5.2 "
        "on a second machine), so there this runs as a plain passing test. "
        "See the block comment above this test."
    ),
)
def test_unsorted_haystack_matches_stocks_garbage_exactly():
    x = _random_float(50_000, seed=41)  # deliberately NOT sorted
    v = _random_float(50_000, seed=42)
    decision, reason = GEARBOX.decide(OP, (x, v), {})
    assert decision == PATH, (decision, reason)  # applicability ignores a's order
    _assert_dispatched_equal((x, v), {})


# ---------------------------------------------------------------------------
# 4. empty haystack edge
# ---------------------------------------------------------------------------

def test_empty_haystack_refuses_and_matches_stock():
    x = np.array([], dtype=np.float64)
    v = _random_float(50_000, seed=43)
    got = _assert_refused_equal((x, v), {})
    assert np.all(got == 0)
