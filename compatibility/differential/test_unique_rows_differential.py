"""Differential tests: unique_rows_lexsort fast path vs stock numpy.unique.

Contract (src/pyoverdrive/fastpaths/unique_rows_lexsort.py): applies only to
np.unique(a, axis=0) where a is a plain 2-D int64/int32 ndarray, K_MIN <= a.shape[1]
<= K_MAX, a.shape[0] >= ROWS_MIN, axis given by keyword or as the fifth
positional argument, and the only other argument is return_counts.
return_index/return_inverse/equal_nan are refused. axis must be exactly 0.
Single-column 2-D belongs to unique_axis0_column; 1-D belongs to unique_sort
(or stock, since unique_sort requires an empty kwargs dict). Comparison mode:
bit-identical.

numpy.unique carries four OTHER registered paths besides this one
(unique_sort, unique_char_view, unique_axis0_column, plus stock itself), so
every dispatch assertion below pins the exact path name, not just "not
stock".
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.unique_rows_lexsort import K_MAX, K_MIN, ROWS_MIN

UNIQUE_OP = "numpy.unique"
PATH = "unique_rows_lexsort"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([UNIQUE_OP])
    yield
    pyoverdrive.disable()


def _stock():
    return GEARBOX.stock_fn(UNIQUE_OP)


def _rand_rows(rows, k, seed, dtype=np.int64, low=-1_000, high=1_000):
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=(rows, k), dtype=dtype)


def _assert_arrays_equal(got, stock):
    assert type(got) is type(stock)
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.array_equal(got, stock)


def _assert_result_equal(got, stock, input_dtype):
    """Compare a unique result: a lone array, or (values, counts). Every
    member is checked bit-identical for dtype, shape and values; the values
    array must keep input_dtype."""
    if isinstance(got, tuple):
        assert isinstance(stock, tuple)
        assert len(got) == len(stock)
        for i, (g, s) in enumerate(zip(got, stock)):
            _assert_arrays_equal(g, s)
            if i == 0:
                assert g.dtype == input_dtype
    else:
        assert not isinstance(stock, tuple)
        _assert_arrays_equal(got, stock)
        assert got.dtype == input_dtype


def _assert_dispatched_equal(args, kwargs):
    decision, reason = GEARBOX.decide(UNIQUE_OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.unique(*args, **kwargs)
    stock = _stock()(*args, **kwargs)
    input_dtype = args[0].dtype
    _assert_result_equal(got, stock, input_dtype)
    return got, stock


def _assert_refused(args, kwargs, expected_decision=None):
    """Assert Gearbox refuses this path (routes elsewhere or to stock), then
    prove parity with real stock numpy either way, exceptions included."""
    decision, reason = GEARBOX.decide(UNIQUE_OP, args, kwargs)
    assert decision != PATH, (args, kwargs, decision, reason)
    if expected_decision is not None:
        assert decision == expected_decision, (decision, reason)
    got_exc = stock_exc = None
    try:
        got = np.unique(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        got_exc = exc
    try:
        stock = _stock()(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        stock_exc = exc
    if got_exc is not None or stock_exc is not None:
        assert type(got_exc) is type(stock_exc), (got_exc, stock_exc)
        return None
    input_dtype = args[0].dtype if hasattr(args[0], "dtype") else None
    if isinstance(got, tuple):
        assert isinstance(stock, tuple)
        assert len(got) == len(stock)
        for i, (g, s) in enumerate(zip(got, stock)):
            _assert_arrays_equal(g, s)
            if i == 0 and input_dtype is not None:
                assert g.dtype == input_dtype
    else:
        _assert_arrays_equal(got, stock)
        if input_dtype is not None:
            assert got.dtype == input_dtype
    return got


# ---------------------------------------------------------------------------
# dispatch + bit-identity
# ---------------------------------------------------------------------------

def test_dispatch_int64_rows2000_k2_negative_salted():
    # Order-semantics witness (see the fast path's own docstring): stock
    # returns rows in NUMERIC lexicographic order, which little-endian void
    # memcmp would NOT reproduce for negative ints. Wide negative/positive
    # range so any order divergence would show up as a non-match, not
    # merely as a set-equality pass.
    a = _rand_rows(2000, 2, seed=100, low=-(10**9), high=10**9)
    got, stock = _assert_dispatched_equal((a,), {"axis": 0})
    # Explicit row-order witness, not just array_equal above: every row of
    # got must appear at the SAME index as in stock.
    assert np.array_equal(got, stock), "row order diverges from stock numeric lexicographic order"


def test_dispatch_int32_k3():
    a = _rand_rows(3000, 3, seed=101, dtype=np.int32, low=-50_000, high=50_000)
    _assert_dispatched_equal((a,), {"axis": 0})


def test_dispatch_k_max_boundary():
    a = _rand_rows(ROWS_MIN, K_MAX, seed=102)
    assert a.shape[1] == K_MAX
    _assert_dispatched_equal((a,), {"axis": 0})


def test_refusal_k_max_plus_one():
    a = _rand_rows(ROWS_MIN, K_MAX + 1, seed=103)
    _assert_refused((a,), {"axis": 0}, expected_decision="stock")


def test_dispatch_rows_min_boundary():
    a = _rand_rows(ROWS_MIN, K_MIN, seed=104)
    assert a.shape[0] == ROWS_MIN
    _assert_dispatched_equal((a,), {"axis": 0})


def test_refusal_rows_min_minus_one():
    a = _rand_rows(ROWS_MIN - 1, K_MIN, seed=105)
    _assert_refused((a,), {"axis": 0}, expected_decision="stock")


def test_dispatch_low_cardinality_many_duplicates():
    a = _rand_rows(5000, 2, seed=106, low=0, high=5)
    _assert_dispatched_equal((a,), {"axis": 0})


def test_dispatch_all_identical_rows():
    a = np.tile(np.array([[7, -3, 42]], dtype=np.int64), (ROWS_MIN, 1))
    assert a.shape == (ROWS_MIN, 3)
    got, stock = _assert_dispatched_equal((a,), {"axis": 0})
    assert got.shape == (1, 3)


def test_dispatch_return_counts_true():
    a = _rand_rows(4000, 2, seed=107, low=0, high=50)
    got, stock = _assert_dispatched_equal((a,), {"axis": 0, "return_counts": True})
    assert isinstance(got, tuple) and len(got) == 2


def test_axis_as_fifth_positional_argument():
    # _normalize accepts (ar, return_index, return_inverse, return_counts,
    # axis) positionally; confirm what it actually does and assert parity
    # either way.
    a = _rand_rows(2000, 2, seed=108)
    args = (a, False, False, False, 0)
    decision, reason = GEARBOX.decide(UNIQUE_OP, args, {})
    assert decision == PATH, (decision, reason, "positional axis form should be accepted per _normalize")
    _assert_dispatched_equal(args, {})


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_refusal_return_index_true():
    a = _rand_rows(2000, 2, seed=109)
    _assert_refused((a,), {"axis": 0, "return_index": True}, expected_decision="stock")


def test_refusal_return_inverse_true():
    a = _rand_rows(2000, 2, seed=110)
    _assert_refused((a,), {"axis": 0, "return_inverse": True}, expected_decision="stock")


def test_refusal_axis_1():
    a = _rand_rows(2000, 2, seed=111)
    _assert_refused((a,), {"axis": 1}, expected_decision="stock")


def test_refusal_axis_none_flattens():
    # Flattening changes semantics entirely (1-D result). Neither this path
    # (requires axis == 0) nor unique_sort (requires an empty kwargs dict)
    # accept it, so this is expected to land on stock; assert NOT this path
    # and prove parity regardless of where it actually lands.
    a = _rand_rows(2000, 2, seed=112)
    _assert_refused((a,), {"axis": None})


def test_refusal_float64_rows():
    rng = np.random.default_rng(113)
    a = rng.random((2000, 2))
    _assert_refused((a,), {"axis": 0}, expected_decision="stock")


def test_refusal_uint64_rows():
    a = _rand_rows(2000, 2, seed=114, dtype=np.uint64, low=0, high=1000)
    _assert_refused((a,), {"axis": 0}, expected_decision="stock")


def test_refusal_1d_array():
    rng = np.random.default_rng(115)
    a = rng.integers(-1000, 1000, size=2000, dtype=np.int64)
    assert a.ndim == 1
    _assert_refused((a,), {"axis": 0}, expected_decision="stock")


def test_refusal_single_column_routes_to_axis0_column():
    a = _rand_rows(2000, 1, seed=116)
    assert a.shape[1] == 1
    _assert_refused((a,), {"axis": 0}, expected_decision="unique_axis0_column")


def test_refusal_equal_nan_false_kwarg():
    a = _rand_rows(2000, 2, seed=117)
    _assert_refused((a,), {"axis": 0, "equal_nan": False}, expected_decision="stock")


def test_refusal_python_nested_list():
    rows = [[i % 7, (i * 3) % 11] for i in range(2000)]
    _assert_refused((rows,), {"axis": 0}, expected_decision="stock")


def test_kill_switch_restores_stock_routing():
    a = _rand_rows(2000, 2, seed=118)
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {"axis": 0})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {"axis": 0})
        assert decision != PATH, (decision, reason)
        got = np.unique(a, axis=0)
        stock = _stock()(a, axis=0)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)
