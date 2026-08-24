"""Differential tests: argmax_blocked_transpose fast path vs stock numpy.argmax.

Contract (src/pyoverdrive/fastpaths/argmax_blocked.py): applies only to
argmax(a, axis) where a is a plain C-contiguous 2-D float64/float32/int64
ndarray, axis is 0 or -2, no other kwargs, a.shape[0] >= ROWS_MIN and
a.size >= SIZE_MIN. Dispatch does a cache-blocked transpose (an exact
permutation) then argmax along the now-fast axis, so first-occurrence ties
and first-NaN semantics are preserved exactly. Comparison mode:
bit-identical (intp indices).

This path is CALIBRATION-GATED (it registers disabled; only a per-machine
probe enables it - see src/pyoverdrive/calibration.py). The fixture below
force-enables it so the differential contract is verified on every
machine regardless of whether calibration would turn it on here.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.argmax_blocked import ROWS_MIN, SIZE_MIN

OP = "numpy.argmax"
PATH = "argmax_blocked_transpose"

ROWS = ROWS_MIN
COLS = SIZE_MIN // ROWS_MIN
assert ROWS * COLS >= SIZE_MIN


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable_path(PATH)  # calibration-gated: off unless forced
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()
    pyoverdrive.disable_path(PATH)


def test_gated_off_by_default_without_calibration():
    # the registered default is disabled; only calibration.apply() or an
    # explicit enable_path may turn it on (this fixture did, so check the
    # registry's stored default via a fresh FastPath lookup is not
    # possible here - instead verify the toggle round-trip is honored)
    pyoverdrive.disable_path(PATH)
    try:
        a = _base_array()
        decision, reason = GEARBOX.decide(OP, (a,), {"axis": 0})
        assert decision == "stock", (decision, reason)
    finally:
        pyoverdrive.enable_path(PATH)


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


_BASE_CACHE = {}


def _base_array():
    # module-level cache: one (ROWS, COLS) float64 random array reused by
    # every test that just needs "some" dispatching-size content.
    arr = _BASE_CACHE.get("base")
    if arr is None:
        rng = np.random.default_rng(100)
        arr = rng.random((ROWS, COLS))
        _BASE_CACHE["base"] = arr
    return arr


def _assert_dispatched_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.argmax(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert got.dtype == np.intp
    assert stock.dtype == np.intp
    assert got.shape == stock.shape
    assert np.array_equal(got, stock)
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (decision, reason)
    got = np.argmax(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.array_equal(got, stock)
    return got


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.argmax(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + bit-identity
# ---------------------------------------------------------------------------


def test_dispatch_float64_exact_minimum_shape():
    a = _base_array()
    assert a.shape == (ROWS_MIN, SIZE_MIN // ROWS_MIN)
    assert a.dtype == np.float64 and a.flags.c_contiguous
    _assert_dispatched_equal((a,), {"axis": 0})


def test_dispatch_axis_negative_2():
    a = _base_array()
    _assert_dispatched_equal((a,), {"axis": -2})


def test_dispatch_positional_axis():
    a = _base_array()
    _assert_dispatched_equal((a, 0), {})


def test_dispatch_float32():
    rng = np.random.default_rng(101)
    a = rng.random((ROWS, COLS)).astype(np.float32)
    assert a.flags.c_contiguous
    _assert_dispatched_equal((a,), {"axis": 0})


def test_dispatch_int64():
    rng = np.random.default_rng(102)
    a = rng.integers(-1_000_000, 1_000_000, size=(ROWS, COLS), dtype=np.int64)
    assert a.flags.c_contiguous
    _assert_dispatched_equal((a,), {"axis": 0})


def test_dispatch_tie_heavy_first_occurrence():
    # only 3 distinct values per column -> heavy duplication of the column
    # maximum; first-occurrence-wins must match stock exactly.
    rng = np.random.default_rng(103)
    a = rng.integers(0, 3, size=(ROWS, COLS)).astype(np.float64)
    got, stock = _assert_dispatched_equal((a,), {"axis": 0})
    # sanity: ties really do occur (more than one row equals the column max
    # for at least some columns), otherwise this test proves nothing.
    col_max = a.max(axis=0)
    tie_counts = (a == col_max).sum(axis=0)
    assert (tie_counts > 1).any()


def test_dispatch_nan_salted_first_nan_wins():
    a = _base_array().copy()
    # some columns get exactly one NaN, others get two NaNs at different
    # rows; stock argmax returns the index of the FIRST NaN encountered.
    single_nan_cols = range(0, COLS, 500)
    for j in single_nan_cols:
        a[10, j] = np.nan
    double_nan_cols = range(1, COLS, 700)
    for j in double_nan_cols:
        a[20, j] = np.nan
        a[5, j] = np.nan  # earlier row -> this is the "first" NaN
    got, stock = _assert_dispatched_equal((a,), {"axis": 0})
    for j in single_nan_cols:
        assert got[j] == 10
    for j in double_nan_cols:
        assert got[j] == 5


def test_dispatch_all_equal_column_argmax_is_zero():
    col_vals = np.random.default_rng(104).random(COLS)
    a = np.tile(col_vals, (ROWS, 1))
    assert a.shape == (ROWS, COLS)
    got, stock = _assert_dispatched_equal((a,), {"axis": 0})
    assert np.all(got == 0)


def test_refusal_boundary_rows_one_under_minimum():
    # rows just below ROWS_MIN, cols raised so size still clears SIZE_MIN.
    rows = ROWS_MIN - 1
    cols = -(-SIZE_MIN // rows)  # ceil division
    assert rows * cols >= SIZE_MIN
    assert rows < ROWS_MIN
    rng = np.random.default_rng(105)
    a = rng.random((rows, cols)).astype(np.float32)
    decision, reason = GEARBOX.decide(OP, (a,), {"axis": 0})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((a,), {"axis": 0})


def test_refusal_boundary_size_one_under_minimum():
    # rows clears ROWS_MIN but total size is just under SIZE_MIN.
    rows = ROWS_MIN
    cols = (SIZE_MIN // rows) - 1
    assert rows >= ROWS_MIN
    assert rows * cols < SIZE_MIN
    rng = np.random.default_rng(106)
    a = rng.random((rows, cols)).astype(np.float32)
    decision, reason = GEARBOX.decide(OP, (a,), {"axis": 0})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((a,), {"axis": 0})


# ---------------------------------------------------------------------------
# 2. refusal routes
# ---------------------------------------------------------------------------


def test_refusal_axis_1_already_fast_axis():
    a = _base_array()
    _assert_refused_equal((a,), {"axis": 1})


def test_refusal_axis_none_flat_argmax():
    a = _base_array()
    _assert_refused_equal((a,), {"axis": None})


def test_refusal_f_order_array_dispatching_size():
    a = np.asfortranarray(_base_array())
    assert a.flags.f_contiguous and not a.flags.c_contiguous
    assert a.shape[0] >= ROWS_MIN and a.size >= SIZE_MIN
    _assert_refused_equal((a,), {"axis": 0})


def test_refusal_float16_dispatching_size():
    rng = np.random.default_rng(107)
    a = rng.random((ROWS, COLS)).astype(np.float16)
    _assert_refused_equal((a,), {"axis": 0})


def test_refusal_3d_array():
    rng = np.random.default_rng(108)
    a = rng.random((10, 10, 10))
    _assert_refused_equal((a,), {"axis": 0})


def test_refusal_1d_array():
    rng = np.random.default_rng(109)
    a = rng.random(500)
    _assert_refused_equal((a,), {"axis": 0})


def test_refusal_keepdims_true():
    a = _base_array()
    _assert_refused_equal((a,), {"axis": 0, "keepdims": True})


def test_refusal_out_kwarg():
    a = _base_array()
    out_got = np.empty(COLS, dtype=np.intp)
    out_stock = np.empty(COLS, dtype=np.intp)
    decision, reason = GEARBOX.decide(OP, (a,), {"axis": 0, "out": out_got})
    assert decision == "stock", (decision, reason)
    got = np.argmax(a, axis=0, out=out_got)
    stock = _stock(a, axis=0, out=out_stock)
    assert got is out_got
    assert np.array_equal(got, stock)
    assert got.dtype == np.intp


def test_refusal_python_nested_list():
    a = [[1.0, 2.0, 3.0], [4.0, 9.0, 6.0], [7.0, 8.0, 0.0]]
    decision, reason = GEARBOX.decide(OP, (a,), {"axis": 0})
    assert decision == "stock", (decision, reason)
    got = np.argmax(a, axis=0)
    stock = _stock(a, axis=0)
    assert np.array_equal(got, stock)


def test_refusal_axis_5_raises_same_exception():
    a = _base_array()
    _assert_refused_raises((a,), {"axis": 5})


def test_kill_switch_restores_stock_routing():
    a = _base_array()
    decision, reason = GEARBOX.decide(OP, (a,), {"axis": 0})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (a,), {"axis": 0})
        assert decision == "stock", (decision, reason)
        got = np.argmax(a, axis=0)
        stock = _stock(a, axis=0)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)
