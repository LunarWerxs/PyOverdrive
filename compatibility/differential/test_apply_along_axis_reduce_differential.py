"""Differential tests: apply_along_axis_reduce fast path vs stock
numpy.apply_along_axis.

Contract (src/pyoverdrive/fastpaths/apply_along_axis_reduce.py): serves
apply_along_axis(func1d, axis, arr) where func1d is one of numpy's own
reducers (_NAMES, matched by object IDENTITY against a table rebuilt on
every Gearbox generation change - not by name), arr is a plain
np.ndarray (no subclasses) of a listed dtype with no zero-length
dimension, axis is a plain python int valid for arr.ndim, and there are
at least SLICE_MIN 1-D slices along axis. Order-independent reducers
(_EXACT_ANY_AXIS) are served on any axis; order-sensitive reducers
(_ORDER_SENSITIVE_LAST_AXIS) are served only on the LAST axis (negative
spellings included), because floating-point accumulation order differs
off the last axis. Extra *args/**kwargs, non-ndarray subclasses,
zero-length dimensions, unlisted dtypes, and out-of-range/non-int axis
all refuse to stock. 1-D input returns a 0-d ndarray (stock's own return
type), never a bare NumPy scalar. Comparison mode: bit-identical.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.apply_along_axis_reduce import (
    _EXACT_ANY_AXIS,
    _NAMES,
    _ORDER_SENSITIVE_LAST_AXIS,
    _lookup,
    SLICE_MIN,
)

OP = "numpy.apply_along_axis"
PATH = "apply_along_axis_reduce"

_DTYPES = (np.float64, np.float32, np.int64, np.int32, np.uint64, np.bool_)

# every size below is derived from SLICE_MIN, never an independent literal
_AXIS_LEN = max(2, SLICE_MIN // 5)  # elements per reduced 1-D slice
_BELOW_MIN = SLICE_MIN - 1          # one short of the dispatch floor

_seed_counter = itertools.count(1000)


def _next_seed() -> int:
    return next(_seed_counter)


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _reducer(name):
    return getattr(np, name)


def _array(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    if dtype is np.bool_:
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 100, size=shape).astype(dtype)
    return rng.uniform(-10.0, 10.0, size=shape).astype(dtype)


def _assert_dispatched(func1d, axis, arr):
    decision, reason = GEARBOX.decide(OP, (func1d, axis, arr), {})
    assert decision == PATH, (decision, reason)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = np.apply_along_axis(func1d, axis, arr)
    assert not any(w.category is RuntimeWarning for w in caught), caught
    stock = _stock(func1d, axis, arr)
    assert type(got) is type(stock)
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.array_equal(got, stock, equal_nan=True)
    return got, stock


def _call_both(args, kwargs):
    def _call(fn):
        try:
            return ("ok", fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - parity capture, not handling
            return ("raised", exc)

    return _call(np.apply_along_axis), _call(_stock)


def _assert_refused(args, kwargs=None):
    kwargs = kwargs or {}
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    (got_kind, got_val), (stock_kind, stock_val) = _call_both(args, kwargs)
    assert got_kind == stock_kind, (got_kind, got_val, stock_kind, stock_val)
    assert type(got_val) is type(stock_val)
    if got_kind == "ok":
        assert np.array_equal(got_val, stock_val, equal_nan=True)


# ---------------------------------------------------------------------------
# 1. bit-identity, every served reducer, 2-D and 3-D, several dtypes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("name", _NAMES)
def test_bit_identical_last_axis_every_reducer(name, ndim, dtype):
    if name == "ptp" and dtype is np.bool_:
        pytest.skip("np.ptp subtracts min from max; unsupported for bool on stock too")
    shape = (SLICE_MIN, _AXIS_LEN) if ndim == 2 else (SLICE_MIN, _AXIS_LEN, _AXIS_LEN)
    arr = _array(shape, dtype, seed=_next_seed())
    _assert_dispatched(_reducer(name), -1, arr)


# ---------------------------------------------------------------------------
# 2. the order-sensitive split, proven both ways
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", _EXACT_ANY_AXIS)
def test_order_independent_dispatches_non_last_axis_2d(name):
    arr = _array((_AXIS_LEN, SLICE_MIN), np.float64, seed=_next_seed())
    _assert_dispatched(_reducer(name), 0, arr)


@pytest.mark.parametrize("name", _EXACT_ANY_AXIS)
def test_order_independent_dispatches_middle_axis_3d(name):
    arr = _array((SLICE_MIN, _AXIS_LEN, SLICE_MIN), np.float64, seed=_next_seed())
    _assert_dispatched(_reducer(name), 1, arr)


@pytest.mark.parametrize("name", _ORDER_SENSITIVE_LAST_AXIS)
def test_order_sensitive_refuses_non_last_axis_2d(name):
    arr = _array((_AXIS_LEN, SLICE_MIN), np.float64, seed=_next_seed())
    _assert_refused((_reducer(name), 0, arr), {})


@pytest.mark.parametrize("name", _ORDER_SENSITIVE_LAST_AXIS)
def test_order_sensitive_refuses_middle_axis_3d(name):
    arr = _array((SLICE_MIN, _AXIS_LEN, SLICE_MIN), np.float64, seed=_next_seed())
    _assert_refused((_reducer(name), 1, arr), {})


@pytest.mark.parametrize("name", _ORDER_SENSITIVE_LAST_AXIS)
def test_order_sensitive_dispatches_last_axis_positive_and_negative(name):
    arr = _array((SLICE_MIN, _AXIS_LEN), np.float64, seed=_next_seed())
    got_pos, stock_pos = _assert_dispatched(_reducer(name), 1, arr)
    got_neg, stock_neg = _assert_dispatched(_reducer(name), -1, arr)
    assert np.array_equal(got_pos, got_neg, equal_nan=True)
    assert np.array_equal(stock_pos, stock_neg, equal_nan=True)


# ---------------------------------------------------------------------------
# 3. return-type parity for 1-D input
#
# A 1-D array has exactly ONE 1-D slice along its only axis (slices =
# prod(shape) // shape[axis] == 1 for any N), which is always below
# SLICE_MIN - so 1-D input structurally can never clear the dispatch
# floor and always refuses. That refusal is itself the return-type-parity
# proof: the wrapper delegates to stock, and stock's own 0-d-ndarray (not
# bare-scalar) return type is what both routes therefore share.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["max", "sum"])
def test_return_type_parity_1d_input_always_refuses_and_matches_stock(name):
    arr = _array((SLICE_MIN,), np.float64, seed=_next_seed())
    decision, reason = GEARBOX.decide(OP, (_reducer(name), 0, arr), {})
    assert decision == "stock", (decision, reason)
    _assert_refused((_reducer(name), 0, arr), {})
    stock = _stock(_reducer(name), 0, arr)
    assert type(stock) is np.ndarray and stock.shape == ()


# ---------------------------------------------------------------------------
# 4. refusals - each isolates exactly one disqualifying condition
# ---------------------------------------------------------------------------

def _ok_shape_arr(dtype=np.float64, seed=None):
    return _array((SLICE_MIN, _AXIS_LEN), dtype, seed=seed or _next_seed())


def test_refusal_matrix_subclass():
    arr = np.matrix(_ok_shape_arr())
    _assert_refused((_reducer("sum"), -1, arr), {})


def test_refusal_masked_array():
    arr = np.ma.array(_ok_shape_arr(), mask=False)
    _assert_refused((_reducer("sum"), -1, arr), {})


def test_refusal_zero_length_non_axis_dimension():
    arr = np.zeros((0, _AXIS_LEN), dtype=np.float64)
    _assert_refused((_reducer("sum"), -1, arr), {})


def test_refusal_zero_length_axis_dimension():
    arr = np.zeros((SLICE_MIN, 0), dtype=np.float64)
    _assert_refused((_reducer("sum"), -1, arr), {})


def test_refusal_object_dtype():
    arr = _ok_shape_arr(np.int64).astype(object)
    _assert_refused((_reducer("sum"), -1, arr), {})


def test_refusal_complex_dtype():
    arr = _ok_shape_arr().astype(np.complex128)
    _assert_refused((_reducer("sum"), -1, arr), {})


def test_refusal_unmatched_user_function_named_mean():
    def mean(a):
        return float(np.sum(a))

    arr = _ok_shape_arr()
    _assert_refused((mean, -1, arr), {})


def test_refusal_extra_positional_args():
    arr = _ok_shape_arr()
    _assert_refused((_reducer("sum"), -1, arr, None), {})


def test_refusal_any_kwargs():
    arr = _ok_shape_arr()
    _assert_refused((_reducer("sum"), -1, arr), {"dtype": None})


def test_refusal_slice_count_below_min():
    arr = _array((_BELOW_MIN, _AXIS_LEN), np.float64, seed=_next_seed())
    _assert_refused((_reducer("sum"), -1, arr), {})


def test_refusal_axis_out_of_range():
    arr = _ok_shape_arr()
    _assert_refused((_reducer("sum"), arr.ndim, arr), {})


def test_refusal_non_int_axis():
    arr = _ok_shape_arr()
    _assert_refused((_reducer("sum"), np.int64(-1), arr), {})


# ---------------------------------------------------------------------------
# 5. kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_restores_stock_routing():
    arr = _ok_shape_arr()
    decision, reason = GEARBOX.decide(OP, (np.sum, -1, arr), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (np.sum, -1, arr), {})
        assert decision == "stock", (decision, reason)
        got = np.apply_along_axis(np.sum, -1, arr)
        stock = _stock(np.sum, -1, arr)
        assert np.array_equal(got, stock, equal_nan=True)
    finally:
        pyoverdrive.enable_path(PATH)


# ---------------------------------------------------------------------------
# 6. patched-identity: the live np.mean must be recognized, not just stock
# ---------------------------------------------------------------------------

def test_live_np_mean_dispatches_via_lookup_identity():
    arr = _ok_shape_arr()
    assert id(np.mean) in _lookup()
    decision, reason = GEARBOX.decide(OP, (np.mean, -1, arr), {})
    assert decision == PATH, (decision, reason)
    _assert_dispatched(np.mean, -1, arr)
