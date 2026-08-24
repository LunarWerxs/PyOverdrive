"""Differential tests: roll_concat_1d fast path vs stock numpy.roll.

Contract (src/pyoverdrive/fastpaths/roll_concat.py): applies only to
roll(a, shift[, axis]) where axis is absent or None, a is a plain 1-D
ndarray of a measured dtype (int64/float64/int32/float32/bool), with
1 <= a.size <= SIZE_CAP, and shift is a Python/numpy integer (bool
excluded). Non-contiguity of a is explicitly allowed by the predicate.
shift is normalized modulo n exactly as stock does; shift % n == 0
returns a fresh copy (np.roll never returns the input, aliased or not).
Comparison mode: bit-identical, dtype and contiguity included.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.roll_concat import SIZE_CAP

OP = "numpy.roll"
PATH = "roll_concat_1d"

_SUPPORTED_DTYPES = [np.int64, np.float64, np.int32, np.float32, np.bool_]


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _arr(n, dtype, seed=1):
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=n).astype(np.bool_)
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return (rng.random(n) * 1000 - 500).astype(dtype)
    return rng.integers(-1000, 1000, size=n).astype(dtype)


def _assert_flags_equal(got, stock):
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert got.flags["C_CONTIGUOUS"] == stock.flags["C_CONTIGUOUS"]
    assert got.flags["F_CONTIGUOUS"] == stock.flags["F_CONTIGUOUS"]


def _assert_dispatched_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.roll(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    _assert_flags_equal(got, stock)
    assert np.array_equal(got, stock)
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.roll(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert np.array_equal(got, stock)
    return got


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.roll(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + bit-identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_dispatch_every_supported_dtype_n500_shift7(dtype):
    a = _arr(500, dtype, seed=1)
    got, stock = _assert_dispatched_equal((a, 7), {})
    assert got.dtype == np.dtype(dtype)


def test_dispatch_shift_zero_returns_fresh_copy():
    a = _arr(200, np.int64, seed=2)
    got, stock = _assert_dispatched_equal((a, 0), {})
    assert got is not a
    original = a.copy()
    got[0] = got[0] + 1 if got.dtype != np.bool_ else not got[0]
    assert np.array_equal(a, original)


def test_dispatch_negative_shift_minus_one():
    a = _arr(500, np.int64, seed=3)
    _assert_dispatched_equal((a, -1), {})


def test_dispatch_negative_shift_minus_n_plus_3():
    n = 500
    a = _arr(n, np.int64, seed=4)
    _assert_dispatched_equal((a, -(n + 3)), {})


def test_dispatch_shift_larger_than_n_wraps():
    n = 500
    a = _arr(n, np.float64, seed=5)
    _assert_dispatched_equal((a, n + 37), {})


def test_dispatch_shift_exactly_n_is_zero_mod_and_copies():
    n = 500
    a = _arr(n, np.float64, seed=6)
    got, stock = _assert_dispatched_equal((a, n), {})
    assert got is not a


def test_dispatch_numpy_int32_scalar_shift():
    a = _arr(500, np.int64, seed=7)
    _assert_dispatched_equal((a, np.int32(11)), {})


def test_dispatch_numpy_int64_scalar_shift():
    a = _arr(500, np.int64, seed=8)
    _assert_dispatched_equal((a, np.int64(11)), {})


def test_dispatch_shift_as_keyword():
    a = _arr(500, np.int64, seed=9)
    _assert_dispatched_equal((a,), {"shift": 5})


def test_dispatch_size_one_array():
    a = _arr(1, np.int64, seed=10)
    got, stock = _assert_dispatched_equal((a, 3), {})
    assert got is not a


def test_dispatch_size_equals_size_cap():
    a = _arr(SIZE_CAP, np.int64, seed=11)
    _assert_dispatched_equal((a, 17), {})


def test_refusal_size_cap_plus_one():
    a = _arr(SIZE_CAP + 1, np.int64, seed=12)
    decision, reason = GEARBOX.decide(OP, (a, 17), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((a, 17), {})


def test_dispatch_noncontiguous_1d_view_matches_stock_exactly():
    big = _arr(40, np.int64, seed=13)
    a = big[::2]
    assert a.flags["C_CONTIGUOUS"] is False
    assert type(a) is np.ndarray and a.ndim == 1
    decision, reason = GEARBOX.decide(OP, (a, 3), {})
    assert decision == PATH, (decision, reason)
    got, stock = _assert_dispatched_equal((a, 3), {})
    assert got.flags["C_CONTIGUOUS"] is True
    assert stock.flags["C_CONTIGUOUS"] is True


# ---------------------------------------------------------------------------
# 2. refusal routes
# ---------------------------------------------------------------------------


def test_refusal_axis_0_kwarg():
    a = _arr(200, np.int64, seed=14)
    _assert_refused_equal((a, 3), {"axis": 0})


def test_refusal_axis_tuple():
    a = _arr(200, np.int64, seed=15)
    _assert_refused_equal((a, 3), {"axis": (0,)})


def test_refusal_tuple_shift():
    a = _arr(200, np.int64, seed=16)
    _assert_refused_equal((a, (1,)), {})


def test_refusal_bool_shift_true():
    a = _arr(200, np.int64, seed=17)
    _assert_refused_equal((a, True), {})


def test_refusal_float_shift():
    # Stock's float-shift behavior is VERSION-DEPENDENT: numpy 2.4/2.5
    # accepts it (truncated internally) while 2.0.2 raises TypeError from
    # its slice arithmetic (caught by the public repo's oldest-numpy CI
    # leg). Either way the call must refuse to the path and mirror stock:
    # same values when stock returns, same exception type when it raises.
    a = _arr(200, np.int64, seed=18)
    decision, reason = GEARBOX.decide(OP, (a, 1.0), {})
    assert decision == "stock", (decision, reason)
    try:
        stock = _stock(a, 1.0)
    except Exception as stock_exc:  # noqa: BLE001 - parity capture
        with pytest.raises(type(stock_exc)):
            np.roll(a, 1.0)
    else:
        got = np.roll(a, 1.0)
        assert type(got) is type(stock)
        assert np.array_equal(got, stock)


def test_refusal_2d_input():
    a = _arr(200, np.int64, seed=19).reshape(20, 10)
    _assert_refused_equal((a, 3), {})


def test_refusal_size_zero_empty_array():
    a = np.array([], dtype=np.int64)
    assert a.size == 0
    decision, reason = GEARBOX.decide(OP, (a, 1), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((a, 1), {})


def test_refusal_uint8_dtype_unmeasured():
    a = _arr(200, np.uint8, seed=20)
    _assert_refused_equal((a, 3), {})


def test_refusal_complex128_dtype():
    a = (_arr(200, np.float64, seed=21) + 1j * _arr(200, np.float64, seed=22)).astype(
        np.complex128
    )
    _assert_refused_equal((a, 3), {})


def test_refusal_python_list_input():
    a = [1, 2, 3, 4, 5]
    decision, reason = GEARBOX.decide(OP, (a, 2), {})
    assert decision == "stock", (decision, reason)
    got = np.roll(a, 2)
    stock = _stock(a, 2)
    assert np.array_equal(got, stock)


def test_refusal_missing_shift_raises_same_exception():
    a = _arr(200, np.int64, seed=23)
    _assert_refused_raises((a,), {})


# ---------------------------------------------------------------------------
# 3. kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_restores_stock_routing():
    a = _arr(500, np.int64, seed=24)
    decision, reason = GEARBOX.decide(OP, (a, 7), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (a, 7), {})
        assert decision == "stock", (decision, reason)
        got = np.roll(a, 7)
        stock = _stock(a, 7)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)
