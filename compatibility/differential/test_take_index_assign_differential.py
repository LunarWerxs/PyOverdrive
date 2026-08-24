"""Differential tests: take_index_assign fast path vs stock numpy.take.

Covers take(a, indices, out=out) via fancy-index gather plus assignment,
a plain 1-D float64/int64 (SIZE_MIN=10_000 provisional), against stock
np.take(a, indices, out=out). Comparison mode is bit-identical, and the
result must be the exact `out` object stock returns. Out-of-bounds
indices are the StockRaised route: the predicate still accepts (decide()
still names the path), the gather raises inside the run, and the path
reruns stock so the caller sees stock's own IndexError with no
RuntimeWarning. Refusals are checked against stock exactly.
"""

import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.take_index_assign import SIZE_MIN

OP = "numpy.take"
PATH = "take_index_assign"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock():
    return GEARBOX.stock_fn(OP)


def _call(fn, args, kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as e:  # noqa: BLE001 - symmetric probe, any exception type
        return ("err", type(e))


def _assert_refused(args, kwargs=None):
    kwargs = kwargs or {}
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (decision, reason)
    got_tag, got = _call(np.take, args, kwargs)
    stock_tag, stock = _call(_stock(), args, kwargs)
    assert got_tag == stock_tag, (got_tag, got, stock_tag, stock)
    if got_tag == "err":
        assert got is stock, (got, stock)
    else:
        assert np.array_equal(got, stock, equal_nan=True)


# --- 1. bit-identity + same-object-return, f64/int64, +/- indices --------


def test_dispatch_f64_positive_indices_returns_same_out_object():
    a = np.arange(SIZE_MIN + 500, dtype=np.float64) * 1.5
    indices = np.arange(SIZE_MIN, dtype=np.intp) % a.size
    out = np.empty(indices.shape, dtype=np.float64)
    decision, reason = GEARBOX.decide(OP, (a, indices), {"out": out})
    assert decision == PATH, (decision, reason)
    got = np.take(a, indices, out=out)
    assert got is out
    stock_out = np.empty(indices.shape, dtype=np.float64)
    stock = _stock()(a, indices, out=stock_out)
    assert stock is stock_out
    assert np.array_equal(got, stock)


def test_dispatch_int64_positive_indices_returns_same_out_object():
    a = np.arange(SIZE_MIN + 500, dtype=np.int64) * 3
    indices = np.arange(SIZE_MIN, dtype=np.intp) % a.size
    out = np.empty(indices.shape, dtype=np.int64)
    decision, reason = GEARBOX.decide(OP, (a, indices), {"out": out})
    assert decision == PATH, (decision, reason)
    got = np.take(a, indices, out=out)
    assert got is out
    stock_out = np.empty(indices.shape, dtype=np.int64)
    stock = _stock()(a, indices, out=stock_out)
    assert stock is stock_out
    assert np.array_equal(got, stock)


def test_dispatch_f64_negative_indices_wrap_like_stock():
    a = np.arange(SIZE_MIN + 500, dtype=np.float64) * 1.5
    rng = np.random.default_rng(1)
    indices = rng.integers(-a.size, a.size, size=SIZE_MIN).astype(np.intp)
    out = np.empty(indices.shape, dtype=np.float64)
    decision, reason = GEARBOX.decide(OP, (a, indices), {"out": out})
    assert decision == PATH, (decision, reason)
    got = np.take(a, indices, out=out)
    assert got is out
    stock_out = np.empty(indices.shape, dtype=np.float64)
    stock = _stock()(a, indices, out=stock_out)
    assert np.array_equal(got, stock)


def test_dispatch_int64_negative_indices_wrap_like_stock():
    a = np.arange(SIZE_MIN + 500, dtype=np.int64) * 3
    rng = np.random.default_rng(2)
    indices = rng.integers(-a.size, a.size, size=SIZE_MIN).astype(np.intp)
    out = np.empty(indices.shape, dtype=np.int64)
    decision, reason = GEARBOX.decide(OP, (a, indices), {"out": out})
    assert decision == PATH, (decision, reason)
    got = np.take(a, indices, out=out)
    assert got is out
    stock_out = np.empty(indices.shape, dtype=np.int64)
    stock = _stock()(a, indices, out=stock_out)
    assert np.array_equal(got, stock)


def test_dispatch_at_exact_floor():
    a = np.arange(SIZE_MIN + 10, dtype=np.float64)
    indices = np.arange(SIZE_MIN, dtype=np.intp)
    out = np.empty(indices.shape, dtype=np.float64)
    decision, reason = GEARBOX.decide(OP, (a, indices), {"out": out})
    assert decision == PATH, (decision, reason)
    got = np.take(a, indices, out=out)
    assert got is out
    assert np.array_equal(got, a[indices])


# --- 2. out-of-bounds: StockRaised route, path still named, no warning ---


def test_out_of_bounds_raises_stock_indexerror_no_warning_positive():
    a = np.arange(SIZE_MIN + 10, dtype=np.float64)
    indices = np.arange(SIZE_MIN, dtype=np.intp)
    indices[-1] = a.size + 5  # out of bounds
    out = np.empty(indices.shape, dtype=np.float64)
    decision, reason = GEARBOX.decide(OP, (a, indices), {"out": out})
    assert decision == PATH, (decision, reason)

    stock_out = np.empty(indices.shape, dtype=np.float64)
    with pytest.raises(IndexError):
        _stock()(a, indices, out=stock_out)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(IndexError):
            np.take(a, indices, out=out)


def test_out_of_bounds_raises_stock_indexerror_no_warning_negative():
    a = np.arange(SIZE_MIN + 10, dtype=np.int64)
    indices = np.arange(SIZE_MIN, dtype=np.intp)
    indices[0] = -(a.size + 1)  # out of bounds (negative)
    out = np.empty(indices.shape, dtype=np.int64)
    decision, reason = GEARBOX.decide(OP, (a, indices), {"out": out})
    assert decision == PATH, (decision, reason)

    stock_out = np.empty(indices.shape, dtype=np.int64)
    with pytest.raises(IndexError):
        _stock()(a, indices, out=stock_out)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(IndexError):
            np.take(a, indices, out=out)


# --- 3. refusals, stock parity ---------------------------------------------


def test_refusal_no_out_kwarg():
    a = np.arange(SIZE_MIN + 10, dtype=np.float64)
    indices = np.arange(SIZE_MIN, dtype=np.intp)
    _assert_refused((a, indices))


def test_refusal_axis_given():
    a = np.arange(SIZE_MIN + 10, dtype=np.float64)
    indices = np.arange(SIZE_MIN, dtype=np.intp) % a.size
    out = np.empty(indices.shape, dtype=np.float64)
    _assert_refused((a, indices), {"out": out, "axis": 0})


def test_refusal_mode_given():
    a = np.arange(SIZE_MIN + 10, dtype=np.float64)
    indices = np.arange(SIZE_MIN, dtype=np.intp) % a.size
    out = np.empty(indices.shape, dtype=np.float64)
    _assert_refused((a, indices), {"out": out, "mode": "clip"})


def test_refusal_below_size_min():
    a = np.arange(SIZE_MIN, dtype=np.float64)
    indices = np.arange(SIZE_MIN - 1, dtype=np.intp) % a.size
    out = np.empty(indices.shape, dtype=np.float64)
    _assert_refused((a, indices), {"out": out})


def test_refusal_2d_a():
    a = np.arange((SIZE_MIN + 10) * 2, dtype=np.float64).reshape(-1, 2)
    indices = np.arange(SIZE_MIN, dtype=np.intp) % a.shape[0]
    out = np.empty(indices.shape, dtype=np.float64)
    _assert_refused((a, indices), {"out": out, "axis": 0})


def test_refusal_2d_indices():
    a = np.arange(SIZE_MIN + 10, dtype=np.float64)
    indices = (np.arange(SIZE_MIN, dtype=np.intp) % a.size).reshape(-1, 1)
    out = np.empty(indices.shape, dtype=np.float64)
    _assert_refused((a, indices), {"out": out})


def test_refusal_int32_indices():
    a = np.arange(SIZE_MIN + 10, dtype=np.float64)
    indices = (np.arange(SIZE_MIN, dtype=np.int32) % a.size).astype(np.int32)
    out = np.empty(indices.shape, dtype=np.float64)
    _assert_refused((a, indices), {"out": out})


def test_refusal_f32_a():
    a = np.arange(SIZE_MIN + 10, dtype=np.float32)
    indices = np.arange(SIZE_MIN, dtype=np.intp) % a.size
    out = np.empty(indices.shape, dtype=np.float32)
    _assert_refused((a, indices), {"out": out})


def test_refusal_out_dtype_mismatch():
    a = np.arange(SIZE_MIN + 10, dtype=np.float64)
    indices = np.arange(SIZE_MIN, dtype=np.intp) % a.size
    out = np.empty(indices.shape, dtype=np.int64)
    _assert_refused((a, indices), {"out": out})


def test_refusal_out_shape_mismatch():
    a = np.arange(SIZE_MIN + 10, dtype=np.float64)
    indices = np.arange(SIZE_MIN, dtype=np.intp) % a.size
    out = np.empty(indices.size + 1, dtype=np.float64)
    _assert_refused((a, indices), {"out": out})


def test_refusal_out_as_list():
    a = np.arange(SIZE_MIN + 10, dtype=np.float64)
    indices = np.arange(SIZE_MIN, dtype=np.intp) % a.size
    out = [0.0] * indices.size
    _assert_refused((a, indices), {"out": out})
