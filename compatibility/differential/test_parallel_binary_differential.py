"""Differential tests: PyRallel chunked binary-elementwise fast paths vs stock
NumPy.

Contract is bit-identical output wherever a ``pyrallel_<op>`` path dispatches,
and correct fallback (stock result) everywhere else. Table-driven from
``parallel_binary.SUPPORTED``; hardcodes no op, dtype, or threshold.
"""

from __future__ import annotations

import concurrent.futures

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pyoverdrive
from pyoverdrive.fastpaths.parallel_binary import SUPPORTED
from pyoverdrive.parallel import pyrallel

STOCK = {op: getattr(np, op) for op in SUPPORTED}

SUPPORTED_CASES = [
    (op, dtype, threshold)
    for op, table in SUPPORTED.items()
    for dtype, threshold in table.items()
]

_FLOAT_CASES = [(op, dtype, t) for op, dtype, t in SUPPORTED_CASES if dtype.kind == "f"]

RNG = np.random.default_rng(20260823)

_FIRST_OP = next(iter(SUPPORTED))
_F64 = np.dtype(np.float64)


class Sub(np.ndarray):
    """Trivial ndarray subclass; predicate requires ``type(x) is np.ndarray``."""


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([f"numpy.{op}" for op in SUPPORTED])
    yield
    pyoverdrive.disable()
    pyrallel.shutdown()


def _operand(dtype: np.dtype, shape, rng=RNG) -> np.ndarray:
    if dtype.kind == "f":
        return (rng.random(shape) + 0.5).astype(dtype)
    return rng.integers(1, 1000, size=shape).astype(dtype)


def _assert_bit_identical(got, expected):
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    assert np.array_equal(got, expected, equal_nan=True)


# -- dispatch + bit-identity, table-driven -----------------------------------

_DISPATCH_PARAMS = [
    (op, dtype, threshold, size)
    for op, dtype, threshold in SUPPORTED_CASES
    for size in (threshold, threshold + 7)
]


@pytest.mark.parametrize("op,dtype,threshold,size", _DISPATCH_PARAMS)
def test_dispatch_bit_identical(op, dtype, threshold, size):
    a = _operand(dtype, size)
    b = _operand(dtype, size)
    decision, reason = pyoverdrive.explain(f"numpy.{op}", a, b)
    assert decision == f"pyrallel_{op}", (decision, reason)
    got = getattr(np, op)(a, b)
    expected = STOCK[op](a, b)
    _assert_bit_identical(got, expected)


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_dispatch_2d_bit_identical(op, dtype, threshold):
    cols = 100
    rows = threshold // cols + 1
    a = _operand(dtype, (rows, cols))
    b = _operand(dtype, (rows, cols))
    assert a.flags.c_contiguous and b.flags.c_contiguous
    assert a.size >= threshold
    decision, reason = pyoverdrive.explain(f"numpy.{op}", a, b)
    assert decision == f"pyrallel_{op}", (decision, reason)
    got = getattr(np, op)(a, b)
    expected = STOCK[op](a, b)
    _assert_bit_identical(got, expected)


@pytest.mark.parametrize("op,dtype,threshold", _FLOAT_CASES)
def test_special_values_bit_identical(op, dtype, threshold):
    a = _operand(dtype, threshold)
    b = _operand(dtype, threshold)
    specials = [np.nan, np.inf, -np.inf, -0.0]
    for i, v in enumerate(specials):
        a[i] = v
        b[i] = v
    if op == "divide":
        # zeros in the divisor, on operands otherwise untouched by specials.
        b[len(specials) : len(specials) + 4] = 0.0
    with np.errstate(all="ignore"):
        decision, reason = pyoverdrive.explain(f"numpy.{op}", a, b)
        assert decision == f"pyrallel_{op}", (decision, reason)
        got = getattr(np, op)(a, b)
        expected = STOCK[op](a, b)
    _assert_bit_identical(got, expected)


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_inplace_and_out_form(op, dtype, threshold):
    a = _operand(dtype, threshold)
    b = _operand(dtype, threshold)
    expected = STOCK[op](a.copy(), b.copy())

    a_inplace = a.copy()
    result_inplace = getattr(np, op)(a_inplace, b, out=a_inplace)
    assert result_inplace is a_inplace
    _assert_bit_identical(result_inplace, expected)

    out = np.empty_like(a)
    result_out = getattr(np, op)(a, b, out=out)
    assert result_out is out
    _assert_bit_identical(result_out, expected)


# -- fallbacks: explain() says "stock" AND the result still equals stock ----


def _check_fallback(op, a, b, kwargs=None):
    kwargs = kwargs or {}
    decision, reason = pyoverdrive.explain(f"numpy.{op}", a, b, **kwargs)
    assert decision == "stock", (decision, reason)
    got = getattr(np, op)(a, b, **kwargs)
    expected = STOCK[op](a, b, **kwargs)
    _assert_bit_identical(got, expected)


def test_fallback_scalar_second_operand():
    threshold = SUPPORTED["add"][_F64]
    a = _operand(_F64, threshold)
    _check_fallback("add", a, 2.0)


def test_fallback_broadcast_1d_vs_scalar_shape():
    op = "add"
    threshold = SUPPORTED[op][_F64]
    a = _operand(_F64, threshold)
    b = _operand(_F64, (1,))
    _check_fallback(op, a, b)


def test_fallback_broadcast_2d_vs_1d():
    op = "add"
    threshold = SUPPORTED[op][_F64]
    cols = 100
    rows = threshold // cols + 1
    a = _operand(_F64, (rows, cols))
    b = _operand(_F64, (cols,))
    _check_fallback(op, a, b)


def test_fallback_mixed_dtype_float64_float32():
    op = "add"
    threshold = SUPPORTED[op][_F64]
    a = _operand(_F64, threshold)
    b = _operand(np.dtype(np.float32), threshold)
    _check_fallback(op, a, b)


def test_fallback_mixed_dtype_int64_float64():
    op = "add"
    threshold = SUPPORTED[op][np.dtype(np.int64)]
    a = _operand(np.dtype(np.int64), threshold)
    b = _operand(_F64, threshold)
    _check_fallback(op, a, b)


def test_fallback_fortran_order():
    op = "add"
    threshold = SUPPORTED[op][_F64]
    cols = 100
    rows = threshold // cols + 1
    a = np.asfortranarray(_operand(_F64, (rows, cols)))
    b = np.asfortranarray(_operand(_F64, (rows, cols)))
    assert not a.flags.c_contiguous and not b.flags.c_contiguous
    _check_fallback(op, a, b)


def test_fallback_noncontiguous_operand():
    op = "add"
    threshold = SUPPORTED[op][_F64]
    a_base = _operand(_F64, threshold * 2)
    a = a_base[::2]
    assert not a.flags.c_contiguous
    assert a.size == threshold
    b = _operand(_F64, threshold)
    _check_fallback(op, a, b)


def test_fallback_ndarray_subclass():
    op = "add"
    threshold = SUPPORTED[op][_F64]
    a = _operand(_F64, threshold).view(Sub)
    b = _operand(_F64, threshold)
    _check_fallback(op, a, b)


def test_fallback_where_kwarg():
    op = "add"
    threshold = SUPPORTED[op][_F64]
    a = _operand(_F64, threshold)
    b = _operand(_F64, threshold)
    mask = np.zeros(threshold, dtype=bool)
    mask[::2] = True
    out_got = np.zeros(threshold, dtype=_F64)
    out_expected = out_got.copy()
    decision, reason = pyoverdrive.explain(
        f"numpy.{op}", a, b, where=mask, out=out_got
    )
    assert decision == "stock", (decision, reason)
    got = getattr(np, op)(a, b, where=mask, out=out_got)
    expected = STOCK[op](a, b, where=mask, out=out_expected)
    assert got is out_got
    _assert_bit_identical(got, expected)


def test_fallback_out_wrong_dtype():
    op = "add"
    threshold = SUPPORTED[op][_F64]
    a = _operand(_F64, threshold)
    b = _operand(_F64, threshold)
    out_got = np.empty(threshold, dtype=np.float32)
    out_expected = np.empty(threshold, dtype=np.float32)
    decision, reason = pyoverdrive.explain(f"numpy.{op}", a, b, out=out_got)
    assert decision == "stock", (decision, reason)
    got = getattr(np, op)(a, b, out=out_got)
    expected = STOCK[op](a, b, out=out_expected)
    assert got is out_got
    _assert_bit_identical(got, expected)


def test_fallback_readonly_out():
    op = "add"
    threshold = SUPPORTED[op][_F64]
    a = _operand(_F64, threshold)
    b = _operand(_F64, threshold)
    out_ro = np.empty(threshold, dtype=_F64)
    out_ro.flags.writeable = False
    decision, reason = pyoverdrive.explain(f"numpy.{op}", a, b, out=out_ro)
    assert decision == "stock", (decision, reason)
    with pytest.raises(ValueError):
        getattr(np, op)(a, b, out=out_ro)
    with pytest.raises(ValueError):
        STOCK[op](a, b, out=out_ro)


def test_fallback_below_threshold():
    op = "add"
    threshold = SUPPORTED[op][_F64]
    a = _operand(_F64, threshold - 1)
    b = _operand(_F64, threshold - 1)
    _check_fallback(op, a, b)


# -- errstate -----------------------------------------------------------------


def test_errstate_raise_forces_stock():
    op = _FIRST_OP
    dtype = next(iter(SUPPORTED[op]))
    threshold = SUPPORTED[op][dtype]
    a = _operand(dtype, threshold)
    b = _operand(dtype, threshold)
    with np.errstate(all="raise"):
        decision, reason = pyoverdrive.explain(f"numpy.{op}", a, b)
        assert decision == "stock", (decision, reason)


# -- concurrency ----------------------------------------------------------------


def test_concurrent_threads_bit_identical():
    op = _FIRST_OP
    dtype = next(iter(SUPPORTED[op]))
    threshold = SUPPORTED[op][dtype]
    fn = getattr(np, op)
    stock_fn = STOCK[op]

    def worker(seed):
        rng = np.random.default_rng(seed)
        ok = True
        for _ in range(3):
            a = _operand(dtype, threshold, rng=rng)
            b = _operand(dtype, threshold, rng=rng)
            expected = stock_fn(a.copy(), b.copy())
            got = fn(a.copy(), b.copy())
            ok = ok and got.dtype == expected.dtype and np.array_equal(
                got, expected, equal_nan=True
            )
        return ok

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(worker, seed) for seed in range(8)]
        outcomes = [f.result() for f in futures]
    assert all(outcomes)


# -- hypothesis fuzz ------------------------------------------------------------

@given(data=st.data())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[
        HealthCheck.data_too_large,
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
    ],
)
def test_hypothesis_fuzz_bit_identical(data):
    op, dtype, threshold = data.draw(st.sampled_from(_FLOAT_CASES))
    size = data.draw(st.integers(min_value=threshold, max_value=threshold + 64))
    width = 32 if dtype == np.dtype(np.float32) else 64
    a = data.draw(
        hnp.arrays(
            dtype=dtype,
            shape=size,
            elements=st.floats(allow_nan=True, allow_infinity=True, width=width),
        )
    )
    b = data.draw(
        hnp.arrays(
            dtype=dtype,
            shape=size,
            elements=st.floats(allow_nan=True, allow_infinity=True, width=width),
        )
    )
    with np.errstate(all="ignore"):
        got = getattr(np, op)(a.copy(), b.copy())
        expected = STOCK[op](a.copy(), b.copy())
    _assert_bit_identical(got, expected)
