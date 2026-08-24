"""Differential tests: PyRallel chunked ufunc fast paths vs stock NumPy.

Contract is bit-identical output wherever a ``pyrallel_<op>`` path dispatches,
and correct fallback (stock result) everywhere else. Table-driven from
``parallel_ufunc.SUPPORTED``; hardcodes no op, dtype, or threshold.
"""

from __future__ import annotations

import concurrent.futures

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pyoverdrive
from pyoverdrive.fastpaths.parallel_ufunc import SUPPORTED
from pyoverdrive.parallel import pyrallel

STOCK = {op: getattr(np, op) for op in SUPPORTED}

SUPPORTED_CASES = [
    (op, dtype, threshold)
    for op, table in SUPPORTED.items()
    for dtype, threshold in table.items()
]

RNG = np.random.default_rng(20260823)

_FINITE_DOMAINS = {
    "sin": (-1000.0, 1000.0),
    "cos": (-1000.0, 1000.0),
    "tan": (-1000.0, 1000.0),
    "tanh": (-1000.0, 1000.0),
    "exp": (-5.0, 5.0),
    "log": (1e-6, 1000.0),
    "log10": (1e-6, 1000.0),
    "sqrt": (0.0, 1000.0),
}

_LOG_FAMILY = ("log", "log10", "sqrt")


def _domain_for(op: str) -> tuple[float, float]:
    return _FINITE_DOMAINS.get(op, (-1000.0, 1000.0))


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([f"numpy.{op}" for op in SUPPORTED])
    yield
    pyoverdrive.disable()
    pyrallel.shutdown()


def _assert_bit_identical(got, expected):
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    assert np.array_equal(got, expected, equal_nan=True)


# -- dispatch + bit-identity, table-driven -------------------------------------

_DISPATCH_PARAMS = [
    (op, dtype, threshold, size)
    for op, dtype, threshold in SUPPORTED_CASES
    for size in (threshold, threshold + 7, 3 * threshold + 1)
]


@pytest.mark.parametrize("op,dtype,threshold,size", _DISPATCH_PARAMS)
def test_dispatch_bit_identical(op, dtype, threshold, size):
    lo, hi = _domain_for(op)
    x = RNG.uniform(lo, hi, size=size).astype(dtype)
    decision, reason = pyoverdrive.explain(f"numpy.{op}", x)
    assert decision == f"pyrallel_{op}", (decision, reason)
    got = getattr(np, op)(x)
    expected = STOCK[op](x)
    _assert_bit_identical(got, expected)


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_dispatch_2d_bit_identical(op, dtype, threshold):
    cols = 100
    rows = threshold // cols + 1
    lo, hi = _domain_for(op)
    x = RNG.uniform(lo, hi, size=(rows, cols)).astype(dtype)
    assert x.flags.c_contiguous
    assert x.size >= threshold
    decision, reason = pyoverdrive.explain(f"numpy.{op}", x)
    assert decision == f"pyrallel_{op}", (decision, reason)
    got = getattr(np, op)(x)
    expected = STOCK[op](x)
    _assert_bit_identical(got, expected)


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_special_values_bit_identical(op, dtype, threshold):
    lo, hi = _domain_for(op)
    fill_lo = max(lo, -100.0)
    fill_hi = min(hi, 100.0)
    x = RNG.uniform(fill_lo, fill_hi, size=threshold).astype(dtype)
    specials = [np.nan, np.inf, -np.inf, -0.0]
    if op in _LOG_FAMILY:
        specials.append(-5.0)
    for i, v in enumerate(specials):
        x[i] = v
    with np.errstate(all="ignore"):
        decision, reason = pyoverdrive.explain(f"numpy.{op}", x)
        assert decision == f"pyrallel_{op}", (decision, reason)
        got = getattr(np, op)(x)
        expected = STOCK[op](x)
    _assert_bit_identical(got, expected)


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_inplace_and_out_form(op, dtype, threshold):
    lo, hi = _domain_for(op)
    x = RNG.uniform(lo, hi, size=threshold).astype(dtype)
    expected = STOCK[op](x.copy())

    x_inplace = x.copy()
    result_inplace = getattr(np, op)(x_inplace, out=x_inplace)
    assert result_inplace is x_inplace
    _assert_bit_identical(result_inplace, expected)

    x_out = x.copy()
    out = np.empty_like(x_out)
    result_out = getattr(np, op)(x_out, out=out)
    assert result_out is out
    _assert_bit_identical(result_out, expected)


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_fortran_order_falls_back_but_correct(op, dtype, threshold):
    lo, hi = _domain_for(op)
    n = threshold * 2
    x = np.asfortranarray(RNG.uniform(lo, hi, size=(n, 2)).astype(dtype))
    assert not x.flags.c_contiguous
    decision, _ = pyoverdrive.explain(f"numpy.{op}", x)
    assert decision == "stock"
    got = getattr(np, op)(x)
    expected = STOCK[op](x)
    _assert_bit_identical(got, expected)


# -- concurrency ------------------------------------------------------------------

def test_concurrent_threads_bit_identical():
    op, dtype, threshold = SUPPORTED_CASES[0]
    lo, hi = _domain_for(op)
    size = threshold * 2
    fn = getattr(np, op)
    stock_fn = STOCK[op]

    def worker(seed):
        rng = np.random.default_rng(seed)
        ok = True
        for _ in range(5):
            x = rng.uniform(lo, hi, size=size).astype(dtype)
            expected = stock_fn(x.copy())
            got = fn(x.copy())
            ok = ok and got.dtype == expected.dtype and np.array_equal(
                got, expected, equal_nan=True
            )
        return ok

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(worker, seed) for seed in range(8)]
        outcomes = [f.result() for f in futures]
    assert all(outcomes)


# -- hypothesis fuzz ---------------------------------------------------------------

@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_hypothesis_fuzz_bit_identical(data):
    op, dtype, threshold = data.draw(st.sampled_from(SUPPORTED_CASES))
    size = data.draw(st.integers(min_value=threshold, max_value=threshold * 2))
    width = 32 if np.dtype(dtype) == np.dtype(np.float32) else 64
    x = data.draw(
        hnp.arrays(
            dtype=dtype,
            shape=size,
            elements=st.floats(allow_nan=True, allow_infinity=True, width=width),
        )
    )
    with np.errstate(all="ignore"):
        got = getattr(np, op)(x.copy())
        expected = STOCK[op](x.copy())
    _assert_bit_identical(got, expected)
