"""Differential tests: relayout_blocked fast path vs stock np.ascontiguousarray.

Contract is BIT-IDENTICAL output wherever the path dispatches (fresh
C-contiguous copy, not a view of the input) and correct fallback (stock
result, same object semantics where stock itself avoids a copy) everywhere
the predicate declines.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.fastpaths.relayout_blocked import SUPPORTED, TILE
from pyoverdrive.parallel import pyrallel

STOCK = np.ascontiguousarray  # captured before enabling

RNG = np.random.default_rng(20260823)


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(["numpy.ascontiguousarray"])
    yield
    pyoverdrive.disable()
    pyrallel.shutdown()


def _fortran2d(shape, dtype, rng=None):
    """F-contiguous, NOT C-contiguous, array of `shape`/`dtype` via base.T."""
    rng = RNG if rng is None else rng
    dtype = np.dtype(dtype)
    rows, cols = shape
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        base = rng.integers(info.min, info.max, size=(cols, rows), dtype=dtype, endpoint=True)
    elif np.issubdtype(dtype, np.complexfloating):
        real = rng.standard_normal(size=(cols, rows))
        imag = rng.standard_normal(size=(cols, rows))
        base = (real + 1j * imag).astype(dtype)
    else:
        base = rng.standard_normal(size=(cols, rows)).astype(dtype)
    x = base.T
    assert x.shape == shape
    assert x.flags["F_CONTIGUOUS"] and not x.flags["C_CONTIGUOUS"]
    return x


def _assert_identical(got, x):
    expected = STOCK(x)
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    assert np.array_equal(got, expected, equal_nan=True)
    assert got.flags["C_CONTIGUOUS"]


# -- dispatch shapes, per dtype ------------------------------------------------

def _shapes_for(threshold):
    n = int(np.sqrt(threshold))
    assert n * n == threshold
    shapes = {
        "square_at_threshold": (n, n),
        "square_n+1": (n + 1, n + 1),
        "square_n+7": (n + 7, n + 7),
        "tall": (4 * n, n // 2),
        "wide": (n // 2, 4 * n),
    }
    one_row = (TILE + 3, 4 * TILE + 5)
    if one_row[0] * one_row[1] >= threshold:
        shapes["one_row_of_tiles"] = one_row
    return shapes


DISPATCH_CASES = []
for _dtype, _threshold in SUPPORTED.items():
    for _name, _shape in _shapes_for(_threshold).items():
        DISPATCH_CASES.append(pytest.param(_dtype, _shape, id=f"{_dtype}-{_name}"))


@pytest.mark.parametrize("dtype, shape", DISPATCH_CASES)
def test_dispatched_bit_identical(dtype, shape):
    x = _fortran2d(shape, dtype)
    decision, reason = pyoverdrive.explain("numpy.ascontiguousarray", x)
    assert decision == "relayout_blocked", (decision, reason)
    got = np.ascontiguousarray(x)
    _assert_identical(got, x)
    assert not np.shares_memory(got, x)


# -- special float values -------------------------------------------------------

@pytest.mark.parametrize("dtype", [d for d in SUPPORTED if np.issubdtype(d, np.floating)])
def test_special_values_bit_identical(dtype):
    threshold = SUPPORTED[dtype]
    n = int(np.sqrt(threshold))
    x = _fortran2d((n, n), dtype)
    x[0, 0] = np.nan
    x[0, 1] = np.inf
    x[1, 0] = -np.inf
    x[1, 1] = -0.0
    x[n - 1, n - 1] = np.nan
    x[n // 2, n // 2] = np.inf
    x[n // 2, n // 2 - 1] = -0.0
    assert pyoverdrive.explain("numpy.ascontiguousarray", x)[0] == "relayout_blocked"
    got = np.ascontiguousarray(x)
    expected = STOCK(x)
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected, equal_nan=True)
    assert np.array_equal(np.signbit(got), np.signbit(expected))
    assert not np.shares_memory(got, x)


# -- exact element mapping -------------------------------------------------------

def test_exact_element_mapping_float64():
    dtype = np.dtype(np.float64)
    threshold = SUPPORTED[dtype]
    n = int(np.sqrt(threshold))
    x = _fortran2d((n, n), dtype)
    assert pyoverdrive.explain("numpy.ascontiguousarray", x)[0] == "relayout_blocked"
    got = np.ascontiguousarray(x)
    assert np.array_equal(got, x)
    idx_rng = np.random.default_rng(2026823)
    for _ in range(20):
        i = int(idx_rng.integers(0, n))
        j = int(idx_rng.integers(0, n))
        assert got[i, j] == x[i, j]


# -- fallbacks --------------------------------------------------------------------

def test_c_contiguous_input_falls_back():
    dtype = np.dtype(np.float64)
    n = int(np.sqrt(SUPPORTED[dtype]))
    c = RNG.standard_normal(size=(n, n)).astype(dtype)
    assert c.flags["C_CONTIGUOUS"]
    assert pyoverdrive.explain("numpy.ascontiguousarray", c)[0] == "stock"
    assert STOCK(c) is c
    assert np.ascontiguousarray(c) is c


def test_below_threshold_falls_back():
    dtype = np.dtype(np.float64)
    threshold = SUPPORTED[dtype]
    n = int(np.sqrt(threshold))
    x = _fortran2d((n - 1, n - 1), dtype)
    assert pyoverdrive.explain("numpy.ascontiguousarray", x)[0] == "stock"
    got = np.ascontiguousarray(x)
    _assert_identical(got, x)


def test_1d_input_falls_back():
    x = RNG.standard_normal(300_000).astype(np.float64)
    assert pyoverdrive.explain("numpy.ascontiguousarray", x)[0] == "stock"
    got = np.ascontiguousarray(x)
    _assert_identical(got, x)


def test_3d_input_falls_back():
    dtype = np.dtype(np.float64)
    shape = (64, 64, 64)  # size 262144 >= float64 threshold
    base = RNG.standard_normal(size=shape[::-1]).astype(dtype)
    x = base.T
    assert x.ndim == 3
    assert x.flags["F_CONTIGUOUS"] and not x.flags["C_CONTIGUOUS"]
    assert x.size >= SUPPORTED[dtype]
    assert pyoverdrive.explain("numpy.ascontiguousarray", x)[0] == "stock"
    got = np.ascontiguousarray(x)
    _assert_identical(got, x)


@pytest.mark.parametrize("dtype", [np.int32, np.float16, np.complex128])
def test_unsupported_dtype_falls_back(dtype):
    x = _fortran2d((600, 600), dtype)
    assert pyoverdrive.explain("numpy.ascontiguousarray", x)[0] == "stock"
    got = np.ascontiguousarray(x)
    _assert_identical(got, x)


def test_ndarray_subclass_falls_back():
    class Sub(np.ndarray):
        pass

    dtype = np.dtype(np.float64)
    n = int(np.sqrt(SUPPORTED[dtype]))
    x = _fortran2d((n, n), dtype).view(Sub)
    assert x.flags["F_CONTIGUOUS"] and not x.flags["C_CONTIGUOUS"]
    assert pyoverdrive.explain("numpy.ascontiguousarray", x)[0] == "stock"
    got = np.ascontiguousarray(x)
    _assert_identical(got, x)


def test_dtype_kwarg_falls_back():
    dtype = np.dtype(np.float64)
    n = int(np.sqrt(SUPPORTED[dtype]))
    x = _fortran2d((n, n), dtype)
    assert pyoverdrive.explain("numpy.ascontiguousarray", x, dtype=np.float64)[0] == "stock"
    got = np.ascontiguousarray(x, dtype=np.float64)
    expected = STOCK(x, dtype=np.float64)
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


def test_neither_c_nor_f_contiguous_view_falls_back():
    dtype = np.dtype(np.float64)
    n = int(np.sqrt(SUPPORTED[dtype]))
    base = RNG.standard_normal(size=(2 * n, 2 * n)).astype(dtype)
    x = base[::2, ::2]
    assert not x.flags["C_CONTIGUOUS"] and not x.flags["F_CONTIGUOUS"]
    assert pyoverdrive.explain("numpy.ascontiguousarray", x)[0] == "stock"
    got = np.ascontiguousarray(x)
    _assert_identical(got, x)


def test_python_lists_fall_back():
    x = [[1.0, 2.0], [3.0, 4.0]]
    assert pyoverdrive.explain("numpy.ascontiguousarray", x)[0] == "stock"
    got = np.ascontiguousarray(x)
    expected = STOCK(x)
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


# -- input untouched ---------------------------------------------------------------

def test_input_untouched():
    dtype = np.dtype(np.float64)
    n = int(np.sqrt(SUPPORTED[dtype]))
    x = _fortran2d((n, n), dtype)
    before = x.copy()
    assert pyoverdrive.explain("numpy.ascontiguousarray", x)[0] == "relayout_blocked"
    np.ascontiguousarray(x)
    assert np.array_equal(x, before)


# -- concurrency ---------------------------------------------------------------------

def test_concurrent_dispatch_bit_identical():
    dtype = np.dtype(np.float64)
    n = int(np.sqrt(SUPPORTED[dtype]))
    n_threads = 6
    iterations = 3
    errors = []
    barrier = threading.Barrier(n_threads)

    def worker(idx):
        rng = np.random.default_rng(900_000 + idx)
        for _ in range(iterations):
            x = _fortran2d((n, n), dtype, rng=rng)
            barrier.wait()
            try:
                decision = pyoverdrive.explain("numpy.ascontiguousarray", x)[0]
                got = np.ascontiguousarray(x)
                expected = STOCK(x)
                ok = (
                    decision == "relayout_blocked"
                    and got.dtype == expected.dtype
                    and got.flags["C_CONTIGUOUS"]
                    and np.array_equal(got, expected)
                )
                if not ok:
                    errors.append((idx, decision))
            except Exception as exc:  # pragma: no cover - failure surfaced via errors
                errors.append((idx, repr(exc)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
