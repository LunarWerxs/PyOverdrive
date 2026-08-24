"""Differential tests: intersect_sorted fast path vs stock np.intersect1d.

Contract is BIT-IDENTICAL output wherever the path dispatches (same dtype,
same values, same order), and correct fallback (stock result) everywhere
the predicate declines.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pyoverdrive
from pyoverdrive.fastpaths.intersect_sorted import SIZE_THRESHOLD, _THRESHOLDS

STOCK = np.intersect1d

DTYPES = (np.int32, np.int64, np.uint32, np.uint64)
SMALL_DTYPES = (np.int8, np.uint8, np.int16, np.uint16)


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(["numpy.intersect1d"])
    yield
    pyoverdrive.disable()


def _assert_identical(got, expected):
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


# -- regime generators: (rng, dtype) -> (a, b) -------------------------------

def _both_sorted_unique(rng, dtype, na=2000, nb=1800, pool=3000):
    a = np.sort(rng.choice(pool, size=na, replace=False)).astype(dtype)
    b = np.sort(rng.choice(pool, size=nb, replace=False)).astype(dtype)
    return a, b


def _both_random_dup(rng, dtype, na=2000, nb=1800, low=0, high=500):
    return (
        rng.integers(low, high, size=na, dtype=dtype),
        rng.integers(low, high, size=nb, dtype=dtype),
    )


def _one_sorted_one_random(rng, dtype):
    a, _ = _both_sorted_unique(rng, dtype)
    _, b = _both_random_dup(rng, dtype)
    return a, b


def _identical(rng, dtype):
    arr = rng.integers(0, 1000, size=2000, dtype=dtype)
    return arr.copy(), arr.copy()


def _disjoint(rng, dtype):
    a = rng.integers(0, 1000, size=2000, dtype=dtype)
    b = rng.integers(2000, 3000, size=1800, dtype=dtype)
    return a, b


def _reversed(rng, dtype):
    a, b = _both_sorted_unique(rng, dtype)
    return a[::-1], b[::-1]


def _full_range(rng, dtype):
    info = np.iinfo(dtype)
    a = rng.integers(info.min, info.max, size=2000, dtype=dtype, endpoint=True)
    b = rng.integers(info.min, info.max, size=1800, dtype=dtype, endpoint=True)
    a[0], a[1] = info.min, info.max
    b[0], b[1] = info.min, info.max
    return a, b


def _one_vs_large(rng, dtype):
    a = rng.integers(0, 1000, size=1, dtype=dtype)
    b = rng.integers(0, 1000, size=5000, dtype=dtype)
    return a, b


REGIMES = {
    "both_sorted_unique": _both_sorted_unique,
    "both_random_dup": _both_random_dup,
    "one_sorted_one_random": _one_sorted_one_random,
    "identical": _identical,
    "disjoint": _disjoint,
    "reversed": _reversed,
    "full_range": _full_range,
    "one_vs_large": _one_vs_large,
}

RNG = np.random.default_rng(27042)

DISPATCH_CASES = []
for dtype in DTYPES:
    for name, gen in REGIMES.items():
        a, b = gen(RNG, dtype)
        DISPATCH_CASES.append(pytest.param(a, b, id=f"{np.dtype(dtype)}-{name}"))


@pytest.mark.parametrize("a, b", DISPATCH_CASES)
def test_dispatched_bit_identical(a, b):
    decision, reason = pyoverdrive.explain("numpy.intersect1d", a, b)
    assert decision == "intersect_sorted", (decision, reason)
    _assert_identical(np.intersect1d(a, b), STOCK(a, b))


# -- threshold boundary -------------------------------------------------------

def test_threshold_dispatches():
    a = RNG.integers(0, 1000, size=32, dtype=np.int64)
    b = RNG.integers(0, 1000, size=SIZE_THRESHOLD - 32, dtype=np.int64)
    assert a.size + b.size == SIZE_THRESHOLD
    assert pyoverdrive.explain("numpy.intersect1d", a, b)[0] == "intersect_sorted"
    _assert_identical(np.intersect1d(a, b), STOCK(a, b))


def test_below_threshold_falls_back():
    a = RNG.integers(0, 1000, size=32, dtype=np.int64)
    b = RNG.integers(0, 1000, size=SIZE_THRESHOLD - 33, dtype=np.int64)
    assert a.size + b.size == SIZE_THRESHOLD - 1
    assert pyoverdrive.explain("numpy.intersect1d", a, b)[0] == "stock"
    _assert_identical(np.intersect1d(a, b), STOCK(a, b))


# -- small-dtype threshold (12_000 combined, itemsize <= 2 radix floor) ------

def _small_random(rng, dtype, na, nb):
    info = np.iinfo(dtype)
    return (
        rng.integers(info.min, info.max, size=na, dtype=dtype, endpoint=True),
        rng.integers(info.min, info.max, size=nb, dtype=dtype, endpoint=True),
    )


SMALL_DISPATCH_CASES = [
    pytest.param(
        *_small_random(RNG, dtype, _THRESHOLDS[np.dtype(dtype)] // 2, _THRESHOLDS[np.dtype(dtype)] // 2),
        id=f"{np.dtype(dtype)}-small-random",
    )
    for dtype in SMALL_DTYPES
]


@pytest.mark.parametrize("a, b", SMALL_DISPATCH_CASES)
def test_small_dtype_dispatched_bit_identical(a, b):
    decision, reason = pyoverdrive.explain("numpy.intersect1d", a, b)
    assert decision == "intersect_sorted", (decision, reason)
    _assert_identical(np.intersect1d(a, b), STOCK(a, b))


def test_small_dtype_below_floor_falls_back():
    for dtype in SMALL_DTYPES:
        floor = _THRESHOLDS[np.dtype(dtype)]
        a = RNG.integers(0, 100, size=floor // 2, dtype=dtype)
        b = RNG.integers(0, 100, size=floor // 2 - 1, dtype=dtype)
        assert a.size + b.size == floor - 1
        assert pyoverdrive.explain("numpy.intersect1d", a, b)[0] == "stock", dtype
        _assert_identical(np.intersect1d(a, b), STOCK(a, b))


# -- fallback regimes ---------------------------------------------------------

def test_assume_unique_kwarg_falls_back():
    a, b = _both_sorted_unique(RNG, np.int64)
    assert pyoverdrive.explain("numpy.intersect1d", a, b, assume_unique=True)[0] == "stock"
    _assert_identical(
        np.intersect1d(a, b, assume_unique=True), STOCK(a, b, assume_unique=True)
    )


def test_return_indices_kwarg_falls_back():
    a, b = _both_random_dup(RNG, np.int64)
    assert pyoverdrive.explain("numpy.intersect1d", a, b, return_indices=True)[0] == "stock"
    got = np.intersect1d(a, b, return_indices=True)
    expected = STOCK(a, b, return_indices=True)
    for g, e in zip(got, expected):
        _assert_identical(g, e)


def test_mixed_dtypes_fall_back():
    a = RNG.integers(0, 1000, size=2000, dtype=np.int64)
    b = RNG.integers(0, 1000, size=1800, dtype=np.int32)
    assert pyoverdrive.explain("numpy.intersect1d", a, b)[0] == "stock"
    _assert_identical(np.intersect1d(a, b), STOCK(a, b))


def test_float_inputs_fall_back():
    a = RNG.random(2000)
    b = RNG.random(1800)
    assert pyoverdrive.explain("numpy.intersect1d", a, b)[0] == "stock"
    _assert_identical(np.intersect1d(a, b), STOCK(a, b))


def test_2d_inputs_fall_back():
    a = RNG.integers(0, 1000, size=(64, 32), dtype=np.int64)
    b = RNG.integers(0, 1000, size=(64, 32), dtype=np.int64)
    assert pyoverdrive.explain("numpy.intersect1d", a, b)[0] == "stock"
    _assert_identical(np.intersect1d(a, b), STOCK(a, b))


def test_python_lists_fall_back():
    a = list(RNG.integers(0, 1000, size=2000, dtype=np.int64))
    b = list(RNG.integers(0, 1000, size=1800, dtype=np.int64))
    assert pyoverdrive.explain("numpy.intersect1d", a, b)[0] == "stock"
    _assert_identical(np.intersect1d(a, b), STOCK(a, b))


def test_ndarray_subclass_falls_back():
    class Sub(np.ndarray):
        pass

    a = RNG.integers(0, 1000, size=2000, dtype=np.int64).view(Sub)
    b = RNG.integers(0, 1000, size=1800, dtype=np.int64).view(Sub)
    assert pyoverdrive.explain("numpy.intersect1d", a, b)[0] == "stock"
    _assert_identical(np.intersect1d(a, b), STOCK(a, b))


def test_empty_plus_large_matches_stock_regardless_of_dispatch():
    a = np.empty(0, dtype=np.int64)
    b = RNG.integers(0, 1000, size=2000, dtype=np.int64)
    # combined size is above SIZE_THRESHOLD so this may dispatch; either way
    # the result must be bit-identical to stock: an empty array of the right
    # dtype.
    _assert_identical(np.intersect1d(a, b), STOCK(a, b))


# -- non-contiguous views -----------------------------------------------------

def test_noncontiguous_1d_input_dispatches():
    base_a = RNG.integers(0, 1000, size=4000, dtype=np.int64)
    base_b = RNG.integers(0, 1000, size=3600, dtype=np.int64)
    a, b = base_a[::2], base_b[::2]
    assert not a.flags["C_CONTIGUOUS"]
    assert pyoverdrive.explain("numpy.intersect1d", a, b)[0] == "intersect_sorted"
    _assert_identical(np.intersect1d(a, b), STOCK(a, b))


# -- property-based sweep ------------------------------------------------------

@given(
    a=hnp.arrays(
        dtype=np.int64,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=400),
        elements=st.integers(min_value=-1000, max_value=1000),
    ),
    b=hnp.arrays(
        dtype=np.int64,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=400),
        elements=st.integers(min_value=-1000, max_value=1000),
    ),
)
@settings(max_examples=40, deadline=None)
def test_property_matches_stock(a, b):
    got = np.intersect1d(a, b)
    expected = STOCK(a, b)
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)
