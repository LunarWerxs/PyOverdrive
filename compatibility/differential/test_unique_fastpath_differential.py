"""Differential tests: unique_sort fast path vs stock np.unique.

Contract is BIT-IDENTICAL output wherever the path dispatches, and correct
fallback (stock result, stock dtype rules) everywhere else.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pyoverdrive
from pyoverdrive.fastpaths.unique_sort import SIZE_THRESHOLD, _SUPPORTED_DTYPES, _THRESHOLDS

STOCK_UNIQUE = np.unique
STOCK_UNIQUE_VALUES = np.unique_values


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(["numpy.unique", "numpy.unique_values"])
    yield
    pyoverdrive.disable()


def _assert_identical(got, expected):
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


# -- regimes where the fast path DISPATCHES ---------------------------------

RNG = np.random.default_rng(20260823)

DISPATCH_ARRAYS = []
for dtype in sorted(_SUPPORTED_DTYPES, key=str):
    info_kind = np.dtype(dtype).kind
    n = max(_THRESHOLDS[np.dtype(dtype)] * 4, 2048)
    if info_kind in "iu":
        info = np.iinfo(dtype)
        DISPATCH_ARRAYS += [
            RNG.integers(info.min, info.max, size=n, dtype=dtype),      # high card
            RNG.integers(0, 17, size=n, dtype=dtype),                    # low card
            np.sort(RNG.integers(info.min, info.max, size=n, dtype=dtype)),  # presorted
            RNG.integers(0, 100, size=(64, n // 64), dtype=dtype),       # 2-D flattens
        ]
    else:
        DISPATCH_ARRAYS += [
            RNG.random(n).astype(dtype),
            RNG.choice(RNG.random(13).astype(dtype), size=n),
        ]


@pytest.mark.parametrize("arr", DISPATCH_ARRAYS, ids=lambda a: f"{a.dtype}-{a.shape}")
def test_dispatched_bit_identical(arr):
    decision, reason = pyoverdrive.explain("numpy.unique", arr)
    assert decision == "unique_sort", (decision, reason)
    _assert_identical(np.unique(arr), STOCK_UNIQUE(arr))


def test_strided_and_fortran_views_dispatch_and_match():
    base = RNG.integers(0, 1000, size=SIZE_THRESHOLD * 8, dtype=np.int64)
    views = [
        base[::2],
        base[::-1],
        np.asfortranarray(base.reshape(64, -1)),
        base.reshape(64, -1)[::2, 1::3],
    ]
    for v in views:
        if v.size >= SIZE_THRESHOLD:
            assert pyoverdrive.explain("numpy.unique", v)[0] == "unique_sort"
        _assert_identical(np.unique(v), STOCK_UNIQUE(v))


def test_unique_values_sorted_superset_of_contract():
    arr = RNG.integers(0, 10_000, size=SIZE_THRESHOLD * 4, dtype=np.int32)
    got = np.unique_values(arr)
    expected = STOCK_UNIQUE_VALUES(arr)
    # unique_values guarantees no order; compare as sorted sets
    np.testing.assert_array_equal(np.sort(got), np.sort(expected))


# -- regimes that MUST fall back --------------------------------------------

def test_small_arrays_fall_back():
    arr = np.arange(SIZE_THRESHOLD - 1, dtype=np.int64)
    assert pyoverdrive.explain("numpy.unique", arr)[0] == "stock"
    _assert_identical(np.unique(arr), STOCK_UNIQUE(arr))


def test_unsupported_dtypes_fall_back():
    for arr in (
        RNG.random(SIZE_THRESHOLD * 4),                          # float64 excluded
        RNG.random(SIZE_THRESHOLD * 4).astype(np.float32),       # float32 excluded
        RNG.integers(0, 100, SIZE_THRESHOLD * 4, dtype=np.int8),   # below int8 floor (1000)
        RNG.integers(0, 100, SIZE_THRESHOLD * 4, dtype=np.int16),  # below int16 floor (10_000)
        np.array(["a", "b"] * SIZE_THRESHOLD),                   # strings
        (RNG.random(SIZE_THRESHOLD * 4) < 0.5),                  # bool
    ):
        decision, _ = pyoverdrive.explain("numpy.unique", arr)
        assert decision == "stock", arr.dtype


def test_small_dtype_just_below_floor_falls_back():
    for dtype in (np.int8, np.uint8, np.int16, np.uint16):
        floor = _THRESHOLDS[np.dtype(dtype)]
        arr = RNG.integers(0, 100, size=floor - 1, dtype=dtype)
        decision, _ = pyoverdrive.explain("numpy.unique", arr)
        assert decision == "stock", (dtype, floor, decision)
        _assert_identical(np.unique(arr), STOCK_UNIQUE(arr))


def test_kwargs_fall_back_with_stock_semantics():
    arr = RNG.integers(0, 50, size=SIZE_THRESHOLD * 4, dtype=np.int64)
    got_vals, got_counts = np.unique(arr, return_counts=True)
    exp_vals, exp_counts = STOCK_UNIQUE(arr, return_counts=True)
    _assert_identical(got_vals, exp_vals)
    _assert_identical(got_counts, exp_counts)


def test_nan_float_falls_back_but_still_correct():
    arr = RNG.random(SIZE_THRESHOLD * 4)
    arr[::97] = np.nan
    assert pyoverdrive.explain("numpy.unique", arr)[0] == "stock"
    got, expected = np.unique(arr), STOCK_UNIQUE(arr)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


def test_subclasses_fall_back():
    class MyArray(np.ndarray):
        pass

    arr = RNG.integers(0, 50, size=SIZE_THRESHOLD * 4, dtype=np.int64).view(MyArray)
    assert pyoverdrive.explain("numpy.unique", arr)[0] == "stock"


# -- property-based sweep across both regimes -------------------------------

@given(data=st.data(), dtype=st.sampled_from([np.int32, np.int64, np.uint8]))
@settings(max_examples=120, deadline=None)
def test_property_matches_stock(data, dtype):
    shape = data.draw(hnp.array_shapes(min_dims=1, max_dims=2, max_side=64))
    arr = data.draw(hnp.arrays(dtype=dtype, shape=shape))
    _assert_identical(np.unique(arr), STOCK_UNIQUE(arr))
