"""Differential tests: unique_axis0_column fast path vs stock np.unique.

Contract is BIT-IDENTICAL output wherever the path dispatches (a plain 2-D
(n, 1) ndarray, n >= SIZE_THRESHOLD, dtype in int32/int64/uint32/uint64,
called as ``np.unique(a, axis=0)``), and correct fallback (stock result)
everywhere else.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import pyoverdrive
from pyoverdrive.fastpaths.unique_axis0_column import SIZE_THRESHOLD, _SUPPORTED_DTYPES

STOCK_UNIQUE = np.unique


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(["numpy.unique"])
    yield
    pyoverdrive.disable()


def _assert_identical(got, expected):
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


RNG = np.random.default_rng(20260823)


# -- regimes where the fast path DISPATCHES ---------------------------------

DISPATCH_COLUMNS = []
DISPATCH_IDS = []
for dtype in sorted(_SUPPORTED_DTYPES, key=str):
    info = np.iinfo(dtype)
    is_signed = info.min < 0
    for n in (SIZE_THRESHOLD, SIZE_THRESHOLD + 7):
        regimes = [
            ("high_card", RNG.integers(info.min, info.max, size=n, dtype=dtype)),
            ("dup_range10", RNG.integers(0, 10, size=n, dtype=dtype)),
            (
                "negatives" if is_signed else "small_range",
                RNG.integers(-50 if is_signed else 0, 51 if is_signed else 51, size=n, dtype=dtype),
            ),
            (
                "full_range",
                RNG.integers(info.min, info.max, size=n, dtype=dtype, endpoint=True),
            ),
        ]
        for regime, vals in regimes:
            DISPATCH_COLUMNS.append((dtype, n, regime, vals))
            DISPATCH_IDS.append(f"{dtype}-n{n}-{regime}")


@pytest.mark.parametrize("dtype,n,regime,vals", DISPATCH_COLUMNS, ids=DISPATCH_IDS)
def test_dispatched_bit_identical(dtype, n, regime, vals):
    a = vals.reshape(-1, 1)
    decision, reason = pyoverdrive.explain("numpy.unique", a, axis=0)
    assert decision == "unique_axis0_column", (dtype, n, regime, decision, reason)
    got = np.unique(a, axis=0)
    expected = STOCK_UNIQUE(a, axis=0)
    assert expected.shape[1] == 1
    _assert_identical(got, expected)


SMALL_DTYPES = (np.int8, np.uint8, np.int16, np.uint16)


@pytest.mark.parametrize("dtype", SMALL_DTYPES, ids=lambda d: str(np.dtype(d)))
def test_small_dtype_column_dispatched_bit_identical(dtype):
    info = np.iinfo(dtype)
    n = 2000
    a = RNG.integers(info.min, info.max, size=n, dtype=dtype, endpoint=True).reshape(-1, 1)
    decision, reason = pyoverdrive.explain("numpy.unique", a, axis=0)
    assert decision == "unique_axis0_column", (dtype, decision, reason)
    got = np.unique(a, axis=0)
    expected = STOCK_UNIQUE(a, axis=0)
    _assert_identical(got, expected)


def test_interplay_1d_uses_unique_sort_2d_column_uses_axis0():
    pyoverdrive.enable(["numpy.unique", "numpy.unique_values"])
    try:
        flat = RNG.integers(0, 1000, size=SIZE_THRESHOLD * 4, dtype=np.int64)
        assert pyoverdrive.explain("numpy.unique", flat)[0] == "unique_sort"

        column = flat.reshape(-1, 1)
        assert pyoverdrive.explain("numpy.unique", column, axis=0)[0] == "unique_axis0_column"
    finally:
        pyoverdrive.enable(["numpy.unique"])


# -- regimes that MUST fall back --------------------------------------------


def _base_column(dtype=np.int64, n=None):
    n = n or SIZE_THRESHOLD * 4
    return RNG.integers(0, 1000, size=n, dtype=dtype).reshape(-1, 1)


@pytest.mark.parametrize("axis", [1, -1])
def test_wrong_axis_value_falls_back(axis):
    a = _base_column()
    decision, _ = pyoverdrive.explain("numpy.unique", a, axis=axis)
    assert decision == "stock"
    _assert_identical(np.unique(a, axis=axis), STOCK_UNIQUE(a, axis=axis))


def test_axis_none_keyword_falls_back():
    a = _base_column()
    decision, _ = pyoverdrive.explain("numpy.unique", a, axis=None)
    assert decision == "stock"
    _assert_identical(np.unique(a, axis=None), STOCK_UNIQUE(a, axis=None))


def test_two_column_matrix_now_dispatches_to_lexsort_path():
    # historically this asserted a stock fallback; since OPP-000040 the
    # multi-column int case belongs to unique_rows_lexsort (its own
    # differential suite owns the details) - here we pin the handoff and
    # exact parity at this path boundary
    a = RNG.integers(0, 1000, size=(max(SIZE_THRESHOLD * 2, 1000), 2), dtype=np.int64)
    decision, _ = pyoverdrive.explain("numpy.unique", a, axis=0)
    assert decision == "unique_rows_lexsort"
    _assert_identical(np.unique(a, axis=0), STOCK_UNIQUE(a, axis=0))


def test_1d_with_axis0_falls_back():
    a = RNG.integers(0, 1000, size=SIZE_THRESHOLD * 4, dtype=np.int64)
    decision, _ = pyoverdrive.explain("numpy.unique", a, axis=0)
    assert decision == "stock"
    _assert_identical(np.unique(a, axis=0), STOCK_UNIQUE(a, axis=0))


def test_float64_column_falls_back():
    a = RNG.random(SIZE_THRESHOLD * 4).reshape(-1, 1)
    decision, _ = pyoverdrive.explain("numpy.unique", a, axis=0)
    assert decision == "stock"
    _assert_identical(np.unique(a, axis=0), STOCK_UNIQUE(a, axis=0))


def test_below_threshold_falls_back():
    a = _base_column(n=SIZE_THRESHOLD - 1)
    decision, _ = pyoverdrive.explain("numpy.unique", a, axis=0)
    assert decision == "stock"
    _assert_identical(np.unique(a, axis=0), STOCK_UNIQUE(a, axis=0))


def test_subclass_falls_back():
    class MyArray(np.ndarray):
        pass

    a = _base_column().view(MyArray)
    decision, _ = pyoverdrive.explain("numpy.unique", a, axis=0)
    assert decision == "stock"


def test_extra_kwarg_falls_back():
    a = _base_column()
    decision, _ = pyoverdrive.explain("numpy.unique", a, axis=0, return_index=True)
    assert decision == "stock"
    got_vals, got_idx = np.unique(a, axis=0, return_index=True)
    exp_vals, exp_idx = STOCK_UNIQUE(a, axis=0, return_index=True)
    _assert_identical(got_vals, exp_vals)
    _assert_identical(got_idx, exp_idx)


def test_fortran_ordered_column_matches_stock():
    a = np.asfortranarray(_base_column())
    decision, _ = pyoverdrive.explain("numpy.unique", a, axis=0)
    got = np.unique(a, axis=0)
    expected = STOCK_UNIQUE(a, axis=0)
    _assert_identical(got, expected)
    # a single column is F-contiguous and C-contiguous at once; whichever way
    # the predicate reads it, dispatch and result must agree with each other.
    if decision == "unique_axis0_column":
        assert a.flags["C_CONTIGUOUS"]


def test_noncontiguous_column_view_matches_stock():
    m = RNG.integers(0, 1000, size=(SIZE_THRESHOLD * 2 * 2, 4), dtype=np.int64)
    v = m[::2, :1]
    assert not v.flags["C_CONTIGUOUS"]
    decision, _ = pyoverdrive.explain("numpy.unique", v, axis=0)
    got = np.unique(v, axis=0)
    expected = STOCK_UNIQUE(v, axis=0)
    _assert_identical(got, expected)
    # the predicate's np.ndarray type-check may or may not accept a view;
    # whichever path runs, the output must be bit-identical to stock.
    assert decision in ("unique_axis0_column", "stock")


# -- property-based sweep across the dispatch regime -------------------------

@given(
    vals=st.lists(
        st.integers(min_value=-50, max_value=50),
        min_size=SIZE_THRESHOLD,
        max_size=SIZE_THRESHOLD + 200,
    )
)
@settings(max_examples=25, deadline=None)
def test_property_matches_stock(vals):
    a = np.array(vals, dtype=np.int64).reshape(-1, 1)
    got = np.unique(a, axis=0)
    expected = STOCK_UNIQUE(a, axis=0)
    _assert_identical(got, expected)
