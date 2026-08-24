"""Differential test: patched numpy.positive must match stock exactly.

Property-based over dtypes, shapes, strides, and values (spec section 9).
This is the template differential harness every future fast path copies.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pyoverdrive

STOCK_POSITIVE = np.positive


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(["numpy.positive"])
    pyoverdrive.enable_path("noop_positive")
    yield
    pyoverdrive.disable_path("noop_positive")
    pyoverdrive.disable()


DTYPES = st.sampled_from(
    [np.float64, np.float32, np.int64, np.int32, np.uint8, np.complex128]
)


@given(
    data=st.data(),
    dtype=DTYPES,
)
@settings(max_examples=150, deadline=None)
def test_positive_matches_stock(data, dtype):
    shape = data.draw(hnp.array_shapes(min_dims=0, max_dims=3, max_side=8))
    arr = data.draw(hnp.arrays(dtype=dtype, shape=shape))
    expected = STOCK_POSITIVE(arr)
    got = np.positive(arr)
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_strided_and_transposed_views_match():
    base = np.arange(64, dtype=np.float64).reshape(8, 8)
    for view in (base[::2, ::3], base.T, base[::-1], base[1:7, 2:5].T):
        np.testing.assert_array_equal(np.positive(view), STOCK_POSITIVE(view))


def test_kwargs_route_falls_back_and_matches():
    x = np.linspace(-3, 3, 17)
    out = np.empty_like(x)
    got = np.positive(x, out=out)
    assert got is out
    np.testing.assert_array_equal(out, STOCK_POSITIVE(x))
