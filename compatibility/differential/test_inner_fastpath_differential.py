"""Differential tests: inner_tensordot fast path vs stock np.inner.

Contract is NUMERIC equivalence (tight tolerance) where the path dispatches,
exact stock behavior everywhere else, and no dispatch at all for the 1-D/2-D
regimes where stock is already optimal.
"""

import numpy as np
import pytest

import pyoverdrive

STOCK_INNER = np.inner
RNG = np.random.default_rng(12778)


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(["numpy.inner"])
    yield
    pyoverdrive.disable()


def _tol(dtype):
    # float32 with zero-mean data suffers cancellation: a near-zero sum of 500
    # O(1) terms carries absolute error up to ~L*eps*scale ~ 1e-3 purely from
    # summation order, which is exactly the reordering this path performs.
    return dict(rtol=1e-9, atol=1e-12) if dtype == np.float64 else dict(
        rtol=1e-3, atol=1e-3
    )


DISPATCH_SHAPES = [
    ((3, 4, 8), (5, 8)),
    ((2, 2, 2, 6), (7, 6)),
    ((5, 8), (3, 4, 8)),          # high-ndim operand on the right
    ((3, 4, 8), (8,)),            # 3-D x 1-D
    ((6, 5, 500), (40, 500)),     # BLAS-sized contraction
    ((1, 1, 3), (2, 3)),          # degenerate leading dims
]


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=str)
@pytest.mark.parametrize("shapes", DISPATCH_SHAPES, ids=str)
def test_dispatched_numeric_equivalence(shapes, dtype):
    sa, sb = shapes
    a = RNG.standard_normal(sa).astype(dtype)
    b = RNG.standard_normal(sb).astype(dtype)
    assert pyoverdrive.explain("numpy.inner", a, b)[0] == "inner_tensordot"
    got = np.inner(a, b)
    expected = STOCK_INNER(a, b)
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, **_tol(dtype))


def test_low_ndim_regimes_never_dispatch():
    for sa, sb in [((500, 500), (500, 500)), ((1000,), (1000,)), ((7, 9), (9,))]:
        a = RNG.standard_normal(sa)
        b = RNG.standard_normal(sb)
        assert pyoverdrive.explain("numpy.inner", a, b)[0] == "stock"
        np.testing.assert_array_equal(np.inner(a, b), STOCK_INNER(a, b))


def test_non_float_and_mixed_dtypes_fall_back():
    a64 = RNG.integers(0, 100, size=(3, 4, 8))
    b64 = RNG.integers(0, 100, size=(5, 8))
    assert pyoverdrive.explain("numpy.inner", a64, b64)[0] == "stock"
    np.testing.assert_array_equal(np.inner(a64, b64), STOCK_INNER(a64, b64))

    af = RNG.standard_normal((3, 4, 8)).astype(np.float32)
    bf = RNG.standard_normal((5, 8))  # float64: mixed pair falls back
    assert pyoverdrive.explain("numpy.inner", af, bf)[0] == "stock"
    np.testing.assert_allclose(np.inner(af, bf), STOCK_INNER(af, bf), rtol=0)


def test_scalar_and_empty_fall_back():
    a = RNG.standard_normal((3, 4, 8))
    assert pyoverdrive.explain("numpy.inner", a, 2.0)[0] == "stock"
    np.testing.assert_allclose(np.inner(a, 2.0), STOCK_INNER(a, 2.0), rtol=0)

    empty = np.empty((2, 3, 0))
    other = np.empty((4, 0))
    assert pyoverdrive.explain("numpy.inner", empty, other)[0] == "stock"
    np.testing.assert_array_equal(
        np.inner(empty, other), STOCK_INNER(empty, other)
    )


def test_shape_mismatch_raises_like_stock():
    a = RNG.standard_normal((3, 4, 8))
    b = RNG.standard_normal((5, 7))
    assert pyoverdrive.explain("numpy.inner", a, b)[0] == "stock"
    with pytest.raises(ValueError):
        np.inner(a, b)


def test_strided_views_dispatch_and_match():
    a = RNG.standard_normal((6, 8, 16))[:, ::2, :]
    b = RNG.standard_normal((10, 32))[:, ::2]
    assert a.shape[-1] == b.shape[-1]
    assert pyoverdrive.explain("numpy.inner", a, b)[0] == "inner_tensordot"
    np.testing.assert_allclose(
        np.inner(a, b), STOCK_INNER(a, b), **_tol(np.float64)
    )
