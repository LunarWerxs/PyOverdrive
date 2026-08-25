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


# Every shape here is INSIDE the measured regime (inner_tensordot._applicable:
# rows_a >= 8, rows_b >= 64, rows_a*rows_b >= 1024, k <= 128). The list used to
# hold shapes like (1, 1, 3) x (2, 3), which the path accepted and ran 2.6x
# SLOWER than stock - it had no size gate at all. Those moved to
# REFUSED_SHAPES below, where they assert the refusal instead.
DISPATCH_SHAPES = [
    ((4, 4, 8), (64, 8)),         # at the corner: rows_a 16, rows_b 64
    ((2, 2, 2, 6), (128, 6)),     # 4-D left operand
    ((64, 8), (8, 8, 8)),         # high-ndim operand on the RIGHT
    ((8, 8, 128), (64, 128)),     # at the contraction cap
    ((20, 5, 32), (256, 32)),     # comfortably inside
]

# Shapes the path must now decline. Each was measured dispatching into a loss
# (or sits outside the corner every measured cell won in).
REFUSED_SHAPES = [
    ((1, 1, 3), (2, 3)),          # degenerate leading dims: measured 0.30x
    ((3, 4, 8), (5, 8)),          # rows_b 5, far below the floor
    ((6, 5, 500), (40, 500)),     # k=500 above the cap; that corner is 0.38-0.62x
    ((3, 4, 8), (8,)),            # 3-D x 1-D: rows_b 1
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
    a = RNG.standard_normal((16, 8, 16))[:, ::2, :]
    b = RNG.standard_normal((128, 32))[:, ::2]
    assert a.shape[-1] == b.shape[-1]
    assert pyoverdrive.explain("numpy.inner", a, b)[0] == "inner_tensordot"
    np.testing.assert_allclose(
        np.inner(a, b), STOCK_INNER(a, b), **_tol(np.float64)
    )


@pytest.mark.parametrize("shapes", REFUSED_SHAPES, ids=str)
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=str)
def test_outside_the_measured_regime_stays_on_stock(shapes, dtype):
    """The path has a measured regime now, and outside it stock must answer.

    This is the test that would have caught the original defect: every one
    of these shapes was accepted before, and the smallest of them ran at
    0.30x - the fast path making the call three times slower. The selfcheck
    never saw it because its canonical input was the first shape in the
    sweep that happened to win.
    """
    sa, sb = shapes
    a = RNG.standard_normal(sa).astype(dtype)
    b = RNG.standard_normal(sb).astype(dtype)
    decision, reason = pyoverdrive.explain("numpy.inner", a, b)
    assert decision == "stock", (shapes, decision, reason)
    np.testing.assert_array_equal(np.inner(a, b), STOCK_INNER(a, b))
