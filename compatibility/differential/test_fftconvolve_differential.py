"""Differential tests: fftconvolve / fftcorrelate vs stock NumPy.

Contract (src/pyoverdrive/fastpaths/fftconvolve.py): applies to mode='full',
'same', and 'valid' (positional, keyword, or the per-function default -
convolve defaults to 'full', correlate to 'valid') on two plain 1-D ndarrays
of the same dtype from {float64, int64, int32}, with min(n, m) >= MIN_LEN
(1000 for all three dtypes) and the MODE's naive-work estimate at or above
its floor in _MODE_WORK_FLOOR (full/same: n*m; valid: (max-min+1)*min).
float64 additionally requires all-finite, non-overflowing operands and
compares numerically to stock; int64/int32 additionally require
max|a| * max|v| * min(n, m) <= the dtype's exactness bound and compare
bit-identically. Everything else falls back to stock, unchanged. Registered
path names are "fftconvolve" (numpy.convolve) and "fftcorrelate"
(numpy.correlate).
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

RNG = np.random.default_rng(20260823)


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(["numpy.convolve", "numpy.correlate"])
    yield
    pyoverdrive.disable()


def _stock(op, *args, **kwargs):
    return GEARBOX.stock_fn(op)(*args, **kwargs)


def _float_pair(n, m):
    return RNG.standard_normal(n), RNG.standard_normal(m)


def _int_pair(n, m, dtype):
    return (
        RNG.integers(-100, 101, size=n, dtype=dtype),
        RNG.integers(-100, 101, size=m, dtype=dtype),
    )


def _atol(a, v):
    # FFT rounding error scales with operand energy (the L2 norm), not with
    # any single element's magnitude: edge lags are single products and can
    # be tiny even when the arrays carry a lot of energy overall.
    return 1e-12 * float(np.linalg.norm(a)) * float(np.linalg.norm(v))


def _assert_close(got, stock, a, v):
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.allclose(got, stock, rtol=1e-9, atol=_atol(a, v))


def _expected_len(n, m, mode):
    if mode == "full":
        return n + m - 1
    if mode == "valid":
        return max(n, m) - min(n, m) + 1
    return max(n, m)  # 'same'


def _assert_refused(op, args, kwargs, equal_nan=False):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == "stock", (op, decision, reason)
    fn = np.convolve if op == "numpy.convolve" else np.correlate
    got = fn(*args, **kwargs)
    stock = _stock(op, *args, **kwargs)
    assert got.dtype == stock.dtype
    assert np.array_equal(got, stock, equal_nan=equal_nan)


# ---------------------------------------------------------------------------
# 1. dispatch + float64 numeric equality (mode='full')
# ---------------------------------------------------------------------------

# (5000, 300) is deliberately NOT here: min(5000, 300) = 300 < MIN_LEN(1000),
# so it is a refusal shape now, not a dispatch shape (see the "thin kernel"
# refusal case below).
SHAPES = [(1500, 1500), (5000, 1000), (1000, 5000)]

CONV_VARIANTS = [
    ("default", (), {}),
    ("mode_kwarg", (), {"mode": "full"}),
    ("mode_positional", ("full",), {}),
]
CORR_VARIANTS = [
    ("mode_kwarg", (), {"mode": "full"}),
    ("mode_positional", ("full",), {}),
]

FLOAT_DISPATCH_CASES = []
for n, m in SHAPES:
    for label, extra, kwargs in CONV_VARIANTS:
        FLOAT_DISPATCH_CASES.append(
            pytest.param(
                "convolve", "numpy.convolve", n, m, extra, kwargs,
                id=f"convolve-{label}-{n}x{m}",
            )
        )
    for label, extra, kwargs in CORR_VARIANTS:
        FLOAT_DISPATCH_CASES.append(
            pytest.param(
                "correlate", "numpy.correlate", n, m, extra, kwargs,
                id=f"correlate-{label}-{n}x{m}",
            )
        )


@pytest.mark.parametrize("kind, op, n, m, extra, kwargs", FLOAT_DISPATCH_CASES)
def test_dispatch_float64_numeric_equality(kind, op, n, m, extra, kwargs):
    a, v = _float_pair(n, m)
    decision, reason = GEARBOX.decide(op, (a, v) + extra, kwargs)
    expected = "fftconvolve" if kind == "convolve" else "fftcorrelate"
    assert decision == expected, (decision, reason)
    numpy_fn = np.convolve if kind == "convolve" else np.correlate
    got = numpy_fn(a, v, *extra, **kwargs)
    stock = _stock(op, a, v, *extra, **kwargs)
    _assert_close(got, stock, a, v)


# ---------------------------------------------------------------------------
# 2. dispatch + integer bit-identical (mode='full')
# ---------------------------------------------------------------------------

INT_DISPATCH_CASES = []
for dtype in (np.int64, np.int32):
    for n, m in SHAPES:
        for kind, op in (
            ("convolve", "numpy.convolve"),
            ("correlate", "numpy.correlate"),
        ):
            INT_DISPATCH_CASES.append(
                pytest.param(
                    dtype, n, m, kind, op, id=f"{np.dtype(dtype)}-{kind}-{n}x{m}"
                )
            )


@pytest.mark.parametrize("dtype, n, m, kind, op", INT_DISPATCH_CASES)
def test_dispatch_integer_bit_identical(dtype, n, m, kind, op):
    a, v = _int_pair(n, m, dtype)
    decision, reason = GEARBOX.decide(op, (a, v), {"mode": "full"})
    expected = "fftconvolve" if kind == "convolve" else "fftcorrelate"
    assert decision == expected, (decision, reason)
    numpy_fn = np.convolve if kind == "convolve" else np.correlate
    got = numpy_fn(a, v, mode="full")
    stock = _stock(op, a, v, mode="full")
    assert got.dtype == stock.dtype
    assert np.array_equal(got, stock)


# ---------------------------------------------------------------------------
# 3. correlate asymmetry: pins convolve(a, v[::-1]) against stock's own swap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", [(1500, 1700), (1700, 1500)])
def test_correlate_asymmetry_matches_stock_both_orders(n, m):
    a, v = _float_pair(n, m)
    decision, reason = GEARBOX.decide("numpy.correlate", (a, v), {"mode": "full"})
    assert decision == "fftcorrelate", (decision, reason)
    got = np.correlate(a, v, mode="full")
    stock = _stock("numpy.correlate", a, v, mode="full")
    _assert_close(got, stock, a, v)


# ---------------------------------------------------------------------------
# 4. dispatch + equality for mode='same' and mode='valid'
# ---------------------------------------------------------------------------

MODE_SHAPES = {
    "same": [(10000, 1000), (1000, 10000), (2000, 2000)],
    "valid": [(10000, 1000), (1000, 10000), (5000, 4000)],
}

MODE_DISPATCH_CASES = []
for mode, shapes in MODE_SHAPES.items():
    for n, m in shapes:
        for kind, op in (
            ("convolve", "numpy.convolve"),
            ("correlate", "numpy.correlate"),
        ):
            for style in ("positional", "kwarg"):
                MODE_DISPATCH_CASES.append(
                    pytest.param(
                        kind, op, n, m, mode, style,
                        id=f"{kind}-{mode}-{style}-{n}x{m}",
                    )
                )


@pytest.mark.parametrize("kind, op, n, m, mode, style", MODE_DISPATCH_CASES)
def test_dispatch_same_valid_float64_numeric_equality(kind, op, n, m, mode, style):
    a, v = _float_pair(n, m)
    if style == "positional":
        extra, kwargs = (mode,), {}
    else:
        extra, kwargs = (), {"mode": mode}
    decision, reason = GEARBOX.decide(op, (a, v) + extra, kwargs)
    expected = "fftconvolve" if kind == "convolve" else "fftcorrelate"
    assert decision == expected, (decision, reason)
    numpy_fn = np.convolve if kind == "convolve" else np.correlate
    got = numpy_fn(a, v, *extra, **kwargs)
    stock = _stock(op, a, v, *extra, **kwargs)
    _assert_close(got, stock, a, v)
    assert got.shape == (_expected_len(n, m, mode),)


def test_dispatch_correlate_no_mode_arg_defaults_to_valid_and_dispatches():
    # correlate's default mode is 'valid'; at this shape 'valid' now clears
    # its work floor, so the no-mode-argument call must dispatch too.
    a, v = _float_pair(10000, 1000)
    decision, reason = GEARBOX.decide("numpy.correlate", (a, v), {})
    assert decision == "fftcorrelate", (decision, reason)
    got = np.correlate(a, v)
    stock = _stock("numpy.correlate", a, v)
    _assert_close(got, stock, a, v)
    assert got.shape == (_expected_len(10000, 1000, "valid"),)


# ---------------------------------------------------------------------------
# 5. correlate 'same' centering trap: even min-length + second-operand-longer
# ---------------------------------------------------------------------------

CENTER_STOCK_SHAPES = [
    (10, 4), (4, 10), (9, 9), (8, 8), (4, 11), (3, 10), (12, 4), (4, 12),
]
CENTER_DISPATCH_SHAPES = [
    (10000, 1000), (1000, 10000), (2000, 2000), (1999, 2000), (2000, 1999),
]


@pytest.mark.parametrize("n, m", CENTER_STOCK_SHAPES)
def test_correlate_same_centering_matches_stock_below_floor(n, m):
    a, v = _float_pair(n, m)
    decision, reason = GEARBOX.decide("numpy.correlate", (a, v), {"mode": "same"})
    assert decision == "stock", (decision, reason)
    got = np.correlate(a, v, mode="same")
    stock = _stock("numpy.correlate", a, v, mode="same")
    assert got.shape == stock.shape
    assert np.allclose(got, stock, rtol=1e-9, atol=_atol(a, v))


@pytest.mark.parametrize("n, m", CENTER_DISPATCH_SHAPES)
def test_correlate_same_centering_matches_stock_dispatching(n, m):
    a, v = _float_pair(n, m)
    decision, reason = GEARBOX.decide("numpy.correlate", (a, v), {"mode": "same"})
    assert decision == "fftcorrelate", (decision, reason)
    got = np.correlate(a, v, mode="same")
    stock = _stock("numpy.correlate", a, v, mode="same")
    _assert_close(got, stock, a, v)
    assert got.shape == (_expected_len(n, m, "same"),)


# ---------------------------------------------------------------------------
# 6. mode='same'/'valid' integer bit-identical
# ---------------------------------------------------------------------------

INT_MODE_CASES = []
for dtype in (np.int64, np.int32):
    for mode in ("same", "valid"):
        for kind, op in (
            ("convolve", "numpy.convolve"),
            ("correlate", "numpy.correlate"),
        ):
            INT_MODE_CASES.append(
                pytest.param(dtype, mode, kind, op, id=f"{np.dtype(dtype)}-{mode}-{kind}")
            )


@pytest.mark.parametrize("dtype, mode, kind, op", INT_MODE_CASES)
def test_dispatch_same_valid_integer_bit_identical(dtype, mode, kind, op):
    a, v = _int_pair(10000, 1000, dtype)
    decision, reason = GEARBOX.decide(op, (a, v), {"mode": mode})
    expected = "fftconvolve" if kind == "convolve" else "fftcorrelate"
    assert decision == expected, (decision, reason)
    numpy_fn = np.convolve if kind == "convolve" else np.correlate
    got = numpy_fn(a, v, mode=mode)
    stock = _stock(op, a, v, mode=mode)
    assert got.dtype == stock.dtype
    assert np.array_equal(got, stock)
    assert got.shape == (_expected_len(10000, 1000, mode),)


# ---------------------------------------------------------------------------
# 7. refusal routes
# ---------------------------------------------------------------------------

def test_refusal_correlate_default_mode_is_valid():
    a, v = _float_pair(1500, 1500)
    _assert_refused("numpy.correlate", (a, v), {})


@pytest.mark.parametrize(
    "op, mode",
    [
        ("numpy.convolve", "same"),
        ("numpy.convolve", "valid"),
        ("numpy.correlate", "same"),
        ("numpy.correlate", "valid"),
    ],
)
def test_refusal_non_full_modes_below_work_floor(op, mode):
    # (1500, 1500) clears MIN_LEN but sits below both the 'same' (3e6) and
    # 'valid' (2e6) work floors: same=1500*1500=2_250_000, valid=1*1500=1500.
    a, v = _float_pair(1500, 1500)
    _assert_refused(op, (a, v), {"mode": mode})


@pytest.mark.parametrize("op", ["numpy.convolve", "numpy.correlate"])
def test_refusal_valid_near_equal_lengths(op):
    # work = (10000 - 9999 + 1) * 9999 = 19998, far below the 'valid' floor
    # of 2_000_000; stock computes 'valid' here in near-nothing since its
    # own cost is exactly that same near-equal-length product.
    a, v = _float_pair(10000, 9999)
    _assert_refused(op, (a, v), {"mode": "valid"})


@pytest.mark.parametrize("op", ["numpy.convolve", "numpy.correlate"])
def test_refusal_same_below_mode_floor(op):
    # work = 1000 * 1000 = 1_000_000, below the 'same' floor of 3_000_000,
    # even though min(n, m) = 1000 clears MIN_LEN on its own.
    a, v = _float_pair(1000, 1000)
    _assert_refused(op, (a, v), {"mode": "same"})


@pytest.mark.parametrize("op", ["numpy.convolve", "numpy.correlate"])
def test_refusal_unknown_mode_raises_like_stock(op):
    a, v = _float_pair(1500, 1500)
    decision, reason = GEARBOX.decide(op, (a, v), {"mode": "wrong"})
    assert decision == "stock", (decision, reason)
    fn = np.convolve if op == "numpy.convolve" else np.correlate
    with pytest.raises(Exception) as got_exc:
        fn(a, v, mode="wrong")
    with pytest.raises(Exception) as stock_exc:
        _stock(op, a, v, mode="wrong")
    assert type(got_exc.value) is type(stock_exc.value)


def test_refusal_float32_operands():
    a = RNG.standard_normal(1500).astype(np.float32)
    v = RNG.standard_normal(1500).astype(np.float32)
    _assert_refused("numpy.convolve", (a, v), {"mode": "full"})


def test_refusal_complex128_operands():
    a = RNG.standard_normal(1500) + 1j * RNG.standard_normal(1500)
    v = RNG.standard_normal(1500) + 1j * RNG.standard_normal(1500)
    _assert_refused("numpy.convolve", (a, v), {"mode": "full"})


def test_refusal_mixed_dtypes():
    a = RNG.standard_normal(1500)
    v = RNG.integers(-100, 101, size=1500, dtype=np.int64)
    _assert_refused("numpy.convolve", (a, v), {"mode": "full"})


def test_refusal_python_lists():
    a = list(RNG.standard_normal(1500))
    v = list(RNG.standard_normal(1500))
    _assert_refused("numpy.convolve", (a, v), {"mode": "full"})


def test_refusal_2d_arrays_raises_like_stock():
    a = RNG.standard_normal((1500, 2))
    v = RNG.standard_normal((1500, 2))
    decision, reason = GEARBOX.decide("numpy.convolve", (a, v), {"mode": "full"})
    assert decision == "stock", (decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.convolve(a, v, mode="full")
    with pytest.raises(Exception) as stock_exc:
        _stock("numpy.convolve", a, v, mode="full")
    assert type(got_exc.value) is type(stock_exc.value)


@pytest.mark.parametrize(
    "n, m",
    [(5000, 999), (5000, 300)],
    ids=["just-below-floor", "thin-kernel-huge-product"],
)
def test_refusal_below_min_len(n, m):
    # MIN_LEN is 1000 (FFTCONV-CAL calibration). A huge product does not
    # rescue a thin kernel: (5000, 300) has product 1_500_000, well past
    # PRODUCT_FLOOR, but min(n, m) = 300 still refuses it.
    a, v = _float_pair(n, m)
    _assert_refused("numpy.convolve", (a, v), {"mode": "full"})


def test_refusal_small_square_below_both_floors():
    a, v = _float_pair(400, 400)
    _assert_refused("numpy.convolve", (a, v), {"mode": "full"})


def test_refusal_float64_with_nan_localizes_exactly():
    a, v = _float_pair(1500, 1500)
    a = a.copy()
    a[750] = np.nan
    _assert_refused("numpy.convolve", (a, v), {"mode": "full"}, equal_nan=True)


def test_refusal_float64_with_inf():
    a, v = _float_pair(1500, 1500)
    a = a.copy()
    a[0] = np.inf
    _assert_refused("numpy.convolve", (a, v), {"mode": "full"})


def test_refusal_int64_exceeds_bound():
    n = 1500
    mag = 2**25
    a = np.full(n, mag, dtype=np.int64)
    v = np.full(n, mag, dtype=np.int64)
    bound = 2**52 - 1
    assert mag * mag * n > bound  # confirms this case sits outside the guarantee
    _assert_refused("numpy.correlate", (a, v), {"mode": "full"})


def test_refusal_int32_exceeds_bound_not_int64_bound():
    n = 1500
    mag = 2**12
    a = np.full(n, mag, dtype=np.int32)
    v = np.full(n, mag, dtype=np.int32)
    bound32 = min(2**52 - 1, 2**31 - 1)
    formula = mag * mag * n
    assert formula > bound32  # violates the int32-specific ceiling
    assert formula <= 2**52 - 1  # but comfortably under the int64 ceiling
    _assert_refused("numpy.convolve", (a, v), {"mode": "full"})


def test_kill_switch_restores_stock_routing():
    a, v = _float_pair(1500, 1500)
    decision, reason = GEARBOX.decide("numpy.convolve", (a, v), {"mode": "full"})
    assert decision == "fftconvolve", (decision, reason)
    pyoverdrive.disable_path("fftconvolve")
    try:
        decision, reason = GEARBOX.decide("numpy.convolve", (a, v), {"mode": "full"})
        assert decision == "stock", (decision, reason)
        got = np.convolve(a, v, mode="full")
        stock = _stock("numpy.convolve", a, v, mode="full")
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path("fftconvolve")


# ---------------------------------------------------------------------------
# 8. NaN-localization semantic witness
# ---------------------------------------------------------------------------

def test_nan_localization_witnesses_stock_not_fft():
    a, v = _float_pair(1500, 1500)
    a = a.copy()
    a[750] = np.nan
    decision, reason = GEARBOX.decide("numpy.convolve", (a, v), {"mode": "full"})
    assert decision == "stock", (decision, reason)
    got = np.convolve(a, v, mode="full")
    nan_mask = np.isnan(got)
    assert nan_mask.any()  # lags touching the NaN are contaminated
    assert not nan_mask.all()  # but naive summation keeps the damage local;
    # an FFT route would smear the NaN across every lag instead
