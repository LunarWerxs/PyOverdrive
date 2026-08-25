"""Property-based differential net over the patched operations.

The hand-written differential suites pin the contracts each fast path
declares. This module is the complementary net: hypothesis generates
argument combinations the authors did NOT think of - sizes straddling
every calibrated floor, adversarial values (NaN, inf, dtype extremes,
ties, empty-ish arrays), argument spellings - and asserts the ONE
invariant every fast path shares:

    calling the patched public name gives what stock gives - the same
    values under the path's comparison discipline, the same dtype and
    shape, and an exception of the same type whenever stock raises.

It does not matter whether a given draw dispatches or refuses; both sides
of the predicate must be indistinguishable to the caller. That makes this
suite immune to threshold recalibration (no size here assumes a floor),
and it is exactly the kind of test that catches predicate holes, like a
dispatching input whose result type differs (the einsum 0-d-vs-scalar
case was found by a manual probe of this kind).

Runtime: sizes are kept modest (<= ~40k elements) so the full module runs
in seconds; the point is argument-space coverage, not benchmark scale.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

SETTINGS = dict(
    max_examples=60,
    deadline=None,  # the box is shared with real workloads; wall time lies
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable()
    yield
    pyoverdrive.disable()


def compare(op: str, args: tuple, kwargs: dict, equal) -> None:
    """Patched call vs stock call: same values (per `equal`), dtype, shape,
    or the same exception type."""
    stock = GEARBOX.stock_fn(op)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            expected = stock(*args, **kwargs)
            stock_exc = None
        except Exception as exc:  # noqa: BLE001 - parity is the contract
            expected, stock_exc = None, exc
        parts = op.split(".")
        assert parts[0] == "numpy"
        holder = np
        for p in parts[1:-1]:  # e.g. numpy.linalg.eigvalsh
            holder = getattr(holder, p)
        try:
            got = getattr(holder, parts[-1])(*args, **kwargs)
            got_exc = None
        except Exception as exc:  # noqa: BLE001
            got, got_exc = None, exc
    if stock_exc is not None or got_exc is not None:
        assert type(got_exc) is type(stock_exc), (
            f"exception mismatch: stock {stock_exc!r} vs patched {got_exc!r}"
        )
        return
    if isinstance(expected, tuple) or isinstance(got, tuple):
        assert isinstance(got, tuple) and isinstance(expected, tuple)
        assert len(got) == len(expected)
        for g_i, e_i in zip(got, expected):
            ge, ee = np.asarray(g_i), np.asarray(e_i)
            assert ge.shape == ee.shape and ge.dtype == ee.dtype
            assert equal(ge, ee), "tuple element values differ"
        return
    ge, ee = np.asarray(got), np.asarray(expected)
    assert ge.shape == ee.shape, f"shape {ge.shape} != {ee.shape}"
    assert ge.dtype == ee.dtype, f"dtype {ge.dtype} != {ee.dtype}"
    assert type(got) is type(expected), f"type {type(got)} != {type(expected)}"
    assert equal(ge, ee), "values differ"


def exact(g, e):
    return bool(np.array_equal(g, e, equal_nan=(g.dtype.kind == "f")))


def byte_exact(g, e):
    """Stricter than `exact`: compares the raw buffer.

    np.array_equal cannot distinguish +0.0 from -0.0. That is harmless for
    a path whose output values all come from the input, but not for one
    where the CALLER supplies a fill value - np.pad(a, 1,
    constant_values=-0.0) must come back with the sign bit set, and only a
    byte comparison proves it did. Object dtype is excluded because its
    buffer holds pointers rather than values.
    """
    if g.dtype != e.dtype or g.shape != e.shape:
        return False
    if g.dtype == object:
        # element-wise, and NaN-aware: two separately produced object
        # arrays hold DISTINCT float objects, and nan != nan, so a plain
        # list comparison reports a difference between identical results
        for x, y in zip(g.ravel().tolist(), e.ravel().tolist()):
            if x is y or x == y:
                continue
            if isinstance(x, float) and isinstance(y, float):
                if math.isnan(x) and math.isnan(y):
                    continue
            return False
        return True
    return g.tobytes() == e.tobytes()


def close_scaled(rtol, atol_frac):
    def eq(g, e):
        if g.dtype.kind != "f" and g.dtype.kind != "c":
            return bool(np.array_equal(g, e))
        scale = float(np.abs(e[np.isfinite(e)]).max()) if np.isfinite(e).any() else 1.0
        return bool(
            np.allclose(g, e, rtol=rtol, atol=atol_frac * max(1.0, scale), equal_nan=True)
        )

    return eq


finite_f64 = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False, width=64)
weird_f64 = st.floats(allow_nan=True, allow_infinity=True, width=64)


# --- convolve / correlate ---------------------------------------------------
@settings(**SETTINGS)
@given(
    a=arrays(np.float64, st.integers(0, 3000), elements=weird_f64),
    v=arrays(np.float64, st.integers(0, 1500), elements=weird_f64),
    mode=st.sampled_from(["full", "same", "valid", None]),
    data=st.data(),
)
def test_convolve_correlate_float64(a, v, mode, data):
    op = data.draw(st.sampled_from(["numpy.convolve", "numpy.correlate"]))
    args, kwargs = ((a, v), {}) if mode is None else ((a, v), {"mode": mode})

    # FFT rounding scales with the OPERAND NORMS, not the output magnitude:
    # hypothesis found draws where huge products cancel to near-zero lags,
    # which an output-scaled atol wrongly flags. (The absolute error bound
    # is ~eps * log2(fft_len) * ||a||_2 * ||v||_2 on both routes.)
    fin_a, fin_v = a[np.isfinite(a)], v[np.isfinite(v)]

    def _scaled_norm(x):
        """Overflow- and underflow-safe 2-norm, ||x|| == m * ||x/m||.

        np.linalg.norm squares first, so a 1e152-scale draw overflows the
        sum of squares to inf and a 1e-191-scale draw underflows it to 0.
        inf * 0 is nan, and a nan atol makes np.allclose reject arrays that
        are in fact bit-identical - the comparator fails, not the code under
        test. Hypothesis found precisely that pair (a ~4.2e+152 against
        v ~8.1e-191) and it looked like a convolve divergence for a while.
        """
        if x.size == 0:
            return 0.0
        m = float(np.abs(x).max())
        if m == 0.0:
            return 0.0
        return m * float(np.linalg.norm(x / m))

    norm_bound = _scaled_norm(fin_a) * _scaled_norm(fin_v)
    # clamp: the product can still overflow for genuinely huge draws, and
    # numpy rejects a non-finite atol; overflow-scale draws are refused by
    # the predicate anyway, so both routes are stock (exact) there
    if not np.isfinite(norm_bound):
        norm_bound = 1e280
    atol = min(1e-11 * (1.0 + norm_bound), 1e280)

    def eq(g, e):
        return bool(np.allclose(g, e, rtol=1e-9, atol=atol, equal_nan=True))

    compare(op, args, kwargs, eq)


@settings(**SETTINGS)
@given(
    a=arrays(np.int64, st.integers(1, 3000), elements=st.integers(-(2**40), 2**40)),
    v=arrays(np.int64, st.integers(1, 1500), elements=st.integers(-(2**40), 2**40)),
)
def test_convolve_int64_bit_identical(a, v):
    # values up to 2**40 cross the 2**52 exactness bound both ways at these
    # lengths, exercising dispatch AND refusal; equality must be exact
    compare("numpy.convolve", (a, v), {}, exact)


# --- nanquantile ------------------------------------------------------------
@settings(**SETTINGS)
@given(
    a=arrays(
        np.float64,
        st.tuples(st.integers(1, 60), st.integers(1, 60)),
        elements=st.floats(-1e3, 1e3, width=64) | st.just(float("nan")),
    ),
    q=st.one_of(st.floats(0, 1), st.sampled_from([0, 1, 0.5])),
    axis=st.sampled_from([0, 1, -1, -2]),
)
def test_nanquantile_2d(a, q, axis):
    compare("numpy.nanquantile", (a, q), {"axis": axis}, close_scaled(1e-9, 1e-12))


# --- einsum -----------------------------------------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(1, 130),
    m=st.integers(1, 130),
    k=st.integers(1, 130),
    subs=st.sampled_from(["ij,jk->ik", "ij,jk", "ij,ij->", "ij,ij->ij", "ij,kj->ik"]),
    dtype=st.sampled_from([np.float64, np.float32]),
    data=st.data(),
)
def test_einsum_two_operand(n, m, k, subs, dtype, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.uniform(0.5, 1.5, (n, m)).astype(dtype)
    b_shape = {"ij,jk->ik": (m, k), "ij,jk": (m, k), "ij,ij->": (n, m),
               "ij,ij->ij": (n, m), "ij,kj->ik": (k, m)}[subs]
    b = rng.uniform(0.5, 1.5, b_shape).astype(dtype)
    tol = close_scaled(1e-6, 1e-9) if dtype is np.float64 else close_scaled(1e-3, 1e-4)
    compare("numpy.einsum", (subs, a, b), {}, tol)


# --- searchsorted -----------------------------------------------------------
@settings(**SETTINGS)
@given(
    x=arrays(np.float64, st.integers(0, 15_000), elements=finite_f64),
    v=arrays(np.float64, st.integers(0, 40_000), elements=weird_f64),
    side=st.sampled_from(["left", "right", None]),
)
def test_searchsorted_float64(x, v, side):
    x = np.sort(x)
    kwargs = {} if side is None else {"side": side}
    compare("numpy.searchsorted", (x, v), kwargs, exact)


@settings(**SETTINGS)
@given(
    x=arrays(np.int64, st.integers(1, 15_000), elements=st.integers(-(2**60), 2**60)),
    v=arrays(np.int64, st.integers(1, 40_000), elements=st.integers(-(2**60), 2**60)),
    order=st.sampled_from(["random", "sorted", "descending"]),
)
def test_searchsorted_int64_orderings(x, v, order):
    x = np.sort(x)
    if order == "sorted":
        v = np.sort(v)
    elif order == "descending":
        v = np.sort(v)[::-1].copy()
    compare("numpy.searchsorted", (x, v), {}, exact)


# --- isclose ----------------------------------------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(0, 1_500),
    dtype=st.sampled_from([np.float64, np.float32]),
    tols=st.sampled_from([
        {}, {"rtol": 1e-3}, {"atol": 1e-6}, {"rtol": 0.0, "atol": 0.0},
        {"equal_nan": True}, {"atol": float("inf")}, {"rtol": float("nan")},
    ]),
    weird=st.booleans(),
    data=st.data(),
)
def test_isclose_pairs(n, dtype, tols, weird, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.uniform(-10, 10, n).astype(dtype)
    b = (a + rng.uniform(-1e-4, 1e-4, n)).astype(dtype)
    if weird and n:
        idx = rng.integers(0, n, size=max(1, n // 50))
        b[idx] = data.draw(st.sampled_from([np.nan, np.inf, -np.inf, 0.0]))
    compare("numpy.isclose", (a, b), dict(tols), exact)


@settings(**SETTINGS)
@given(
    a=st.one_of(finite_f64, st.integers(-(2**64), 2**64), weird_f64),
    b=st.one_of(finite_f64, st.integers(-(2**64), 2**64), weird_f64),
)
def test_isclose_scalars(a, b):
    def eq(g, e):
        return bool(np.array_equal(g, e))

    compare("numpy.isclose", (a, b), {}, eq)


# --- quantile with a q array -------------------------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(1, 3000),
    slices=st.integers(0, 3),
    nq=st.integers(1, 300),
    salt_nan=st.booleans(),
    bad_q=st.booleans(),
    data=st.data(),
)
def test_quantile_dense_q(n, slices, nq, salt_nan, bad_q, data):
    # n straddles the 512 reduced-length floor, nq straddles the 4-quantile
    # floor; bad_q draws exercise exception parity (stock raises on q > 1)
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    shape = (n,) if slices == 0 else (slices, n)
    a = rng.standard_normal(shape)
    if salt_nan and a.size:
        a.flat[rng.integers(0, a.size, size=max(1, a.size // 100))] = np.nan
    q = np.sort(rng.random(nq))
    if bad_q and nq:
        q[rng.integers(0, nq)] = 1.5
    kwargs = {} if slices == 0 else {"axis": -1}
    compare("numpy.quantile", (a, q), kwargs, exact)


# --- isin StringDType --------------------------------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(0, 2000),
    m=st.integers(0, 400),
    vocab=st.integers(1, 60),
    invert=st.booleans(),
    data=st.data(),
)
def test_isin_stringdtype(n, m, vocab, invert, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    words = np.array(
        ["", "\x00", "abc", "αβ", "long" * 50] + [f"w{i}" for i in range(vocab)],
        dtype=np.dtypes.StringDType(),
    )
    el = words[rng.integers(0, words.size, size=n)]
    te = words[rng.integers(0, words.size, size=m)]
    compare("numpy.isin", (el, te), {"invert": invert} if invert else {}, exact)


# --- dot real @ complex ------------------------------------------------------
@settings(**SETTINGS)
@given(
    rows=st.integers(1, 400),
    cols=st.integers(1, 200),
    data=st.data(),
)
def test_dot_real_complex(rows, cols, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    A = rng.standard_normal((rows, cols))
    b = rng.standard_normal(cols) + 1j * rng.standard_normal(cols)

    def eq(g, e):
        scale = max(1.0, float(np.abs(e).max()))
        return bool(np.allclose(g, e, rtol=1e-12, atol=1e-12 * scale))

    compare("numpy.dot", (A, b), {}, eq)


# --- intersect1d / unique ---------------------------------------------------
@settings(**SETTINGS)
@given(
    a=arrays(np.int64, st.integers(0, 2000), elements=st.integers(-500, 500)),
    b=arrays(np.int64, st.integers(0, 2000), elements=st.integers(-500, 500)),
    presort=st.booleans(),
)
def test_intersect1d_int64(a, b, presort):
    if presort:
        a, b = np.sort(a), np.sort(b)
    compare("numpy.intersect1d", (a, b), {}, exact)


@settings(**SETTINGS)
@given(
    a=arrays(np.int64, st.integers(0, 3000), elements=st.integers(-(2**62), 2**62)),
    kw=st.sampled_from([{}, {"return_counts": True}, {"axis": None}]),
)
def test_unique_int64(a, kw):
    def eq(g, e):
        return bool(np.array_equal(g, e))

    compare("numpy.unique", (a,), dict(kw), eq)


# --- small-int radix routes (OPP-000010) ------------------------------------
@settings(**SETTINGS)
@given(
    dtype=st.sampled_from([np.int8, np.uint8, np.int16, np.uint16]),
    n=st.integers(0, 15_000),
    data=st.data(),
)
def test_unique_small_ints(dtype, n, data):
    # sizes straddle every small-dtype floor (1000 / 10_000)
    info = np.iinfo(dtype)
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    lo, hi = data.draw(st.sampled_from([(0, 10), (info.min, info.max + 1)]))
    a = rng.integers(lo, max(lo + 1, hi), size=n, dtype=dtype)
    compare("numpy.unique", (a,), {}, lambda g, e: bool(np.array_equal(g, e)))


@settings(**SETTINGS)
@given(
    dtype=st.sampled_from([np.int8, np.uint8, np.int16, np.uint16]),
    n=st.integers(0, 8_000),
    m=st.integers(0, 8_000),
    data=st.data(),
)
def test_intersect_small_ints(dtype, n, m, data):
    info = np.iinfo(dtype)
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.integers(info.min, info.max + 1, size=n, dtype=dtype)
    b = rng.integers(info.min, info.max + 1, size=m, dtype=dtype)
    compare("numpy.intersect1d", (a, b), {}, exact)


# --- single-char sort/unique via int view (OPP-000024) ----------------------
_CODEPOINT = st.integers(0, 0x10FFFF).filter(lambda c: not 0xD800 <= c <= 0xDFFF)


@settings(**SETTINGS)
@given(
    kind=st.sampled_from(["U1", "S1"]),
    n=st.integers(0, 15_000),
    op=st.sampled_from(["sort", "unique"]),
    data=st.data(),
)
def test_char_view_sort_unique(kind, n, op, data):
    # sizes straddle every floor (300 / 1000 / 10_000); alphabets include
    # codepoint 0 (the ''-equivalence), > 0x7F, and the 0x10FFFF max
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    if kind == "U1":
        codes = data.draw(st.lists(_CODEPOINT, min_size=1, max_size=20))
        alphabet = np.array([chr(c) for c in codes], dtype="U1")
    else:
        codes = data.draw(st.lists(st.integers(0, 255), min_size=1, max_size=20))
        alphabet = np.frombuffer(bytes(codes), dtype="S1")
    a = alphabet[rng.integers(0, len(alphabet), size=n)]
    if op == "sort":
        kwargs = data.draw(
            st.sampled_from([{}, {"axis": -1}, {"axis": 0}, {"kind": "stable"}])
        )
        compare("numpy.sort", (a,), kwargs, exact)
    else:
        flags = {
            name: data.draw(st.booleans())
            for name in ("return_index", "return_inverse", "return_counts")
        }
        compare("numpy.unique", (a,), flags, exact)


# --- tiny-trailing-axis reductions (OPP-000026) -----------------------------
@settings(**SETTINGS)
@given(
    op=st.sampled_from(["mean", "sum"]),
    dtype=st.sampled_from([np.float64, np.float32]),
    rows=st.integers(1, 12_000),
    k=st.integers(1, 7),
    make_3d=st.booleans(),
    axis_style=st.sampled_from(["int0", "neg", "tuple", "revtuple", "none"]),
    salt=st.sampled_from([None, "nan", "inf", "mixinf"]),
    data=st.data(),
)
def test_reduce_tiny_trailing(op, dtype, rows, k, make_3d, axis_style, salt, data):
    # rows straddles ROWS_MIN, k straddles [K_MIN, K_MAX]; both dispatch and
    # refusal draws must be caller-indistinguishable from stock
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.uniform(0.5, 1.5, size=(rows, k)).astype(dtype)
    if salt and rows >= 2:
        col = int(rng.integers(0, k))
        if salt == "nan":
            a[rng.integers(0, rows), col] = np.nan
        elif salt == "inf":
            a[rng.integers(0, rows), col] = np.inf
        else:
            a[0, col], a[1, col] = np.inf, -np.inf
    if make_3d and rows % 2 == 0 and rows > 0:
        a = np.ascontiguousarray(a.reshape(2, rows // 2, k))
        axis = {
            "int0": 0,  # refusal on 3-D: keeps two axes
            "neg": (-3, -2),
            "tuple": (0, 1),
            "revtuple": (1, 0),
            "none": None,
        }[axis_style]
    else:
        axis = {"int0": 0, "neg": -2, "tuple": (0,), "revtuple": (-2,), "none": None}[
            axis_style
        ]
    tol = (1e-9, 1e-12) if dtype is np.float64 else (1e-3, 1e-6)
    compare(f"numpy.{op}", (a,), {"axis": axis}, close_scaled(*tol))


# --- batched 2x2 eigvalsh closed form (OPP-000030) ---------------------------
@settings(**SETTINGS)
@given(
    batch=st.integers(1, 300),
    dtype=st.sampled_from([np.float64, np.float32]),
    uplo=st.sampled_from([None, "L", "U"]),
    lead_4d=st.booleans(),
    garbage_upper=st.booleans(),
    salt_nan=st.booleans(),
    data=st.data(),
)
def test_eigvalsh_2x2(batch, dtype, uplo, lead_4d, garbage_upper, salt_nan, data):
    # batch straddles BATCH_MIN; UPLO='U' and NaN must refuse transparently;
    # asymmetric upper garbage must NOT change a dispatching result (both
    # routes read only the lower triangle under UPLO='L')
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.uniform(-1.0, 1.0, size=(batch, 2, 2)).astype(dtype)
    a = np.ascontiguousarray(a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(2, dtype=dtype))
    if garbage_upper:
        a[..., 0, 1] = rng.uniform(-9.0, 9.0, size=batch).astype(dtype)
    if salt_nan and batch >= 1:
        a[rng.integers(0, batch), 1, 0] = np.nan
    if lead_4d and batch % 2 == 0 and batch > 0:
        a = a.reshape(2, batch // 2, 2, 2)
    kwargs = {} if uplo is None else {"UPLO": uplo}
    tol = (1e-9, 1e-12) if dtype is np.float64 else (1e-3, 1e-6)
    compare("numpy.linalg.eigvalsh", (a,), kwargs, close_scaled(*tol))


# --- complex-by-real matmul split (OPP-000029) -------------------------------
@settings(**SETTINGS)
@given(
    m=st.integers(1, 300),
    n=st.sampled_from([64, 999, 1000, 1100]),
    q=st.sampled_from([64, 499, 500, 620]),
    pair=st.sampled_from(["c128f64", "c64f32", "c128f32", "c64f64"]),
    salt=st.sampled_from([None, "nan_c", "inf_r"]),
    data=st.data(),
)
def test_matmul_complex_real(m, n, q, pair, salt, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    cdt = np.complex128 if pair.startswith("c128") else np.complex64
    rdt = np.float64 if pair.endswith("f64") else np.float32
    c = (
        rng.uniform(0.5, 1.5, size=(m, n)) + 1j * rng.uniform(0.5, 1.5, size=(m, n))
    ).astype(cdt)
    r = rng.uniform(0.5, 1.5, size=(n, q)).astype(rdt)
    if salt == "nan_c":
        c[rng.integers(0, m), rng.integers(0, n)] = np.nan
    elif salt == "inf_r":
        r[rng.integers(0, n), rng.integers(0, q)] = np.inf
    rtol = (1e-9, 1e-12) if cdt is np.complex128 else (1e-4, 1e-6)
    compare("numpy.matmul", (c, r), {}, close_scaled(*rtol))


# --- small 1-D roll via concatenate (OPP-000032) -----------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(0, 12_000),
    dtype=st.sampled_from([np.int64, np.float64, np.int32, np.float32, np.bool_, np.uint8]),
    shift=st.one_of(
        st.integers(-30_000, 30_000),
        st.sampled_from([(1,), (0, 1), True, 1.5]),
    ),
    use_kw=st.booleans(),
    data=st.data(),
)
def test_roll_1d(n, dtype, shift, use_kw, data):
    # n straddles the cap and includes 0; shift space includes negatives,
    # overshoots, tuples, bool, float - refusals must be indistinguishable
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    if dtype is np.bool_:
        a = rng.integers(0, 2, size=n).astype(bool)
    elif np.issubdtype(dtype, np.integer):
        a = rng.integers(0, 100, size=n).astype(dtype)
    else:
        a = rng.random(n).astype(dtype)
    if use_kw:
        compare("numpy.roll", (a,), {"shift": shift}, exact)
    else:
        compare("numpy.roll", (a, shift), {}, exact)


# NOTE: an argmax_blocked_transpose path (OPP-000034) was built and fully
# tested here, then UNSHIPPED before release: its Intel-calibrated regime
# (2.2-4x from (3000, 3000) up) measured as a 0.75-0.84x REGRESSION on the
# Zen 4 dev box, whose stock strided argmax is ~2.3x faster than Intel's.
# It later returned CALIBRATION-GATED (off unless a per-machine probe
# enables it); its correctness is pinned by its differential suite.


# --- percentile sibling of the dense-quantile route (OPP-000022) -------------
@settings(**SETTINGS)
@given(
    n=st.integers(1, 4_000),
    nq=st.integers(1, 40),
    hi=st.sampled_from([100.0, 100.0, 101.0]),  # occasional out-of-domain q
    data=st.data(),
)
def test_percentile_dense(n, nq, hi, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.standard_normal(n)
    q = np.linspace(0.0, hi, nq)
    compare("numpy.percentile", (a, q), {}, exact)


# --- batched small-matrix inverse (OPP-000035) -------------------------------
@settings(**SETTINGS)
@given(
    batch=st.integers(1, 1_500),
    d=st.sampled_from([2, 3, 4]),
    dtype=st.sampled_from([np.float64, np.float32]),
    salt=st.sampled_from([None, "nan", "zero_matrix", "near_singular"]),
    data=st.data(),
)
def test_inv_small_batch(batch, d, dtype, salt, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.uniform(-1.0, 1.0, size=(batch, d, d)).astype(dtype)
    a = np.ascontiguousarray(a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(d, dtype=dtype))
    if salt == "nan":
        a[rng.integers(0, batch), 0, 0] = np.nan
    elif salt == "zero_matrix":
        a[rng.integers(0, batch)] = 0.0
    elif salt == "near_singular":
        i = rng.integers(0, batch)
        a[i] = np.outer(np.arange(1, d + 1), np.arange(1, d + 1)).astype(dtype)
    tol = (1e-9, 1e-12) if dtype is np.float64 else (1e-3, 1e-5)
    compare("numpy.linalg.inv", (a,), {}, close_scaled(*tol))


# --- isin on object arrays (OPP-000036) --------------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(0, 2_000),
    m=st.integers(0, 300),
    pool=st.sampled_from(["str", "int", "mixed", "nan_salted", "unhashable_salted"]),
    invert=st.booleans(),
    data=st.data(),
)
def test_isin_object(n, m, pool, invert, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    base = [f"s{i}" for i in range(40)] + list(range(40)) + [1.5, True, ("t", 1)]
    if pool == "nan_salted":
        base = base + [float("nan")]
    elif pool == "unhashable_salted":
        base = base + [[1, 2]]
    elif pool == "str":
        base = [f"s{i}" for i in range(60)]
    elif pool == "int":
        base = list(range(60))
    el = np.array([base[i] for i in rng.integers(0, len(base), size=n)], dtype=object)
    te = np.array([base[i] for i in rng.integers(0, len(base), size=m)], dtype=object)
    compare("numpy.isin", (el, te), {"invert": invert}, exact)


# --- median via partition (OPP-000037) ---------------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(0, 6_000),
    dtype=st.sampled_from([np.float64, np.float32, np.int64]),
    axis=st.sampled_from([None, 0]),
    salt_nan=st.booleans(),
    data=st.data(),
)
def test_median_partition(n, dtype, axis, salt_nan, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    if np.issubdtype(dtype, np.integer):
        a = rng.integers(-100, 100, size=n).astype(dtype)
    else:
        a = rng.random(n).astype(dtype)
    if salt_nan and n and not np.issubdtype(dtype, np.integer):
        a[rng.integers(0, n)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # empty-array median warns on both routes
        compare("numpy.median", (a,), {"axis": axis}, exact)


# --- uniform-bin histogram2d (OPP-000038) ------------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(0, 5_000),
    bx=st.sampled_from([5, 30, 40]),
    by=st.sampled_from([5, 30, 41]),
    weighted=st.booleans(),
    edge_heavy=st.booleans(),
    data=st.data(),
)
def test_hist2d_uniform(n, bx, by, weighted, edge_heavy, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    if edge_heavy and n:
        ex = np.linspace(-3.0, 3.0, bx + 1)
        x = ex[rng.integers(0, bx + 1, size=n)]
        y = rng.normal(0.0, 2.0, size=n)  # includes out-of-range tails
    else:
        x = rng.normal(0.0, 2.0, size=n)
        y = rng.normal(0.0, 2.0, size=n)
    kwargs = {"bins": [bx, by], "range": [[-3.0, 3.0], [-3.0, 3.0]]}
    if weighted:
        kwargs["weights"] = rng.random(n)
    compare("numpy.histogram2d", (x, y), kwargs, exact)


# --- unique(axis=0) int rows (OPP-000040) ------------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(0, 3_000),
    k=st.integers(1, 10),
    dtype=st.sampled_from([np.int64, np.int32, np.float64]),
    counts=st.booleans(),
    axis=st.sampled_from([0, None, 1]),
    data=st.data(),
)
def test_unique_rows(n, k, dtype, counts, axis, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    if np.issubdtype(dtype, np.integer):
        a = rng.integers(-50, 50, size=(n, k)).astype(dtype)
    else:
        a = rng.random(size=(n, k)).astype(dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compare("numpy.unique", (a,), {"axis": axis, "return_counts": counts}, exact)


# --- searchsorted out-of-range python-int key (OPP-000039) -------------------
@settings(**SETTINGS)
@given(
    n=st.integers(0, 500),
    dtype=st.sampled_from([np.int64, np.int32, np.int8, np.uint64, np.uint8]),
    side=st.sampled_from(["left", "right"]),
    key_kind=st.sampled_from(["far_above", "far_below", "just_above", "just_below", "at_max", "in_range"]),
    data=st.data(),
)
def test_searchsorted_extreme_key(n, dtype, side, key_kind, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    info = np.iinfo(dtype)
    a = np.sort(rng.integers(info.min, info.max, size=n, dtype=dtype))
    key = {
        "far_above": int(info.max) + 2**20,
        "far_below": int(info.min) - 2**20,
        "just_above": int(info.max) + 1,
        "just_below": int(info.min) - 1,
        "at_max": int(info.max),
        "in_range": int(info.max) // 2,
    }[key_kind]
    compare("numpy.searchsorted", (a, key), {"side": side}, exact)


# --- nan-family scan routes (OPP-000041/42/43) ------------------------------
@settings(**SETTINGS)
@given(
    shape=st.one_of(
        st.integers(1, 2000),
        st.tuples(st.integers(1, 60), st.integers(1, 50)),
    ),
    salt=st.sampled_from(["none", "some", "all", "one_slice"]),
    axis_style=st.sampled_from(["none", "kw", "pos", "neg"]),
    data=st.data(),
)
def test_nan_reductions_scan(shape, salt, axis_style, data):
    op = data.draw(st.sampled_from(
        ["numpy.nanmean", "numpy.nansum", "numpy.nanstd", "numpy.nanvar"]
    ))
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.standard_normal(shape)
    if salt == "some":
        a[rng.random(np.shape(a)) < 0.1] = np.nan
    elif salt == "all":
        a[...] = np.nan
    elif salt == "one_slice" and a.ndim == 2:
        a[0, :] = np.nan
    if axis_style == "none" or a.ndim == 1 and axis_style == "neg":
        args, kwargs = (a,), {}
    elif axis_style == "kw":
        args, kwargs = (a,), {"axis": data.draw(st.integers(0, a.ndim - 1))}
    elif axis_style == "pos":
        args, kwargs = (a, data.draw(st.integers(0, a.ndim - 1))), {}
    else:
        args, kwargs = (a,), {"axis": -1}
    compare(op, args, kwargs, exact)


@settings(**SETTINGS)
@given(
    shape=st.one_of(
        st.integers(1, 2000),
        st.tuples(st.integers(1, 60), st.integers(1, 50)),
    ),
    salt=st.sampled_from(["none", "some", "all", "one_slice"]),
    use_axis=st.booleans(),
    data=st.data(),
)
def test_nanargminmax_scan(shape, salt, use_axis, data):
    op = data.draw(st.sampled_from(["numpy.nanargmax", "numpy.nanargmin"]))
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.standard_normal(shape)
    if salt == "some":
        a[rng.random(np.shape(a)) < 0.1] = np.nan
    elif salt == "all":
        a[...] = np.nan
    elif salt == "one_slice" and a.ndim == 2:
        a[0, :] = np.nan
    kwargs = {"axis": data.draw(st.integers(-a.ndim, a.ndim - 1))} if use_axis else {}
    compare(op, (a,), kwargs, exact)


@settings(**SETTINGS)
@given(
    rows=st.integers(1, 700),
    cols=st.integers(1, 400),
    salt=st.sampled_from(["none", "some", "one_row"]),
    axis=st.sampled_from([0, 1, -1, None]),
    data=st.data(),
)
def test_nanmedian_scan(rows, cols, salt, axis, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.standard_normal((rows, cols))
    if salt == "some":
        a[rng.random((rows, cols)) < 0.05] = np.nan
    elif salt == "one_row":
        a[0, :] = np.nan
    args, kwargs = ((a,), {}) if axis is None else ((a,), {"axis": axis})
    compare("numpy.nanmedian", args, kwargs, exact)


# --- integer matmul via exact float64 BLAS (OPP-000044) ---------------------
@settings(**SETTINGS)
@given(
    m=st.integers(1, 80),
    k=st.integers(1, 80),
    n=st.integers(1, 80),
    dtype=st.sampled_from([np.int64, np.int32]),
    mag_exp=st.integers(0, 30),
    data=st.data(),
)
def test_matmul_int_blas(m, k, n, dtype, mag_exp, data):
    op = data.draw(st.sampled_from(["numpy.matmul", "numpy.dot"]))
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    info = np.iinfo(dtype)
    hi = min(2**mag_exp, int(info.max) // 2)
    x = rng.integers(-hi, hi + 1, (m, k)).astype(dtype)
    y = rng.integers(-hi, hi + 1, (k, n)).astype(dtype)
    compare(op, (x, y), {}, exact)


# --- small-batch det/slogdet/solve closed forms (OPP-000045) ----------------
@settings(**SETTINGS)
@given(
    batch=st.integers(1, 1500),
    d=st.sampled_from([2, 3]),
    hazard=st.sampled_from(["clean", "singular", "near_singular", "nonfinite"]),
    data=st.data(),
)
def test_linalg_small_batch(batch, d, hazard, data):
    op = data.draw(st.sampled_from(
        ["numpy.linalg.det", "numpy.linalg.slogdet", "numpy.linalg.solve"]
    ))
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.standard_normal((batch, d, d))
    if hazard == "singular":
        a[batch // 2, -1, :] = a[batch // 2, 0, :]
    elif hazard == "near_singular":
        a[batch // 2, -1, :] = a[batch // 2, 0, :] * (1.0 + 1e-13)
    elif hazard == "nonfinite":
        a[batch // 2, 0, 0] = data.draw(st.sampled_from([np.nan, np.inf, -np.inf]))
    if op == "numpy.linalg.solve":
        b = rng.standard_normal((batch, d, 1))
        compare(op, (a, b), {}, close_scaled(1e-9, 1e-12))
    else:
        compare(op, (a,), {}, close_scaled(1e-9, 1e-12))


# --- nan_to_num where-route (OPP-000046) ------------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(0, 20000),
    salt=st.sampled_from(["clean", "nan", "inf", "mix", "all_special"]),
    data=st.data(),
)
def test_nan_to_num_where(n, salt, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.standard_normal(n)
    if salt in ("nan", "mix") and n:
        a[rng.random(n) < 0.05] = np.nan
    if salt in ("inf", "mix") and n:
        a[rng.random(n) < 0.05] = np.inf
        a[rng.random(n) < 0.05] = -np.inf
    if salt == "all_special" and n:
        a[:] = np.nan
        a[: n // 2] = np.inf
    compare("numpy.nan_to_num", (a,), {}, exact)


# --- nan_to_num scalar overrides (OPP-000046 extension) ---------------------
@settings(**SETTINGS)
@given(
    n=st.integers(0, 20000),
    salt=st.sampled_from(["clean", "nan", "mix"]),
    nan_fill=st.sampled_from([None, 0.0, 1.5, -2, True, np.inf, np.nan]),
    posinf_fill=st.sampled_from([None, 100.0, -1e300]),
    neginf_fill=st.sampled_from([None, -7.0]),
    data=st.data(),
)
def test_nan_to_num_overrides(n, salt, nan_fill, posinf_fill, neginf_fill, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.standard_normal(n)
    if salt in ("nan", "mix") and n:
        a[rng.random(n) < 0.05] = np.nan
    if salt == "mix" and n:
        a[rng.random(n) < 0.05] = np.inf
        a[rng.random(n) < 0.05] = -np.inf
    kwargs = {}
    if nan_fill is not None:
        kwargs["nan"] = nan_fill
    if posinf_fill is not None:
        kwargs["posinf"] = posinf_fill
    if neginf_fill is not None:
        kwargs["neginf"] = neginf_fill
    compare("numpy.nan_to_num", (a,), kwargs, exact)


# --- batched 2x2/3x3 cholesky closed form (OPP-000047) ----------------------
@settings(**SETTINGS)
@given(
    batch=st.integers(1, 1500),
    d=st.sampled_from([2, 3]),
    dtype=st.sampled_from([np.float64, np.float32]),
    hazard=st.sampled_from(["clean", "non_pd", "nonfinite", "garbage_upper"]),
    data=st.data(),
)
def test_cholesky_small_batch(batch, d, dtype, hazard, data):
    # non-PD and non-finite must refuse transparently (LinAlgError parity for
    # non-PD); asymmetric upper-triangle garbage must NOT change a dispatching
    # result (both routes read only the lower triangle)
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    m = rng.standard_normal((batch, d, d))
    a = np.ascontiguousarray(m @ np.swapaxes(m, -1, -2) + d * np.eye(d)).astype(dtype)
    if hazard == "non_pd":
        a[batch // 2] = -np.eye(d, dtype=dtype)
    elif hazard == "nonfinite":
        a[batch // 2, 0, 0] = data.draw(st.sampled_from([np.nan, np.inf, -np.inf]))
    elif hazard == "garbage_upper":
        a[..., 0, -1] = rng.uniform(-9.0, 9.0, size=batch).astype(dtype)
    tol = (1e-9, 1e-12) if dtype is np.float64 else (1e-3, 1e-6)
    compare("numpy.linalg.cholesky", (a,), {}, close_scaled(*tol))


# --- batched 3x3 eigvalsh trig closed form (OPP-000048) ---------------------
@settings(**SETTINGS)
@given(
    batch=st.integers(1, 300),
    dtype=st.sampled_from([np.float64, np.float32]),
    uplo=st.sampled_from([None, "L", "U"]),
    lead_4d=st.booleans(),
    garbage_upper=st.booleans(),
    hazard=st.sampled_from(
        ["clean", "nan", "identity_multiple", "clustered", "degen_mix"]
    ),
    data=st.data(),
)
def test_eigvalsh_3x3(batch, dtype, uplo, lead_4d, garbage_upper, hazard, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.uniform(-1.0, 1.0, size=(batch, 3, 3)).astype(dtype)
    a = np.ascontiguousarray(
        (a + np.swapaxes(a, -1, -2)) / dtype(2.0)
    )
    if hazard == "identity_multiple":
        a[...] = dtype(2.5) * np.eye(3, dtype=dtype)
    elif hazard == "clustered":
        base = np.diag(np.array([1.0, 1.0 + 1e-9, 5.0], dtype=dtype))
        a[...] = base
    elif hazard == "degen_mix":
        # exactly repeated pairs scattered at a drawn fraction: exercises
        # the batch-9 split-and-recombine below DEGEN_FRAC_MAX and the
        # whole-stack stock bail above it
        frac = data.draw(st.sampled_from([0.01, 0.2, 0.6]))
        nbad = min(batch, max(1, int(frac * batch)))
        idx = rng.choice(batch, nbad, replace=False)
        a[idx] = np.diag(np.array([1.0, 1.0, 5.0], dtype=dtype))
    if garbage_upper:
        a[..., 0, 2] = rng.uniform(-9.0, 9.0, size=batch).astype(dtype)
    if hazard == "nan" and batch >= 1:
        a[rng.integers(0, batch), 2, 0] = np.nan
    if lead_4d and batch % 2 == 0 and batch > 0:
        a = a.reshape(2, batch // 2, 3, 3)
    kwargs = {} if uplo is None else {"UPLO": uplo}
    tol = (1e-9, 1e-12) if dtype is np.float64 else (1e-3, 1e-6)
    compare("numpy.linalg.eigvalsh", (a,), kwargs, close_scaled(*tol))


# --- einsum >=3-operand chain (OPP-000049) ----------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(1, 24),
    m=st.integers(1, 24),
    k=st.integers(1, 24),
    l=st.integers(1, 24),
    subs=st.sampled_from(["ij,jk,kl->il", "ij,jk,kl->", "ij,jk,kl", "ij,jk,kl,lm->im"]),
    dtype=st.sampled_from([np.float64, np.float32]),
    data=st.data(),
)
def test_einsum_chain(n, m, k, l, subs, dtype, data):
    # volumes straddle CHAIN_VOLUME_FLOOR both ways; the 4-operand form
    # keeps dims small so the stock naive loop stays affordable
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    if subs == "ij,jk,kl,lm->im":
        n, m, k, l = min(n, 12), min(m, 12), min(k, 12), min(l, 12)
        ops = [
            rng.uniform(0.5, 1.5, (n, m)).astype(dtype),
            rng.uniform(0.5, 1.5, (m, k)).astype(dtype),
            rng.uniform(0.5, 1.5, (k, l)).astype(dtype),
            rng.uniform(0.5, 1.5, (l, n)).astype(dtype),
        ]
    else:
        ops = [
            rng.uniform(0.5, 1.5, (n, m)).astype(dtype),
            rng.uniform(0.5, 1.5, (m, k)).astype(dtype),
            rng.uniform(0.5, 1.5, (k, l)).astype(dtype),
        ]
    tol = close_scaled(1e-6, 1e-9) if dtype is np.float64 else close_scaled(1e-3, 1e-4)
    compare("numpy.einsum", (subs, *ops), {}, tol)


# --- einsum ellipsis spellings (batch 9: OPP-000018/000049 extension) --------
@settings(**SETTINGS)
@given(
    b=st.integers(1, 6),
    n=st.integers(1, 20),
    m=st.integers(1, 20),
    k=st.integers(1, 20),
    subs=st.sampled_from(
        [
            "...ij,...jk->...ik",
            "...ij,...jk",
            "...ij,...jk,...kl->...il",
            "...ij,...jk,...kl",
        ]
    ),
    dtype=st.sampled_from([np.float64, np.float32]),
    data=st.data(),
)
def test_einsum_ellipsis(b, n, m, k, subs, dtype, data):
    # ellipsis spellings of the two-operand and chain regimes, explicit
    # and implicit outputs; sizes straddle the floors both ways
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    ops = [
        rng.uniform(0.5, 1.5, (b, n, m)).astype(dtype),
        rng.uniform(0.5, 1.5, (b, m, k)).astype(dtype),
    ]
    if subs.count(",") == 2:
        l = data.draw(st.integers(1, 20))
        ops.append(rng.uniform(0.5, 1.5, (b, k, l)).astype(dtype))
    tol = close_scaled(1e-6, 1e-9) if dtype is np.float64 else close_scaled(1e-3, 1e-4)
    compare("numpy.einsum", (subs, *ops), {}, tol)


# --- apply_along_axis known-reducer interception (OPP-000054) ---------------
@settings(**SETTINGS)
@given(
    reducer=st.sampled_from(
        ["mean", "sum", "max", "min", "median", "std", "var", "prod",
         "any", "all", "ptp", "argmax", "argmin"]
    ),
    nslices=st.integers(1, 400),
    slice_len=st.integers(1, 40),
    dtype=st.sampled_from([np.float64, np.float32, np.int64, np.int32, np.bool_]),
    axis_spec=st.sampled_from(["last", "neg1", "first", "middle"]),
    three_d=st.booleans(),
    hazard=st.sampled_from(["clean", "nan", "zero_dim", "subclass", "userfunc"]),
    data=st.data(),
)
def test_apply_along_axis_reduce(
    reducer, nslices, slice_len, dtype, axis_spec, three_d, hazard, data
):
    # straddles the slice floor, both axis classes (order-sensitive reducers
    # must refuse off the last axis), and the refusal hazards
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    shape = (nslices, slice_len)
    if three_d and nslices >= 4:
        shape = (2, nslices // 2, slice_len)
    a = (rng.standard_normal(shape) * 5).astype(dtype)
    if hazard == "nan" and dtype in (np.float64, np.float32) and a.size:
        a.reshape(-1)[0] = np.nan
    elif hazard == "zero_dim":
        a = a[:0] if a.ndim == 2 else a[:, :0]
    elif hazard == "subclass":
        a = np.ma.MaskedArray(a)
    nd = a.ndim
    axis = {"last": nd - 1, "neg1": -1, "first": 0, "middle": nd // 2}[axis_spec]
    func = (lambda s: np.asarray(s).mean()) if hazard == "userfunc" else getattr(np, reducer)
    compare("numpy.apply_along_axis", (func, axis, a), {}, exact)


# --- vectorize(ufunc) direct call (OPP-000055) ------------------------------
@settings(**SETTINGS)
@given(
    name=st.sampled_from(
        ["sin", "cos", "exp", "log", "sqrt", "tanh", "arctan", "rint",
         "sign", "cbrt", "square", "absolute"]
    ),
    n=st.integers(0, 5_000),
    dtype=st.sampled_from([np.float64, np.float32, np.int64]),
    two_d=st.booleans(),
    kind=st.sampled_from(["plain", "otypes", "cache", "excluded", "pyfunc"]),
    data=st.data(),
)
def test_vectorize_ufunc_direct(name, n, dtype, two_d, kind, data):
    # compare() cannot drive a CLASS, so both sides are run by hand: the
    # patched np.vectorize against the stock class captured from the gearbox
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    uf = getattr(np, name)
    x = np.abs(rng.standard_normal(n) * 3).astype(dtype) + dtype(1)
    if two_d and n and n % 2 == 0:
        x = x.reshape(2, n // 2)
    kwargs = {
        "plain": {}, "otypes": {"otypes": [np.float64]}, "cache": {"cache": True},
        "excluded": {"excluded": set()}, "pyfunc": {},
    }[kind]
    pyfunc = (lambda t: uf(t)) if kind == "pyfunc" else uf
    stock_cls = GEARBOX.stock_fn("numpy.vectorize")

    def run(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return ("ok", cls(pyfunc, **kwargs)(x))
            except Exception as exc:  # noqa: BLE001 - parity capture
                return ("raised", type(exc))

    got_kind, got = run(np.vectorize)
    exp_kind, exp = run(stock_cls)
    assert got_kind == exp_kind, (got_kind, got, exp_kind, exp)
    if got_kind == "raised":
        assert got is exp
    else:
        assert type(got) is type(exp)
        assert np.asarray(got).dtype == np.asarray(exp).dtype
        assert np.asarray(got).shape == np.asarray(exp).shape
        assert np.array_equal(np.asarray(got), np.asarray(exp), equal_nan=True)


# --- singular-value family on 2x2/3x3 batches (OPP-000056) ------------------
@settings(**SETTINGS)
@given(
    batch=st.integers(1, 600),
    d=st.sampled_from([2, 3, 4]),
    dtype=st.sampled_from([np.float64, np.float32]),
    op=st.sampled_from(["pinv", "norm2", "svdvals"]),
    hazard=st.sampled_from(
        ["clean", "ill_conditioned", "singular", "all_singular",
         "nonfinite", "rectangular", "lead_4d"]
    ),
    kwarg_twist=st.sampled_from(["none", "rcond", "hermitian", "full", "keepdims"]),
    data=st.data(),
)
def test_svd_small_batch(batch, d, dtype, op, hazard, kwarg_twist, data):
    # straddles the batch floor and both conditioning bands; the singular
    # and ill-conditioned hazards exercise the split-to-stock arm, which is
    # where two valid answers could otherwise diverge
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.standard_normal((batch, d, d))
    if hazard == "ill_conditioned" and batch:
        u, _ = np.linalg.qr(rng.standard_normal((batch, d, d)))
        v, _ = np.linalg.qr(rng.standard_normal((batch, d, d)))
        s = np.geomspace(1.0, 1e-9, d)
        a = u @ (s[None, :, None] * np.swapaxes(v, -1, -2))
    elif hazard == "singular" and batch:
        a[rng.integers(0, batch)] = 0.0
    elif hazard == "all_singular":
        a[...] = 0.0
    elif hazard == "nonfinite" and batch:
        # infinity is withheld from pinv ONLY. compare() has to execute
        # stock, and stock pinv on a matrix 3x3 or larger with an infinite
        # DIAGONAL entry never returns on Linux - it calls svd with
        # compute_uv=True, which spins in LAPACK there (upstream, see
        # docs/research/upstream-pinv-inf-hang.md). svdvals and norm(ord=2)
        # both go through compute_uv=False and are unaffected, so they keep
        # the full hazard.
        bad = [np.nan] if op == "pinv" else [np.nan, np.inf]
        a[rng.integers(0, batch), 0, 0] = data.draw(st.sampled_from(bad))
    elif hazard == "rectangular":
        a = a[..., : max(d - 1, 1)]
    a = np.ascontiguousarray(a).astype(dtype)
    if hazard == "lead_4d" and batch >= 4 and batch % 2 == 0:
        a = a.reshape(2, batch // 2, *a.shape[1:])
    if op == "pinv":
        kwargs = {"rcond": 1e-12} if kwarg_twist == "rcond" else (
            {"hermitian": True} if kwarg_twist == "hermitian" else {}
        )
        compare("numpy.linalg.pinv", (a,), kwargs, close_scaled(1e-7, 1e-9))
    elif op == "norm2":
        kwargs = {"ord": 2, "axis": (-2, -1)}
        if kwarg_twist == "keepdims":
            kwargs["keepdims"] = True
        compare("numpy.linalg.norm", (a,), kwargs, close_scaled(1e-9, 1e-12))
    else:
        kwargs = {"compute_uv": False}
        if kwarg_twist == "hermitian":
            kwargs["hermitian"] = True
        elif kwarg_twist == "full":
            kwargs["full_matrices"] = False
        compare("numpy.linalg.svd", (a,), kwargs, close_scaled(1e-7, 1e-9))


# --- batched 2x2/3x3 qr Householder closed form (OPP-000053) ----------------
@settings(**SETTINGS)
@given(
    batch=st.integers(1, 1200),
    d=st.sampled_from([2, 3]),
    mode=st.sampled_from([None, "reduced", "r", "complete", "raw"]),
    positional=st.booleans(),
    hazard=st.sampled_from(
        [
            "clean",
            "upper_triangular",
            "zero_first_col",
            "all_zero",
            "neg_leads",
            "nonfinite",
            "f32",
        ]
    ),
    data=st.data(),
)
def test_qr_small_batch(batch, d, mode, positional, hazard, data):
    # the sign contract must hold with no abs(): upper_triangular hits the
    # tau=0 identity-reflector branch, zero_first_col/all_zero the beta=0
    # edges, neg_leads the copysign choice; nonfinite/f32/'raw' must refuse
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = rng.standard_normal((batch, d, d))
    if hazard == "upper_triangular":
        a = np.triu(a)
    elif hazard == "zero_first_col":
        a[..., :, 0] = 0.0
    elif hazard == "all_zero":
        a[...] = 0.0
    elif hazard == "neg_leads":
        a[..., 0, 0] = -np.abs(a[..., 0, 0])
    elif hazard == "nonfinite":
        a[batch // 2, 0, 0] = data.draw(st.sampled_from([np.nan, np.inf]))
    a = np.ascontiguousarray(a)
    if hazard == "f32":
        a = a.astype(np.float32)
    args: tuple = (a,)
    kwargs: dict = {}
    if mode is not None:
        if positional:
            args = (a, mode)
        else:
            kwargs = {"mode": mode}
    tol = (
        close_scaled(1e-9, 1e-12)
        if a.dtype == np.float64
        else close_scaled(1e-3, 1e-6)
    )
    compare("numpy.linalg.qr", args, kwargs, tol)


# --- interp on a uniform grid (OPP-000050) ----------------------------------
@settings(**SETTINGS)
@given(
    nq=st.integers(0, 30000),
    ngrid=st.integers(2, 500),
    grid_kind=st.sampled_from(["linspace", "arange", "nonuniform"]),
    q_kind=st.sampled_from(["inside", "straddle", "on_points", "nan"]),
    data=st.data(),
)
def test_interp_uniform(nq, ngrid, grid_kind, q_kind, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    if grid_kind == "linspace":
        xp = np.linspace(-1.0, 2.0, ngrid)
    elif grid_kind == "arange":
        xp = 0.25 + 0.01 * np.arange(ngrid)
    else:
        xp = np.sort(rng.uniform(-1.0, 2.0, ngrid))
        if ngrid >= 2 and xp[0] == xp[-1]:
            xp[-1] += 1.0
    fp = rng.standard_normal(ngrid)
    lo, hi = float(xp[0]), float(xp[-1])
    if q_kind == "inside":
        x = rng.uniform(lo, hi, nq)
    elif q_kind == "straddle":
        x = rng.uniform(lo - 1.0, hi + 1.0, nq)
    elif q_kind == "on_points":
        x = rng.choice(xp, nq) if ngrid else np.empty(0)
    else:
        x = rng.uniform(lo, hi, nq)
        if nq:
            x[rng.integers(0, nq)] = np.nan
    compare("numpy.interp", (x, xp, fp), {}, close_scaled(1e-9, 1e-12))


# --- take with out= (OPP-000051) --------------------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(1, 50000),
    k=st.integers(0, 30000),
    dtype=st.sampled_from([np.float64, np.int64]),
    idx_kind=st.sampled_from(["pos", "mixed", "oob"]),
    data=st.data(),
)
def test_take_index_assign(n, k, dtype, idx_kind, data):
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = (
        rng.standard_normal(n)
        if dtype is np.float64
        else rng.integers(-1000, 1000, n)
    ).astype(dtype)
    lo = 0 if idx_kind == "pos" else -n
    idx = rng.integers(lo, n, k).astype(np.intp)
    if idx_kind == "oob" and k:
        idx[rng.integers(0, k)] = n + 5
    out_got = np.empty(k, dtype=dtype)
    out_exp = np.empty(k, dtype=dtype)
    # compare() shares one kwargs dict between both calls; out must differ,
    # so run the two sides by hand with the same raise-parity contract
    stock = GEARBOX.stock_fn("numpy.take")
    try:
        expected = stock(a, idx, out=out_exp)
        stock_exc = None
    except Exception as exc:  # noqa: BLE001 - parity is the contract
        expected, stock_exc = None, exc
    try:
        got = np.take(a, idx, out=out_got)
        got_exc = None
    except Exception as exc:  # noqa: BLE001
        got, got_exc = None, exc
    if stock_exc is not None or got_exc is not None:
        assert type(got_exc) is type(stock_exc), (got_exc, stock_exc)
        return
    assert got is out_got
    assert expected is out_exp
    assert exact(got, expected)


# --- 1-D constant-mode np.pad (OPP-000057) ----------------------------------
@settings(**SETTINGS)
@given(
    n=st.integers(0, 600),
    before=st.integers(-2, 400),
    after=st.integers(-2, 400),
    dtype=st.sampled_from([np.float64, np.float32, np.int64, np.int32,
                           np.uint8, np.bool_, np.complex128]),
    spelling=st.sampled_from(["pair", "scalar", "nested", "list", "kwarg", "array"]),
    cv=st.sampled_from(["absent", 0, -0.0, 5, -1, 2.5, np.nan, np.inf,
                        (1, 2), 1 + 2j, "x", None]),
    mode=st.sampled_from(["absent", "constant", "edge", "reflect"]),
    hazard=st.sampled_from(["clean", "noncontig", "readonly", "subclass",
                            "twod", "zerod", "strdtype", "objdtype"]),
    data=st.data(),
)
def test_pad_1d_constant(n, before, after, dtype, spelling, cv, mode, hazard, data):
    # byte_exact, not exact: constant_values=-0.0 must come back with its
    # sign bit, and np.array_equal cannot see that
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    a = (rng.standard_normal(n) * 4).astype(dtype)

    if hazard == "noncontig":
        a = (rng.standard_normal(max(n * 2, 2)) * 4).astype(dtype)[::2]
    elif hazard == "readonly" and n:
        a = a.copy()
        a.flags.writeable = False
    elif hazard == "subclass":
        a = a.view(type("Sub", (np.ndarray,), {}))
    elif hazard == "twod":
        a = (rng.standard_normal((3, 4)) * 4).astype(dtype)
    elif hazard == "zerod":
        a = np.asarray(a.dtype.type(1))
    elif hazard == "strdtype":
        a = np.array(["a", "bb", "c"])
    elif hazard == "objdtype":
        a = np.array([1, "x", None], dtype=object)

    if spelling == "pair":
        args, kwargs = (a, (before, after)), {}
    elif spelling == "scalar":
        args, kwargs = (a, before), {}
    elif spelling == "nested":
        args, kwargs = (a, ((before, after),)), {}
    elif spelling == "list":
        args, kwargs = (a, [before, after]), {}
    elif spelling == "array":
        args, kwargs = (a, np.array([before, after])), {}
    else:
        args, kwargs = (a,), {"pad_width": (before, after)}

    if mode != "absent":
        kwargs = {**kwargs, "mode": mode}
    if cv != "absent":
        kwargs = {**kwargs, "constant_values": cv}

    compare("numpy.pad", args, kwargs, byte_exact)
