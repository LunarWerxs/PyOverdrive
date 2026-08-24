"""Differential tests: vectorize_ufunc_direct fast path vs stock np.vectorize.

Contract (src/pyoverdrive/fastpaths/vectorize_ufunc.py): np.vectorize is
patched with a SUBCLASS of the stock class (a ClassPath, not a FastPath
function wrapper), so isinstance/type/attribute/pickling behavior must
survive unchanged. An instance accelerates only when: pyfunc resolves to
one of the served unary float64 ufuncs (_SERVED_NAMES); no otypes,
signature, excluded, or cache were given; and each call is exactly one
positional float64 ndarray argument, ndim >= 1, size > 0, no kwargs.
Every other construction or call shape must reproduce stock's own
__call__ exactly, value and type (or the same exception type).

The served set is measured, not assumed: NumPy's scalar loop (what
vectorize's object loop calls) and its array loop (what the direct call
uses) are separate implementations that could differ in the last ulp for
some math. Test 1 below re-verifies bit-identical agreement for every
served name, on whatever NumPy build/version/CPU dispatch target runs
this suite, so a future regression fails loudly here instead of shipping
silently wrong values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.vectorize_ufunc import _SERVED_NAMES

OP = "numpy.vectorize"
PATH = "vectorize_ufunc_direct"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock_cls():
    return GEARBOX.stock_fn(OP)


def _stock_vectorize(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _assert_accelerated(v):
    """First-principles guard: without this, both sides just run stock's
    own __call__ and the "agreement" would be comparing stock to itself."""
    assert v.__pyoverdrive_direct__ is not None


# ---------------------------------------------------------------------------
# 1. THE LOAD-BEARING TEST: the served set, re-verified from first principles
# ---------------------------------------------------------------------------
#
# _SERVED_NAMES claims NumPy's scalar loop and array loop agree bit-for-bit
# for these ufuncs. That claim can rot silently across NumPy versions,
# builds, or CPU dispatch targets (SIMD vs generic loops can differ in the
# last ulp). This test re-derives the claim on whatever machine runs the
# suite, rather than trusting the comment in vectorize_ufunc.py, so a
# future NumPy build that breaks the property for any member fails CI here
# instead of shipping silently wrong values through the fast path.

_ADVERSARIAL_BASE = np.array(
    [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, math.pi, -math.pi, math.e, -math.e],
    dtype=np.float64,
)

_DOMAIN_OVERRIDES = {
    "arcsin": (-0.999999, 0.999999),
    "arccos": (-0.999999, 0.999999),
    "arctanh": (-0.999999, 0.999999),
    "log": (1e-6, 1e6),
    "log2": (1e-6, 1e6),
    "log10": (1e-6, 1e6),
    "sqrt": (0.0, 1e6),
    "log1p": (-0.999999, 1e6),
}


def _sweep_for(name: str) -> np.ndarray:
    bounds = _DOMAIN_OVERRIDES.get(name)
    if bounds is not None:
        lo, hi = bounds
        domain_lin = np.linspace(lo, hi, 4000, dtype=np.float64)
        specials = _ADVERSARIAL_BASE[(_ADVERSARIAL_BASE >= lo) & (_ADVERSARIAL_BASE <= hi)]
        return np.concatenate([domain_lin, specials])
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    scaled = np.concatenate(
        [
            rng.uniform(-1.0, 1.0, 5990).astype(np.float64) * 1e-8,
            rng.uniform(-1.0, 1.0, 5990).astype(np.float64) * 1e0,
            rng.uniform(-1.0, 1.0, 5990).astype(np.float64) * 1e8,
        ]
    )
    domain_lin = np.linspace(-100.0, 100.0, 2000, dtype=np.float64)
    return np.concatenate([scaled, domain_lin, _ADVERSARIAL_BASE])


@pytest.mark.parametrize("name", _SERVED_NAMES)
def test_served_name_scalar_and_array_loops_agree_bitwise(name):
    # resolved from the real numpy module: ufunc objects are never replaced
    # by patching (only np.vectorize the class is), so this holds whether
    # or not PyOverdrive is currently enabled.
    stock_uf = getattr(np, name)
    x = _sweep_for(name)
    assert x.size <= 20_000
    with np.errstate(all="ignore"):
        array_result = stock_uf(x)
        scalar_result = np.array([stock_uf(float(v)) for v in x.tolist()], dtype=array_result.dtype)
    assert array_result.dtype == scalar_result.dtype == np.float64
    assert array_result.shape == scalar_result.shape
    assert np.array_equal(array_result, scalar_result, equal_nan=True)


# ---------------------------------------------------------------------------
# 2. Values: representative subset, 1-D and 2-D
# ---------------------------------------------------------------------------

_REP_NAMES = ("sin", "exp", "log", "sqrt", "tanh", "rint", "sign")


@pytest.mark.parametrize("name", _REP_NAMES)
def test_value_agreement_1d(name):
    uf = getattr(np, name)
    v = np.vectorize(uf)
    _assert_accelerated(v)
    if name in ("log", "sqrt"):
        x = np.linspace(0.01, 100.0, 500, dtype=np.float64)
    else:
        x = np.linspace(-50.0, 50.0, 500, dtype=np.float64)
    got = v(x)
    sv = _stock_vectorize(uf)
    stock = sv(x)
    assert got.dtype == stock.dtype == np.float64
    assert got.shape == stock.shape
    assert np.array_equal(got, stock, equal_nan=True)


@pytest.mark.parametrize("name", _REP_NAMES)
def test_value_agreement_2d(name):
    uf = getattr(np, name)
    v = np.vectorize(uf)
    _assert_accelerated(v)
    if name in ("log", "sqrt"):
        x = np.linspace(0.01, 100.0, 400, dtype=np.float64).reshape(20, 20)
    else:
        x = np.linspace(-50.0, 50.0, 400, dtype=np.float64).reshape(20, 20)
    assert x.ndim > 1
    got = v(x)
    stock = _stock_vectorize(uf)(x)
    assert got.dtype == stock.dtype == np.float64
    assert got.shape == stock.shape
    assert np.array_equal(got, stock, equal_nan=True)


# ---------------------------------------------------------------------------
# 3. Class mechanics
# ---------------------------------------------------------------------------

def test_class_identity_and_isinstance():
    stock_cls = _stock_cls()
    v = np.vectorize(np.sin)
    assert isinstance(v, np.vectorize)
    assert type(v) is np.vectorize
    assert np.vectorize.__name__ == stock_cls.__name__
    assert np.vectorize.__qualname__ == stock_cls.__qualname__
    assert isinstance(v, stock_cls)


def test_restored_to_exact_stock_class_after_disable():
    stock_cls = GEARBOX.stock_fn(OP)
    pyoverdrive.disable()
    try:
        assert np.vectorize is stock_cls
    finally:
        pyoverdrive.enable([OP])


# ---------------------------------------------------------------------------
# 4. Refusals: fall through to stock's own __call__, matching exactly
# ---------------------------------------------------------------------------

def _assert_call_refused_matches(v, sv, call_args, call_kwargs):
    def _call(fn):
        try:
            return ("ok", fn(*call_args, **call_kwargs))
        except Exception as exc:  # noqa: BLE001 - parity capture
            return ("raised", exc)

    got_kind, got_val = _call(v)
    stock_kind, stock_val = _call(sv)
    assert got_kind == stock_kind, (got_kind, got_val, stock_kind, stock_val)
    if got_kind == "raised":
        assert type(got_val) is type(stock_val)
    else:
        assert type(got_val) is type(stock_val)
        if isinstance(stock_val, np.ndarray):
            assert got_val.dtype == stock_val.dtype
            assert got_val.shape == stock_val.shape
            assert np.array_equal(got_val, stock_val, equal_nan=True)
        else:
            assert got_val == stock_val


def test_refusal_python_scalar_input():
    v = np.vectorize(np.sin)
    _assert_accelerated(v)
    sv = _stock_vectorize(np.sin)
    _assert_call_refused_matches(v, sv, (1.5,), {})


def test_refusal_0d_array_input():
    v = np.vectorize(np.sin)
    _assert_accelerated(v)
    sv = _stock_vectorize(np.sin)
    x = np.array(1.5, dtype=np.float64)
    assert x.ndim == 0
    _assert_call_refused_matches(v, sv, (x,), {})


def test_refusal_empty_array_input_raises():
    v = np.vectorize(np.sin)
    _assert_accelerated(v)
    sv = _stock_vectorize(np.sin)
    x = np.array([], dtype=np.float64)
    # stock RAISES ValueError for size-0 input without otypes
    with pytest.raises(ValueError):
        sv(x)
    with pytest.raises(ValueError):
        v(x)


def test_refusal_float32_input():
    v = np.vectorize(np.sin)
    _assert_accelerated(v)
    sv = _stock_vectorize(np.sin)
    x = np.linspace(-10, 10, 50, dtype=np.float32)
    _assert_call_refused_matches(v, sv, (x,), {})


def test_refusal_int64_input():
    v = np.vectorize(np.sin)
    _assert_accelerated(v)
    sv = _stock_vectorize(np.sin)
    x = np.arange(-10, 10, dtype=np.int64)
    _assert_call_refused_matches(v, sv, (x,), {})


def test_refusal_python_list_input():
    v = np.vectorize(np.sin)
    _assert_accelerated(v)
    sv = _stock_vectorize(np.sin)
    x = [0.0, 0.5, 1.0, -1.0]
    _assert_call_refused_matches(v, sv, (x,), {})


def test_refusal_two_positional_args():
    v = np.vectorize(np.sin)
    _assert_accelerated(v)
    sv = _stock_vectorize(np.sin)
    x = np.linspace(-1, 1, 10, dtype=np.float64)
    y = np.linspace(-1, 1, 10, dtype=np.float64)
    _assert_call_refused_matches(v, sv, (x, y), {})


def test_refusal_keyword_argument():
    def f(x):
        return np.sin(x)

    v = np.vectorize(np.sin)
    _assert_accelerated(v)
    sv = _stock_vectorize(np.sin)
    x = np.linspace(-1, 1, 10, dtype=np.float64)
    _assert_call_refused_matches(v, sv, (), {"x": x})


# ---------------------------------------------------------------------------
# 5. Constructions that must NOT accelerate, but must still work
# ---------------------------------------------------------------------------

def _assert_not_accelerated(v, op_args=(), op_kwargs=None):
    op_kwargs = op_kwargs or {}
    assert v.__pyoverdrive_direct__ is None
    decision, reason = GEARBOX.decide(OP, op_args, op_kwargs)
    assert decision == "stock", (decision, reason)


def test_construction_otypes_not_accelerated_but_works():
    v = np.vectorize(np.sin, otypes=[np.float64])
    _assert_not_accelerated(v, (np.sin,), {"otypes": [np.float64]})
    sv = _stock_vectorize(np.sin, otypes=[np.float64])
    x = np.linspace(-5, 5, 30, dtype=np.float64)
    assert np.array_equal(v(x), sv(x), equal_nan=True)


def test_construction_excluded_not_accelerated_but_works():
    def scaled_sin(x, scale=1.0):
        return np.sin(x) * scale

    v = np.vectorize(scaled_sin, excluded={"scale"})
    _assert_not_accelerated(v, (scaled_sin,), {"excluded": {"scale"}})
    sv = _stock_vectorize(scaled_sin, excluded={"scale"})
    x = np.linspace(-5, 5, 30, dtype=np.float64)
    assert np.array_equal(v(x, scale=2.0), sv(x, scale=2.0), equal_nan=True)


def test_construction_signature_not_accelerated_but_works():
    v = np.vectorize(np.sin, signature="()->()")
    _assert_not_accelerated(v, (np.sin,), {"signature": "()->()"})
    sv = _stock_vectorize(np.sin, signature="()->()")
    x = np.linspace(-5, 5, 30, dtype=np.float64)
    assert np.array_equal(v(x), sv(x), equal_nan=True)


def test_construction_cache_not_accelerated_but_works():
    v = np.vectorize(np.sin, cache=True)
    _assert_not_accelerated(v, (np.sin,), {"cache": True})
    sv = _stock_vectorize(np.sin, cache=True)
    x = np.linspace(-5, 5, 30, dtype=np.float64)
    assert np.array_equal(v(x), sv(x), equal_nan=True)


def test_construction_plain_python_function_not_accelerated_but_works():
    def double_sin(x):
        return np.sin(x) * 2.0

    v = np.vectorize(double_sin)
    _assert_not_accelerated(v, (double_sin,), {})
    sv = _stock_vectorize(double_sin)
    x = np.linspace(-5, 5, 30, dtype=np.float64)
    assert np.array_equal(v(x), sv(x), equal_nan=True)


def test_construction_binary_ufunc_not_accelerated_but_works():
    v = np.vectorize(np.add)
    _assert_not_accelerated(v, (np.add,), {})
    sv = _stock_vectorize(np.add)
    x = np.linspace(-5, 5, 30, dtype=np.float64)
    y = np.linspace(5, -5, 30, dtype=np.float64)
    assert np.array_equal(v(x, y), sv(x, y), equal_nan=True)


def test_construction_non_served_ufunc_isnan_not_accelerated_but_works():
    v = np.vectorize(np.isnan)
    _assert_not_accelerated(v, (np.isnan,), {})
    sv = _stock_vectorize(np.isnan)
    x = np.array([0.0, np.nan, 1.0, np.inf, -np.inf], dtype=np.float64)
    assert np.array_equal(v(x), sv(x))


def test_construction_non_served_ufunc_logical_not_not_accelerated_but_works():
    v = np.vectorize(np.logical_not)
    _assert_not_accelerated(v, (np.logical_not,), {})
    sv = _stock_vectorize(np.logical_not)
    x = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
    assert np.array_equal(v(x), sv(x))


def test_construction_decorator_spelling_not_accelerated_but_works():
    @np.vectorize
    def triple(x):
        return x * 3.0

    assert triple.__pyoverdrive_direct__ is None
    decision, reason = GEARBOX.decide(OP, (triple.pyfunc,), {})
    assert decision == "stock", (decision, reason)

    @_stock_vectorize
    def triple_stock(x):
        return x * 3.0

    x = np.linspace(-5, 5, 30, dtype=np.float64)
    assert np.array_equal(triple(x), triple_stock(x), equal_nan=True)


# ---------------------------------------------------------------------------
# 6. Kill switch: live, per already-constructed instance
# ---------------------------------------------------------------------------

def test_kill_switch_live_on_existing_instance():
    v = np.vectorize(np.sin)
    _assert_accelerated(v)
    x = np.linspace(-10, 10, 200, dtype=np.float64)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (np.sin,), {})
        assert decision == "stock", (decision, reason)
        # the instance's cached __pyoverdrive_direct__ is still set, but the
        # subclass consults path.enabled live at call time
        got = v(x)
        stock = _stock_vectorize(np.sin)(x)
        assert np.array_equal(got, stock, equal_nan=True)
    finally:
        pyoverdrive.enable_path(PATH)
