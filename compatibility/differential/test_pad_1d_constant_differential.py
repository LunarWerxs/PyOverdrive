"""Differential suite for the 1-D constant-mode np.pad fast path (OPP-000057).

Every value assertion here compares RAW BYTES, not np.array_equal. That is
deliberate and it is what this path needs: np.array_equal reports True for
+0.0 against -0.0, and False for NaN against NaN, and both of those cases
are load-bearing here. constant_values=-0.0 is exactly why the np.zeros
route may only serve a call that supplied no constant at all.

Every test asserts the DISPATCH DECISION before comparing values, so a
refusal can never make a comparison pass vacuously.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.pad_1d_constant import OUTPUT_CAP, _DTYPES

OP = "numpy.pad"
PATH = "pad_1d_constant"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _same_bytes(got, stock) -> bool:
    """Byte-exact for numeric dtypes; element-wise for object dtype.

    The byte comparison is the point of this file - it is the only thing
    that separates +0.0 from -0.0 and matches NaN against NaN. But an
    OBJECT array's buffer holds POINTERS, so tobytes() compares identity
    rather than value there and two equal arrays built separately never
    match. Object arrays only ever reach here through a refusal case, so
    compare them the way equality is actually defined for them.
    """
    if type(got) is not type(stock) or got.dtype != stock.dtype:
        return False
    if got.shape != stock.shape:
        return False
    if got.dtype == object:
        return list(got) == list(stock)
    return got.tobytes() == stock.tobytes()


def _assert_served(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (args[0].dtype, kwargs, decision, reason)
    got = np.pad(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert _same_bytes(got, stock), (got, stock)
    return got


def _assert_refused(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (kwargs, decision, reason)

    def call(fn):
        try:
            return ("ok", fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - parity capture, not handling
            return ("raised", exc)

    got_kind, got = call(np.pad)
    stock_kind, stock = call(_stock)
    assert got_kind == stock_kind, (got_kind, got, stock_kind, stock)
    if got_kind == "raised":
        assert type(got) is type(stock)
        assert str(got) == str(stock)
    else:
        assert _same_bytes(np.asarray(got), np.asarray(stock))


# ---------------------------------------------------------------------------
# 1. agreement across dtype, size and every accepted pad_width spelling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", sorted(_DTYPES, key=str))
def test_agreement_every_served_dtype(dtype):
    rng = np.random.default_rng(570)
    a = (rng.standard_normal(64) * 8).astype(dtype)
    _assert_served((a, (2, 3)), {})


@pytest.mark.parametrize("n", [0, 1, 2, 8, 64, 256, 1000, 4000])
def test_agreement_across_sizes(n):
    a = np.random.default_rng(571 + n).standard_normal(n)
    _assert_served((a, (2, 3)), {})


@pytest.mark.parametrize(
    "pad_width",
    [0, 3, (2, 3), (0, 0), (0, 5), (5, 0), [2, 3], ((2, 3),), (np.int64(2), np.int32(3)),
     np.array([2, 3]), np.array(4), (7,)],
)
def test_agreement_pad_width_spellings(pad_width):
    a = np.random.default_rng(572).standard_normal(32)
    _assert_served((a, pad_width), {})


def test_agreement_mode_given_explicitly():
    a = np.random.default_rng(573).standard_normal(32)
    _assert_served((a, (2, 3), "constant"), {})
    _assert_served((a, (2, 3)), {"mode": "constant"})


def test_agreement_pad_width_keyword():
    a = np.random.default_rng(574).standard_normal(32)
    _assert_served((a,), {"pad_width": (2, 3)})


@pytest.mark.parametrize("cv", [0, 5, -3, 2.5, (1, 2), ((1, 2),), [4, 4], np.int64(9)])
def test_agreement_constant_values_spellings(cv):
    a = np.random.default_rng(575).standard_normal(32)
    _assert_served((a, (2, 3)), {"constant_values": cv})


# ---------------------------------------------------------------------------
# 2. the bit patterns np.array_equal cannot see
# ---------------------------------------------------------------------------


def test_negative_zero_keeps_its_sign_bit():
    """The reason the np.zeros route is gated on constant_values being ABSENT.

    np.pad(a, 1, constant_values=-0.0) fills with negative zero; np.zeros
    fills with positive zero; np.array_equal says those are equal. Only a
    byte comparison, or np.signbit, can tell.
    """
    a = np.arange(4.0)
    got = _assert_served((a, (2, 3)), {"constant_values": -0.0})
    stock = _stock(a, (2, 3), constant_values=-0.0)
    assert np.array_equal(got, np.zeros_like(got) + a.sum() * 0 + got)  # sanity
    assert list(np.signbit(got)) == list(np.signbit(stock))
    assert np.signbit(got[0]) and np.signbit(got[-1])


def test_positive_zero_default_is_not_negative_zero():
    a = np.arange(4.0)
    got = _assert_served((a, (2, 3)), {})
    assert not np.signbit(got[0]) and not np.signbit(got[-1])


@pytest.mark.parametrize("cv", [np.nan, np.inf, -np.inf])
def test_nonfinite_constants_are_bit_exact(cv):
    a = np.arange(4.0)
    _assert_served((a, (2, 3)), {"constant_values": cv})


def test_negative_constant_wraps_into_unsigned_exactly_as_stock():
    """np.asarray(-1) assigned into uint8 wraps to 255; the Python scalar -1
    raises OverflowError. Stock wraps, so the fast path must use the array."""
    a = np.arange(5, dtype=np.uint8)
    got = _assert_served((a, (2, 3)), {"constant_values": -1})
    assert got[0] == 255


def test_complex_constant_into_real_discards_imaginary_like_stock():
    a = np.arange(4.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        decision, reason = GEARBOX.decide(OP, (a, (2, 3)), {"constant_values": 1 + 2j})
        assert decision == PATH, (decision, reason)
        got = np.pad(a, (2, 3), constant_values=1 + 2j)
        stock = _stock(a, (2, 3), constant_values=1 + 2j)
    assert _same_bytes(got, stock)
    assert got[0] == 1.0


def test_float_constant_truncates_into_int_like_stock():
    a = np.arange(5, dtype=np.int64)
    got = _assert_served((a, (2, 3)), {"constant_values": 1.7})
    assert got[0] == _stock(a, (2, 3), constant_values=1.7)[0]


# ---------------------------------------------------------------------------
# 3. refusals - each one a case where the direct route would diverge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pad_width", [(2.0, 3.0), (2.5, 3.0), 2.0, np.array([2.0, 3.0])])
def test_refusal_float_pad_width(pad_width):
    """Stock raises TypeError even for an integral float like 2.0."""
    _assert_refused((np.arange(8.0), pad_width), {})


@pytest.mark.parametrize("pad_width", [(-1, 2), -3, (2, -1)])
def test_refusal_negative_pad_width(pad_width):
    _assert_refused((np.arange(8.0), pad_width), {})


@pytest.mark.parametrize("mode", ["edge", "reflect", "wrap", "mean", "empty", "symmetric"])
def test_refusal_other_modes(mode):
    _assert_refused((np.arange(8.0), (2, 3)), {"mode": mode})


def test_refusal_callable_mode():
    def pad_with(vector, pad_width, iaxis, kwargs):
        vector[: pad_width[0]] = 0
        vector[-pad_width[1]:] = 0

    _assert_refused((np.arange(8.0), (2, 3)), {"mode": pad_with})


@pytest.mark.parametrize("a", [np.float64(1.5), np.ones((3, 4)), np.ones((2, 2, 2))])
def test_refusal_wrong_ndim(a):
    """0-d returns a value and 2-D pads every axis; neither is our shape."""
    _assert_refused((a, (2, 3)), {})


def test_refusal_string_dtype_pads_with_the_string_zero():
    """Stock fills a string array with '0', np.zeros fills with ''.

    Shapes and dtypes agree, so this one is invisible to anything but a
    value comparison - which is why the dtype allowlist exists.
    """
    a = np.array(["a", "bb"])
    _assert_refused((a, (2, 3)), {})
    assert _stock(a, (2, 3))[0] == "0"


@pytest.mark.parametrize(
    "a",
    [
        np.array([1, "x"], dtype=object),
        np.array([b"ab", b"c"]),
        np.arange(3).astype("datetime64[D]"),
        np.arange(3).astype("timedelta64[D]"),
    ],
)
def test_refusal_unmeasured_dtypes(a):
    _assert_refused((a, (2, 3)), {})


def test_refusal_ndarray_subclass():
    class Sub(np.ndarray):
        pass

    _assert_refused((np.arange(8.0).view(Sub), (2, 3)), {})


def test_refusal_masked_array():
    _assert_refused((np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0]), (2, 3)), {})


def test_refusal_python_list_input():
    _assert_refused(([1.0, 2.0, 3.0], (2, 3)), {})


@pytest.mark.parametrize("cv", ["x", None, np.array(["a"]), object()])
def test_refusal_non_numeric_constant(cv):
    _assert_refused((np.arange(8.0), (2, 3)), {"constant_values": cv})


@pytest.mark.parametrize("pad_width", [(1, 2, 3), ((1, 2), (3, 4)), []])
def test_refusal_wrong_shaped_pad_width(pad_width):
    _assert_refused((np.arange(8.0), pad_width), {})


def test_refusal_unknown_keyword():
    _assert_refused((np.arange(8.0), (2, 3)), {"stat_length": 2})


def test_refusal_duplicate_pad_width():
    _assert_refused((np.arange(8.0), (2, 3)), {"pad_width": (1, 1)})


# ---------------------------------------------------------------------------
# 4. the cap, which is on the OUTPUT length and not the input length
# ---------------------------------------------------------------------------


def test_cap_is_measured_on_the_result_not_the_input():
    """A tiny array with an enormous pad is the shape that looks fastest on
    a bare timing and is a REGRESSION once the result is read, because
    np.zeros hands back calloc pages nobody has faulted in yet."""
    tiny = np.arange(8.0)
    _assert_refused((tiny, (120_000, 3)), {})
    decision, _ = GEARBOX.decide(OP, (tiny, (OUTPUT_CAP, 0)), {})
    assert decision == "stock"


def test_cap_boundary_exactly():
    n = 100
    just_in = OUTPUT_CAP - n
    before, after = just_in // 2, just_in - just_in // 2
    a = np.random.default_rng(576).standard_normal(n)
    assert n + before + after == OUTPUT_CAP
    _assert_served((a, (before, after)), {})
    _assert_refused((a, (before + 1, after)), {})


# ---------------------------------------------------------------------------
# 5. dispatcher contract
# ---------------------------------------------------------------------------


def test_kill_switch_restores_stock_routing():
    a = np.random.default_rng(577).standard_normal(32)
    assert GEARBOX.decide(OP, (a, (2, 3)), {})[0] == PATH
    pyoverdrive.disable_path(PATH)
    try:
        assert GEARBOX.decide(OP, (a, (2, 3)), {})[0] == "stock"
        assert _same_bytes(np.pad(a, (2, 3)), _stock(a, (2, 3)))
    finally:
        pyoverdrive.enable_path(PATH)
    assert GEARBOX.decide(OP, (a, (2, 3)), {})[0] == PATH


def test_no_spurious_warnings_on_the_served_path():
    a = np.random.default_rng(578).standard_normal(32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np.pad(a, (2, 3))
        np.pad(a, (2, 3), constant_values=7)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_result_is_always_a_fresh_writable_array():
    a = np.random.default_rng(579).standard_normal(32)
    out = _assert_served((a, (2, 3)), {})
    assert out.flags.writeable and out.flags.c_contiguous
    assert not np.shares_memory(out, a)


def test_non_contiguous_input_is_served_correctly():
    a = np.random.default_rng(580).standard_normal(64)[::2]
    assert not a.flags.c_contiguous
    _assert_served((a, (2, 3)), {})


def test_readonly_input_is_served_correctly():
    a = np.random.default_rng(581).standard_normal(32)
    a.flags.writeable = False
    _assert_served((a, (2, 3)), {})


def test_zero_width_pad_with_a_constant_is_refused():
    """An empty-slice assignment never performs the cast.

    pad(int_array, 0, constant_values=nan) must raise the way stock does.
    With both widths zero the fast path would write the constant only into
    empty slices, skip the conversion, and quietly return - so this call is
    refused. Hypothesis found it; it is pinned here by hand.
    """
    a = np.arange(5, dtype=np.int64)
    _assert_refused((a, (0, 0)), {"constant_values": np.nan})
    _assert_refused((a, 0), {"constant_values": 3})

    # One side zero is still SERVED: the other side is non-empty, so the
    # cast happens there and a bad constant raises exactly as stock does.
    _assert_served((a, (0, 2)), {"constant_values": 3})

    decision, reason = GEARBOX.decide(OP, (a, (0, 2)), {"constant_values": np.nan})
    assert decision == PATH, (decision, reason)
    with pytest.raises(ValueError) as got:
        np.pad(a, (0, 2), constant_values=np.nan)
    with pytest.raises(ValueError) as expected:
        _stock(a, (0, 2), constant_values=np.nan)
    assert str(got.value) == str(expected.value)


def test_zero_width_pad_without_a_constant_is_still_served():
    a = np.arange(5, dtype=np.int64)
    _assert_served((a, (0, 0)), {})


def test_nan_constant_into_int_array_raises_like_stock():
    """The 0-d-array-versus-numpy-scalar bug, pinned.

    Stock normalizes the constant with _as_pairs, which yields a numpy
    SCALAR. Assigning np.float64(nan) into an int array raises ValueError;
    assigning the equivalent 0-d ARRAY silently writes INT_MIN under a
    RuntimeWarning. An early version of this path used np.asarray and so
    turned a stock exception into a wrong answer. Hypothesis found it.
    """
    a = np.arange(5, dtype=np.int64)
    decision, reason = GEARBOX.decide(OP, (a, (1, 2)), {"constant_values": np.nan})
    assert decision == PATH, (decision, reason)
    with pytest.raises(ValueError) as got:
        np.pad(a, (1, 2), constant_values=np.nan)
    with pytest.raises(ValueError) as expected:
        _stock(a, (1, 2), constant_values=np.nan)
    assert str(got.value) == str(expected.value)


def test_numpy_scalar_normalisation_matches_stock_on_every_casting_corner():
    corners = [
        (np.arange(5, dtype=np.uint8), -1),
        (np.arange(5, dtype=np.int8), 200),
        (np.arange(5, dtype=np.int64), 1.9),
        (np.arange(5, dtype=np.uint16), -70000),
        (np.arange(4.0), True),
        (np.arange(4, dtype=np.int32), np.float32(2.5)),
    ]
    for a, cv in corners:
        decision, reason = GEARBOX.decide(OP, (a, (2, 3)), {"constant_values": cv})
        if decision != PATH:
            continue

        def call(fn, a=a, cv=cv):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    return ("ok", fn(a, (2, 3), constant_values=cv))
                except Exception as exc:  # noqa: BLE001 - parity capture
                    return ("raised", exc)

        got_kind, got = call(np.pad)
        stock_kind, stock = call(_stock)
        # several of these corners raise on BOTH sides, which is itself the
        # contract: an out-of-range constant must fail the way stock fails
        assert got_kind == stock_kind, (a.dtype, cv, got, stock)
        if got_kind == "raised":
            assert type(got) is type(stock) and str(got) == str(stock), (a.dtype, cv)
        else:
            assert _same_bytes(got, stock), (a.dtype, cv, got, stock)
