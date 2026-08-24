"""Differential tests: sort_char_view / unique_char_view fast paths vs
stock numpy.sort / numpy.unique.

Contract (src/pyoverdrive/fastpaths/char_view.py): both paths apply only to
plain 1-D native-byte-order single-character U1/S1 ndarrays, reinterpreted
via an integer view (U1 -> int32, S1 -> uint8) so that sort order and
equality classes are preserved exactly. sort_char_view covers
sort(a[, axis]) with axis absent/None/0/-1, kind/order/stable absent or
None, size >= SORT_FLOOR. unique_char_view covers
unique(ar[, return_index, return_inverse, return_counts]) with axis
absent/None, no other kwargs, and per-flag floors (UNIQUE_IDXINV_FLOOR if
return_index or return_inverse, else UNIQUE_COUNTS_FLOOR[kind] if
return_counts, else UNIQUE_PLAIN_FLOOR). Comparison mode: bit-identical.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.char_view import (
    SORT_FLOOR,
    UNIQUE_COUNTS_FLOOR,
    UNIQUE_IDXINV_FLOOR,
    UNIQUE_PLAIN_FLOOR,
)

SORT_OP = "numpy.sort"
UNIQUE_OP = "numpy.unique"
SORT_PATH = "sort_char_view"
UNIQUE_PATH = "unique_char_view"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([SORT_OP, UNIQUE_OP])
    yield
    pyoverdrive.disable()


def _stock(op):
    return GEARBOX.stock_fn(op)


def _u1(words):
    return np.array(list(words), dtype="U1")


def _s1_bytes(byte_values):
    return np.frombuffer(bytes(byte_values), dtype="S1")


def _rand_u1(n, seed, alphabet="ASDFGHJKLZ"):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(alphabet), size=n)
    return np.array([alphabet[i] for i in idx], dtype="U1")


def _rand_s1(n, seed):
    rng = np.random.default_rng(seed)
    vals = rng.integers(0, 256, size=n, dtype=np.uint8)
    return np.frombuffer(vals.tobytes(), dtype="S1")


def _assert_arrays_equal(got, stock):
    assert type(got) is type(stock)
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.array_equal(got, stock)


def _assert_result_equal(got, stock, input_dtype):
    """Compare a sort/unique result: a lone array, or a tuple of arrays
    (unique with flags). Every member is checked for bit-identical
    equality, dtype, and shape; the values array must keep input_dtype."""
    if isinstance(got, tuple):
        assert isinstance(stock, tuple)
        assert len(got) == len(stock)
        for i, (g, s) in enumerate(zip(got, stock)):
            _assert_arrays_equal(g, s)
            if i == 0:
                assert g.dtype == input_dtype
    else:
        assert not isinstance(stock, tuple)
        _assert_arrays_equal(got, stock)
        assert got.dtype == input_dtype


def _assert_dispatched_equal(op, path, args, kwargs):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == path, (decision, reason)
    fn = np.sort if op == SORT_OP else np.unique
    got = fn(*args, **kwargs)
    stock = _stock(op)(*args, **kwargs)
    input_dtype = args[0].dtype
    _assert_result_equal(got, stock, input_dtype)
    return got, stock


def _assert_refused_equal(op, args, kwargs):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    fn = np.sort if op == SORT_OP else np.unique
    got = fn(*args, **kwargs)
    stock = _stock(op)(*args, **kwargs)
    input_dtype = args[0].dtype if hasattr(args[0], "dtype") else None
    if isinstance(got, tuple):
        assert isinstance(stock, tuple)
        assert len(got) == len(stock)
        for i, (g, s) in enumerate(zip(got, stock)):
            _assert_arrays_equal(g, s)
            if i == 0 and input_dtype is not None:
                assert g.dtype == input_dtype
    else:
        _assert_arrays_equal(got, stock)
        if input_dtype is not None:
            assert got.dtype == input_dtype
    return got


def _assert_refused_raises(op, args, kwargs):
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    fn = np.sort if op == SORT_OP else np.unique
    with pytest.raises(Exception) as got_exc:
        fn(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(op)(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# sort: dispatch + bit-identity
# ---------------------------------------------------------------------------

def test_dispatch_sort_u1_alpha_5000():
    a = _rand_u1(5000, seed=1)
    _assert_dispatched_equal(SORT_OP, SORT_PATH, (a,), {})


def test_dispatch_sort_u1_greek_letters():
    # built via chr() (codepoints 0x3B1-0x3BA, lowercase alpha..kappa) so the
    # source file itself stays pure ASCII
    greek = "".join(chr(cp) for cp in range(0x3B1, 0x3BB))
    rng = np.random.default_rng(2)
    idx = rng.integers(0, len(greek), size=SORT_FLOOR)
    a = np.array([greek[i] for i in idx], dtype="U1")
    assert all(ord(c) > 0x7F for c in greek)
    _assert_dispatched_equal(SORT_OP, SORT_PATH, (a,), {})


def test_dispatch_sort_u1_empty_and_max_codepoint_mixed_with_letters():
    rng = np.random.default_rng(3)
    special = ["", chr(0x10FFFF)]
    letters = list(_rand_u1(SORT_FLOOR - len(special), seed=4))
    words = special + [str(c) for c in letters]
    rng.shuffle(words)
    a = np.array(words, dtype="U1")
    assert a.size == SORT_FLOOR
    _assert_dispatched_equal(SORT_OP, SORT_PATH, (a,), {})


def test_dispatch_sort_s1_full_byte_range_including_high_bit():
    # int8-vs-uint8 trap: bytes >= 0x80 would sort before 0x00-0x7F under a
    # signed view. Cover the whole 0x00-0xFF range, repeated to clear the
    # floor, built via np.frombuffer so no encoding layer touches the bytes.
    reps = -(-SORT_FLOOR // 256)
    byte_values = list(range(256)) * reps
    a = _s1_bytes(byte_values)
    assert a.size >= SORT_FLOOR
    assert a.dtype == np.dtype("S1")
    _assert_dispatched_equal(SORT_OP, SORT_PATH, (a,), {})


def test_dispatch_sort_axis_kwargs_and_positional():
    a = _rand_u1(SORT_FLOOR, seed=5)
    _assert_dispatched_equal(SORT_OP, SORT_PATH, (a,), {"axis": -1})
    _assert_dispatched_equal(SORT_OP, SORT_PATH, (a,), {"axis": 0})
    _assert_dispatched_equal(SORT_OP, SORT_PATH, (a,), {"axis": None})
    _assert_dispatched_equal(SORT_OP, SORT_PATH, (a, 0), {})


def test_dispatch_sort_duplicate_heavy():
    a = np.array(["Q"] * (SORT_FLOOR - 10) + list("ASDFGHJKLZ"), dtype="U1")
    _assert_dispatched_equal(SORT_OP, SORT_PATH, (a,), {})


def test_dispatch_sort_exact_floor_boundary():
    a = _rand_u1(SORT_FLOOR, seed=6)
    assert a.size == SORT_FLOOR
    decision, reason = GEARBOX.decide(SORT_OP, (a,), {})
    assert decision == SORT_PATH, (decision, reason)
    _assert_dispatched_equal(SORT_OP, SORT_PATH, (a,), {})


def test_refusal_sort_floor_minus_one():
    a = _rand_u1(SORT_FLOOR - 1, seed=7)
    decision, reason = GEARBOX.decide(SORT_OP, (a,), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal(SORT_OP, (a,), {})


# ---------------------------------------------------------------------------
# sort: refusals
# ---------------------------------------------------------------------------

def test_refusal_sort_dtype_u2():
    a = np.array(["AB"] * SORT_FLOOR, dtype="U2")
    _assert_refused_equal(SORT_OP, (a,), {})


def test_refusal_sort_dtype_s2():
    a = np.array([b"AB"] * SORT_FLOOR, dtype="S2")
    _assert_refused_equal(SORT_OP, (a,), {})


def test_refusal_sort_2d_u1_array():
    a = _rand_u1(SORT_FLOOR * 2, seed=8).reshape(SORT_FLOOR, 2)
    _assert_refused_equal(SORT_OP, (a,), {})


def test_refusal_sort_kind_stable_string():
    a = _rand_u1(SORT_FLOOR, seed=9)
    _assert_refused_equal(SORT_OP, (a,), {"kind": "stable"})


def test_refusal_sort_kind_quicksort_string():
    a = _rand_u1(SORT_FLOOR, seed=10)
    _assert_refused_equal(SORT_OP, (a,), {"kind": "quicksort"})


def test_refusal_sort_stable_true():
    a = _rand_u1(SORT_FLOOR, seed=11)
    _assert_refused_equal(SORT_OP, (a,), {"stable": True})


def test_refusal_sort_order_kwarg_raises_same_exception():
    a = _rand_u1(SORT_FLOOR, seed=12)
    _assert_refused_raises(SORT_OP, (a,), {"order": "x"})


def test_refusal_sort_axis_1_on_1d_raises_same_exception():
    a = _rand_u1(SORT_FLOOR, seed=13)
    _assert_refused_raises(SORT_OP, (a,), {"axis": 1})


def test_refusal_sort_byteswapped_dtype():
    a = _rand_u1(SORT_FLOOR, seed=14)
    swapped = a.astype(a.dtype.newbyteorder())
    assert not swapped.dtype.isnative
    _assert_refused_equal(SORT_OP, (swapped,), {})

    a2 = np.array(list("ASDFGHJKLZ") * (SORT_FLOOR // 10 + 1), dtype=">U1")
    a2 = a2[:SORT_FLOOR]
    assert not a2.dtype.isnative
    _assert_refused_equal(SORT_OP, (a2,), {})


def test_refusal_sort_empty_array():
    a = np.array([], dtype="U1")
    _assert_refused_equal(SORT_OP, (a,), {})


def test_refusal_sort_python_list_input():
    words = list("ASDFGHJKLZ") * (SORT_FLOOR // 10 + 1)
    decision, reason = GEARBOX.decide(SORT_OP, (words,), {})
    assert decision == "stock", (decision, reason)
    got = np.sort(words)
    stock = _stock(SORT_OP)(words)
    assert np.array_equal(got, stock)


def test_kill_switch_sort_restores_stock_routing():
    a = _rand_u1(SORT_FLOOR, seed=15)
    decision, reason = GEARBOX.decide(SORT_OP, (a,), {})
    assert decision == SORT_PATH, (decision, reason)
    pyoverdrive.disable_path(SORT_PATH)
    try:
        decision, reason = GEARBOX.decide(SORT_OP, (a,), {})
        assert decision == "stock", (decision, reason)
        got = np.sort(a)
        stock = _stock(SORT_OP)(a)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(SORT_PATH)


# ---------------------------------------------------------------------------
# unique: dispatch + bit-identity
# ---------------------------------------------------------------------------

def test_dispatch_unique_plain_u1():
    a = _rand_u1(UNIQUE_PLAIN_FLOOR, seed=20)
    _assert_dispatched_equal(UNIQUE_OP, UNIQUE_PATH, (a,), {})


def test_dispatch_unique_plain_s1():
    a = _rand_s1(UNIQUE_PLAIN_FLOOR, seed=21)
    _assert_dispatched_equal(UNIQUE_OP, UNIQUE_PATH, (a,), {})


def test_refusal_unique_plain_floor_minus_one():
    a = _rand_u1(UNIQUE_PLAIN_FLOOR - 1, seed=22)
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal(UNIQUE_OP, (a,), {})


def test_dispatch_unique_return_counts_u1_at_floor():
    n = UNIQUE_COUNTS_FLOOR["U"]
    a = _rand_u1(n, seed=23)
    assert a.size == n
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {"return_counts": True})
    assert decision == UNIQUE_PATH, (decision, reason)
    _assert_dispatched_equal(UNIQUE_OP, UNIQUE_PATH, (a,), {"return_counts": True})


def test_refusal_unique_return_counts_u1_floor_minus_one():
    n = UNIQUE_COUNTS_FLOOR["U"] - 1
    a = _rand_u1(n, seed=24)
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {"return_counts": True})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal(UNIQUE_OP, (a,), {"return_counts": True})


def test_dispatch_unique_return_counts_s1_at_own_floor():
    n = UNIQUE_COUNTS_FLOOR["S"]
    a = _rand_s1(n, seed=25)
    assert a.size == n
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {"return_counts": True})
    assert decision == UNIQUE_PATH, (decision, reason)
    _assert_dispatched_equal(UNIQUE_OP, UNIQUE_PATH, (a,), {"return_counts": True})


def test_refusal_unique_return_counts_s1_floor_minus_one():
    n = UNIQUE_COUNTS_FLOOR["S"] - 1
    a = _rand_s1(n, seed=26)
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {"return_counts": True})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal(UNIQUE_OP, (a,), {"return_counts": True})


def test_dispatch_unique_index_inverse_at_floor():
    a = _rand_u1(UNIQUE_IDXINV_FLOOR, seed=27)
    assert a.size == UNIQUE_IDXINV_FLOOR
    kwargs = {"return_index": True, "return_inverse": True}
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), kwargs)
    assert decision == UNIQUE_PATH, (decision, reason)
    _assert_dispatched_equal(UNIQUE_OP, UNIQUE_PATH, (a,), kwargs)


def test_refusal_unique_index_inverse_floor_minus_one():
    a = _rand_u1(UNIQUE_IDXINV_FLOOR - 1, seed=28)
    kwargs = {"return_index": True, "return_inverse": True}
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), kwargs)
    assert decision == "stock", (decision, reason)
    _assert_refused_equal(UNIQUE_OP, (a,), kwargs)


def test_dispatch_unique_all_three_flags():
    a = _rand_s1(UNIQUE_IDXINV_FLOOR, seed=29)
    kwargs = {"return_index": True, "return_inverse": True, "return_counts": True}
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), kwargs)
    assert decision == UNIQUE_PATH, (decision, reason)
    _assert_dispatched_equal(UNIQUE_OP, UNIQUE_PATH, (a,), kwargs)


def test_dispatch_unique_positional_flags_return_index():
    a = _rand_u1(UNIQUE_IDXINV_FLOOR, seed=30)
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a, True), {})
    assert decision == UNIQUE_PATH, (decision, reason)
    _assert_dispatched_equal(UNIQUE_OP, UNIQUE_PATH, (a, True), {})


def test_dispatch_unique_positional_flags_all_three():
    a = _rand_u1(UNIQUE_IDXINV_FLOOR, seed=31)
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a, True, True, True), {})
    assert decision == UNIQUE_PATH, (decision, reason)
    _assert_dispatched_equal(UNIQUE_OP, UNIQUE_PATH, (a, True, True, True), {})


def test_dispatch_unique_flags_as_np_true():
    a = _rand_u1(UNIQUE_IDXINV_FLOOR, seed=32)
    kwargs = {"return_index": np.True_, "return_inverse": np.True_}
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), kwargs)
    assert decision == UNIQUE_PATH, (decision, reason)
    _assert_dispatched_equal(UNIQUE_OP, UNIQUE_PATH, (a,), kwargs)


# ---------------------------------------------------------------------------
# unique: refusals
# ---------------------------------------------------------------------------

def test_refusal_unique_axis0_kwarg():
    a = _rand_u1(UNIQUE_PLAIN_FLOOR, seed=33)
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {"axis": 0})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal(UNIQUE_OP, (a,), {"axis": 0})


def test_refusal_unique_equal_nan_kwarg():
    a = _rand_u1(UNIQUE_PLAIN_FLOOR, seed=34)
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {"equal_nan": True})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal(UNIQUE_OP, (a,), {"equal_nan": True})


def test_refusal_unique_dtype_u2():
    a = np.array(["AB"] * UNIQUE_PLAIN_FLOOR, dtype="U2")
    _assert_refused_equal(UNIQUE_OP, (a,), {})


def test_refusal_unique_2d_array():
    a = _rand_u1(UNIQUE_PLAIN_FLOOR * 2, seed=35).reshape(UNIQUE_PLAIN_FLOOR, 2)
    _assert_refused_equal(UNIQUE_OP, (a,), {})


def test_refusal_unique_stringdtype_array():
    a = np.array(
        [c for c in "ASDFGHJKLZ" for _ in range(UNIQUE_PLAIN_FLOOR // 10 + 1)],
        dtype=np.dtypes.StringDType(),
    )[:UNIQUE_PLAIN_FLOOR]
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {})
    assert decision == "stock", (decision, reason)
    got = np.unique(a)
    stock = _stock(UNIQUE_OP)(a)
    assert np.array_equal(got, stock)


def test_refusal_unique_below_plain_floor_no_flags():
    a = _rand_u1(UNIQUE_PLAIN_FLOOR - 1, seed=36)
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal(UNIQUE_OP, (a,), {})


def test_kill_switch_unique_restores_stock_routing():
    a = _rand_u1(UNIQUE_PLAIN_FLOOR, seed=37)
    decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {})
    assert decision == UNIQUE_PATH, (decision, reason)
    pyoverdrive.disable_path(UNIQUE_PATH)
    try:
        decision, reason = GEARBOX.decide(UNIQUE_OP, (a,), {})
        assert decision == "stock", (decision, reason)
        got = np.unique(a)
        stock = _stock(UNIQUE_OP)(a)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(UNIQUE_PATH)
