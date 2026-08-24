"""Differential tests: isin_string_hash fast path vs stock numpy.isin.

Contract (src/pyoverdrive/fastpaths/isin_string_hash.py): applies only to
isin(element, test_elements[, assume_unique, invert]) where both operands are
plain 1-D ndarrays of the DEFAULT StringDType (dtype.kind 'T' and no
na_object configured, detected via hasattr(dtype, 'na_object') being False),
kwargs are a subset of {assume_unique, invert, kind} with kind absent or
None, invert (if given) a bool/np.bool_, and element.size + test_elements.size
>= 300 (SIZE_FLOOR). Dispatch builds a Python set from test_elements.tolist()
and probes each element via Python string equality, which is exactly
StringDType (no na_object) equality, so the bool result is bit-identical to
stock's. Comparison mode: bit-identical.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX

OP = "numpy.isin"
PATH = "isin_string_hash"

STR_DTYPE = np.dtypes.StringDType()
VOCAB = [f"word{i:03d}" for i in range(50)]


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _strarr(words, dtype=None):
    return np.array(list(words), dtype=dtype if dtype is not None else STR_DTYPE)


def _words(n, seed, vocab=VOCAB):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vocab), size=n)
    return _strarr(vocab[i] for i in idx)


def _assert_dispatched_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.isin(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert got.dtype == np.bool_
    assert got.shape == stock.shape
    assert got.shape == args[0].shape
    assert np.array_equal(got, stock)
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.isin(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    assert type(got) is type(stock)
    assert np.array_equal(got, stock)
    return got


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.isin(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + bit-identity
# ---------------------------------------------------------------------------

def test_dispatch_words_test_set_5():
    element = _words(2000, seed=1)
    rng = np.random.default_rng(2)
    test = _strarr(rng.choice(VOCAB, size=5, replace=False))
    got, stock = _assert_dispatched_equal((element, test), {})
    assert got.any() and not got.all()


def test_dispatch_words_test_set_50_shuffled():
    element = _words(2000, seed=3)
    rng = np.random.default_rng(4)
    test = _strarr(rng.permutation(VOCAB))
    got, stock = _assert_dispatched_equal((element, test), {})
    assert got.all()


def test_dispatch_words_test_all_words_unshuffled():
    element = _words(2000, seed=5)
    test = _strarr(VOCAB)
    got, stock = _assert_dispatched_equal((element, test), {})
    assert got.all()


def test_dispatch_words_empty_overlap():
    element = _words(2000, seed=6)
    other_vocab = [f"other{i:03d}" for i in range(20)]
    test = _strarr(other_vocab)
    got, stock = _assert_dispatched_equal((element, test), {})
    assert not got.any()


def test_dispatch_invert_true():
    element = _words(2000, seed=7)
    rng = np.random.default_rng(8)
    test = _strarr(rng.choice(VOCAB, size=5, replace=False))
    got, stock = _assert_dispatched_equal((element, test), {"invert": True})
    assert got.any() and not got.all()


def test_dispatch_assume_unique_true():
    element = _words(2000, seed=9)
    rng = np.random.default_rng(10)
    test = _strarr(rng.choice(VOCAB, size=5, replace=False))
    _assert_dispatched_equal((element, test), {"assume_unique": True})


def test_dispatch_assume_unique_false():
    element = _words(2000, seed=11)
    rng = np.random.default_rng(12)
    test = _strarr(rng.choice(VOCAB, size=5, replace=False))
    _assert_dispatched_equal((element, test), {"assume_unique": False})


def test_dispatch_empty_test_elements():
    element = _words(2000, seed=13)
    test = _strarr([])
    assert test.size == 0
    got, stock = _assert_dispatched_equal((element, test), {})
    assert not got.any()


def test_dispatch_empty_and_null_and_nonascii_and_long_strings():
    words = [
        "",
        "plain\x00middle",  # embedded null, not the whole string
        "αβγ",  # Greek
        "\U0001f600",  # emoji
        "x" * 2000,
        "y" * 2000,
    ]
    rng = np.random.default_rng(14)
    filler = [VOCAB[i] for i in rng.integers(0, len(VOCAB), size=300)]
    element = _strarr(words + filler)
    test = _strarr(["", "plain\x00middle", "αβγ", "\U0001f600", "x" * 2000])
    got, stock = _assert_dispatched_equal((element, test), {})
    assert got[:4].all()  # "" embedded-null greek emoji all present in test
    assert bool(got[4])  # "x"*2000 present
    assert not bool(got[5])  # "y"*2000 absent


def test_lone_nul_string_stock_quirk_is_refused():
    # Genuine stock numpy quirk (numpy 2.4.5): np.isin on StringDType
    # returns False for a string that is exactly one NUL character even
    # though it is present in test_elements by Python string equality and
    # set membership - while an embedded ('plain\x00middle') or doubled
    # NUL matches fine, i.e. stock disagrees with itself. The hash route
    # would answer correctly, which is still a divergence from stock on an
    # input class whose behavior upstream is a bug in flux, so the
    # predicate REFUSES inputs containing the lone-NUL string and stock
    # keeps answering for them (see isin_string_hash.py's contract).
    rng = np.random.default_rng(141)
    filler = [VOCAB[i] for i in rng.integers(0, len(VOCAB), size=300)]
    element = _strarr(["\x00"] + filler)
    test = _strarr(["\x00"] + VOCAB[:5])
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((element, test), {})
    # doubled NUL is ALSO in stock's broken class (measured diverging), so
    # it is refused with the lone one
    element3 = _strarr(["\x00\x00"] + filler)
    test3 = _strarr(["\x00\x00"] + VOCAB[:5])
    decision3, _ = GEARBOX.decide(OP, (element3, test3), {})
    assert decision3 == "stock"
    _assert_refused_equal((element3, test3), {})
    # a NUL EMBEDDED in a longer string is fine on stock; still dispatches
    element2 = _strarr(["plain\x00middle"] + filler)
    test2 = _strarr(["plain\x00middle"] + VOCAB[:5])
    decision2, _ = GEARBOX.decide(OP, (element2, test2), {})
    assert decision2 == PATH
    _assert_dispatched_equal((element2, test2), {})


def test_dispatch_duplicate_heavy_element_and_test():
    element = _strarr(["dup"] * 400 + ["other"] * 100)
    test = _strarr(["dup"] * 250 + ["missing"] * 50)
    got, stock = _assert_dispatched_equal((element, test), {})
    assert got[:400].all()
    assert not got[400:].any()


# ---------------------------------------------------------------------------
# 2. refusal routes
# ---------------------------------------------------------------------------

def test_refusal_unicode_dtype_pair():
    rng = np.random.default_rng(15)
    element = np.array([VOCAB[i] for i in rng.integers(0, 50, size=200)])
    test = np.array(VOCAB[:10])
    assert element.dtype.kind == "U"
    _assert_refused_equal((element, test), {})


def test_refusal_object_dtype():
    rng = np.random.default_rng(16)
    element = np.array([VOCAB[i] for i in rng.integers(0, 50, size=200)], dtype=object)
    test = np.array(VOCAB[:10], dtype=object)
    _assert_refused_equal((element, test), {})


def test_refusal_stringdtype_with_na_object():
    na_dtype = np.dtypes.StringDType(na_object=float("nan"))
    rng = np.random.default_rng(17)
    element = _strarr((VOCAB[i] for i in rng.integers(0, 50, size=600)), dtype=na_dtype)
    element[3] = float("nan")  # actual na entry, since it's assignable
    test = _strarr(VOCAB[:10], dtype=na_dtype)
    # verify stock's own semantics first rather than assuming nan behaviour,
    # then just assert the two calls agree with each other.
    stock_result = _stock(element, test)
    assert stock_result.dtype == np.bool_
    _assert_refused_equal((element, test), {})


def test_refusal_mixed_T_and_U_operands():
    rng = np.random.default_rng(18)
    element = _words(200, seed=19)
    test = np.array([VOCAB[i] for i in rng.integers(0, 50, size=100)])
    assert element.dtype.kind == "T"
    assert test.dtype.kind == "U"
    _assert_refused_equal((element, test), {})


def test_refusal_2d_element():
    element = _words(400, seed=20).reshape(20, 20)
    test = _strarr(VOCAB[:10])
    _assert_refused_equal((element, test), {})


def test_refusal_combined_size_299_just_under_floor():
    element = _words(280, seed=21)
    test = _words(19, seed=22)
    assert element.size + test.size == 299
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((element, test), {})


def test_dispatch_combined_size_300_exact_floor():
    element = _words(281, seed=23)
    test = _words(19, seed=24)
    assert element.size + test.size == 300
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == PATH, (decision, reason)
    _assert_dispatched_equal((element, test), {})


def test_refusal_kind_sort_explicit():
    element = _words(200, seed=25)
    test = _words(150, seed=26)
    _assert_refused_equal((element, test), {"kind": "sort"})


def test_refusal_kind_table_raises_like_stock():
    # kind='table' is not supported for string dtypes: both the fast path
    # (refuses, since kind is not None) and stock (raises ValueError) agree
    # that this call fails, and fails the same way.
    element = _words(200, seed=27)
    test = _words(150, seed=28)
    _assert_refused_raises((element, test), {"kind": "table"})


def test_refusal_python_list_element():
    test = _words(300, seed=29)
    element = [VOCAB[i % 50] for i in range(300)]
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == "stock", (decision, reason)
    got = np.isin(element, test)
    stock = _stock(element, test)
    assert np.array_equal(got, stock)


def test_kill_switch_restores_stock_routing():
    element = _words(2000, seed=30)
    rng = np.random.default_rng(31)
    test = _strarr(rng.choice(VOCAB, size=5, replace=False))
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (element, test), {})
        assert decision == "stock", (decision, reason)
        got = np.isin(element, test)
        stock = _stock(element, test)
        assert np.array_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)
