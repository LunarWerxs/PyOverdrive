"""Differential tests: isin_object_hash fast path vs stock numpy.isin.

Contract (src/pyoverdrive/fastpaths/isin_object_hash.py): applies only to
isin(element, test_elements[, assume_unique, invert]) where both operands are
plain 1-D ndarrays of dtype object, kwargs are a subset of
{assume_unique, invert, kind} with kind absent or None, invert (if given) a
bool/np.bool_, and element.size + test_elements.size >= 300 (SIZE_FLOOR).
Dispatch builds a Python set from test_elements.tolist() and probes each
element via `in`, EXCEPT when any operand contains a NaN-like value (x != x)
or an unhashable value (TypeError building the set): those calls are handed
to stock directly, so the result is bit-identical to stock's by construction
in every case, not merely for the "clean" ones.

numpy.isin has THREE relevant routes for this differential suite:
- isin_object_hash: both operands plain 1-D object-dtype ndarrays (this file).
- isin_string_hash: both operands plain 1-D default-StringDType ndarrays.
- stock: everything else (including U-dtype string arrays, non-object
  numeric dtypes, 2-D arrays, explicit kind=, below the size floor).
Dispatch assertions below expect "isin_object_hash" ONLY for object-dtype
pairs; other input classes are asserted to route to "isin_string_hash" or
"stock" as appropriate, never to "isin_object_hash".
"""

import warnings
from decimal import Decimal

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.isin_object_hash import SIZE_FLOOR

OP = "numpy.isin"
PATH = "isin_object_hash"
OTHER_PATH = "isin_string_hash"

STR_DTYPE = np.dtypes.StringDType()
VOCAB = [f"word{i:03d}" for i in range(50)]


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _objarr(items):
    arr = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        arr[i] = x
    return arr


def _words(n, seed, vocab=VOCAB):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vocab), size=n)
    return [vocab[i] for i in idx]


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

def test_dispatch_string_objects_400_plus_50():
    element = _objarr(_words(400, seed=1))
    other_vocab = [f"other{i:03d}" for i in range(25)]
    test_words = VOCAB[:25] + other_vocab
    test = _objarr(test_words)
    assert element.size + test.size == 450
    got, stock = _assert_dispatched_equal((element, test), {})
    assert got.any() and not got.all()


def test_dispatch_int_objects():
    rng = np.random.default_rng(3)
    element = _objarr(rng.integers(0, 100, size=400).tolist())
    test = _objarr(rng.integers(0, 100, size=50).tolist())
    got, stock = _assert_dispatched_equal((element, test), {})
    assert got.any()


def test_dispatch_mixed_str_int_float_bool_cross_type_equality():
    # 1 == 1.0 == True in Python; hash(1) == hash(1.0) == hash(True), so
    # they collide into ONE set bucket, exactly mirroring stock's == based
    # equality across these types.
    element = _objarr(
        [1, 1.0, True, 0, 0.0, False, "1", "true", 2, 2.0]
        + _words(300, seed=4)
    )
    test = _objarr([True, 0, "1"] + _words(20, seed=5))
    got, stock = _assert_dispatched_equal((element, test), {})
    assert bool(got[0]) and bool(got[1]) and bool(got[2])  # 1, 1.0, True all match True
    assert bool(got[3]) and bool(got[4]) and bool(got[5])  # 0, 0.0, False all match 0
    assert bool(got[6])  # "1" matches "1"
    assert not bool(got[7])  # "true" != "1"


def test_dispatch_tuples_as_objects_hashable_composites():
    rng = np.random.default_rng(6)
    pool = [(i, i * 2) for i in range(40)]
    element = _objarr([pool[i] for i in rng.integers(0, len(pool), size=350)])
    test = _objarr(pool[:10])
    got, stock = _assert_dispatched_equal((element, test), {})
    assert got.any() and not got.all()


def test_dispatch_invert_true():
    element = _objarr(_words(400, seed=7))
    rng = np.random.default_rng(8)
    test = _objarr(rng.choice(VOCAB, size=5, replace=False).tolist())
    got, stock = _assert_dispatched_equal((element, test), {"invert": True})
    assert got.any() and not got.all()


def test_dispatch_assume_unique_true():
    element = _objarr(_words(400, seed=9))
    rng = np.random.default_rng(10)
    test = _objarr(rng.choice(VOCAB, size=50, replace=False).tolist())
    _assert_dispatched_equal((element, test), {"assume_unique": True})


def test_dispatch_assume_unique_false():
    element = _objarr(_words(400, seed=11))
    rng = np.random.default_rng(12)
    test = _objarr(rng.choice(VOCAB, size=50, replace=False).tolist())
    _assert_dispatched_equal((element, test), {"assume_unique": False})


def test_dispatch_combined_size_300_exact_floor():
    element = _objarr(_words(281, seed=13))
    test = _objarr(_words(19, seed=14))
    assert element.size + test.size == SIZE_FLOOR
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == PATH, (decision, reason)
    _assert_dispatched_equal((element, test), {})


def test_refusal_combined_size_299_just_under_floor():
    element = _objarr(_words(280, seed=15))
    test = _objarr(_words(19, seed=16))
    assert element.size + test.size == SIZE_FLOOR - 1
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((element, test), {})


def test_dispatch_duplicate_heavy_element_and_test():
    element = _objarr(["dup"] * 400 + ["other"] * 100)
    test = _objarr(["dup"] * 250 + ["missing"] * 50)
    got, stock = _assert_dispatched_equal((element, test), {})
    assert got[:400].all()
    assert not got[400:].any()


# ---------------------------------------------------------------------------
# 2. critical semantics: path dispatches but answers VIA stock internally
# ---------------------------------------------------------------------------

def test_nan_in_test_elements_same_nan_object_answers_via_stock():
    nan = float("nan")
    element = _objarr(_words(398, seed=17) + [nan, "zzz"])
    test = _objarr(_words(20, seed=18) + [nan])  # same nan object, both sides
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == PATH, (decision, reason)  # still dispatches...
    got, stock = _assert_dispatched_equal((element, test), {})  # ...but == stock exactly
    assert bool(got[-2]) == bool(stock[-2])


def test_nan_in_test_elements_distinct_nan_objects_answers_via_stock():
    element = _objarr(_words(398, seed=19) + [float("nan"), "zzz"])
    test = _objarr(_words(20, seed=20) + [float("nan")])  # a DIFFERENT nan object
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == PATH, (decision, reason)
    got, stock = _assert_dispatched_equal((element, test), {})
    assert bool(got[-2]) == bool(stock[-2])


def test_nan_in_element_only_answers_via_stock():
    element = _objarr(_words(398, seed=21) + [float("nan"), float("nan")])
    test = _objarr(_words(50, seed=22))
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == PATH, (decision, reason)
    _assert_dispatched_equal((element, test), {})


def test_unhashable_list_member_typeerror_fallback_no_runtimewarning():
    element = _objarr(_words(398, seed=23) + [[1, 2], "zzz"])
    test = _objarr(_words(50, seed=24))
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == PATH, (decision, reason)  # dispatches: predicate can't see the list
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = np.isin(element, test)
        stock = _stock(element, test)
    assert np.array_equal(got, stock)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_unhashable_list_in_test_elements_typeerror_fallback():
    element = _objarr(_words(400, seed=25))
    test = _objarr(_words(48, seed=26) + [[1, 2], {1: 2}])
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == PATH, (decision, reason)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = np.isin(element, test)
        stock = _stock(element, test)
    assert np.array_equal(got, stock)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_decimal_nan_object_answers_via_stock():
    dnan = Decimal("NaN")
    element = _objarr(_words(398, seed=27) + [dnan, "zzz"])
    test = _objarr(_words(50, seed=28) + [dnan])
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == PATH, (decision, reason)
    got, stock = _assert_dispatched_equal((element, test), {})
    assert bool(got[-2]) == bool(stock[-2])


# ---------------------------------------------------------------------------
# 3. refusal / routing-to-other-path routes
# ---------------------------------------------------------------------------

def test_routing_unicode_udtype_pair_goes_to_stock():
    rng = np.random.default_rng(29)
    element = np.array([VOCAB[i] for i in rng.integers(0, 50, size=350)])
    test = np.array(VOCAB[:10])
    assert element.dtype.kind == "U"
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((element, test), {})


def test_routing_stringdtype_pair_goes_to_isin_string_hash_not_object():
    element = np.array(_words(400, seed=30), dtype=STR_DTYPE)
    test = np.array(_words(50, seed=31), dtype=STR_DTYPE)
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == OTHER_PATH, (decision, reason)
    assert decision != PATH


def test_refusal_int64_dtype_pair():
    rng = np.random.default_rng(32)
    element = rng.integers(0, 100, size=400).astype(np.int64)
    test = rng.integers(0, 100, size=50).astype(np.int64)
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision != PATH, (decision, reason)
    _assert_refused_equal((element, test), {})


def test_refusal_2d_object_array():
    element = _objarr(_words(400, seed=33)).reshape(20, 20)
    test = _objarr(_words(50, seed=34))
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((element, test), {})


def test_refusal_kind_sort_explicit():
    element = _objarr(_words(400, seed=35))
    test = _objarr(_words(50, seed=36))
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == PATH  # sanity: same inputs would dispatch without kind
    _assert_refused_equal((element, test), {"kind": "sort"})


def test_refusal_below_floor():
    element = _objarr(_words(200, seed=37))
    test = _objarr(_words(50, seed=38))
    assert element.size + test.size == 250 < SIZE_FLOOR
    decision, reason = GEARBOX.decide(OP, (element, test), {})
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((element, test), {})


def test_kill_switch_restores_stock_routing():
    element = _objarr(_words(400, seed=39))
    rng = np.random.default_rng(40)
    test = _objarr(rng.choice(VOCAB, size=50, replace=False).tolist())
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
