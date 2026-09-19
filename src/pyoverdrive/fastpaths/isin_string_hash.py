"""Fast path: numpy.isin on StringDType via a Python hash set.

Provenance (OPP-000023): numpy/numpy#32161 (open, reported by a numpy
maintainer) - isin on StringDType falls back to the O(n*m) hasobject
path; the reporter measures 2.94 s where fixed-width U dtype takes 20 ms.
Dyno reproduced far beyond that derived ~144x: a plain Python
set-membership probe wins 2318x at the reporter's own shape (8.34 s ->
3.60 ms), 209-299x at n=3000 across cardinalities, 12.8x on a
small-needle disjoint case, and 90x on 2000-char strings - dominating
the thread's own U-cast idea everywhere (which collapses to 2.7x on long
strings because the fixed-width copy explodes). Evidence:
benchmarks/results/OPP-000023/9bbe7063c555.json (idle box, 0% load).

Correctness contract:
- Applies only to isin(element, test_elements[, assume_unique=..,
  invert=..]) where both operands are plain 1-D ndarrays of the DEFAULT
  StringDType (dtype.kind 'T' and no na_object configured: an na-carrying
  StringDType has NaN-like comparison semantics this route does not
  replicate, detected via hasattr(dtype, 'na_object')), kind is absent or
  None (an explicit kind= is the user choosing stock's strategy), and
  combined size >= 300 (the smallest measured winning input, 3.7x).
- assume_unique is accepted with either value: it is a performance hint
  with no effect on the result. invert is accepted and applied as a
  boolean negation of the membership mask, which is exact.
- The route builds set(test_elements.tolist()) and probes each element:
  Python string equality is exactly numpy's string equality for
  StringDType without na_object, so the bool result is bit-identical to
  stock's - with one guarded exception found by this family's
  differential battery: stock numpy (measured on 2.4.5 AND 2.5.2)
  MISSES strings made only of NUL characters even when present in
  test_elements (embedded NULs match fine), where the hash route
  answers correctly. Diverging from stock, even to be right, breaks the
  transparent-drop-in contract on an input class whose stock behavior
  is a bug in flux, so inputs containing pure-NUL strings are refused
  and stock keeps answering for them. The pure-NUL DETECTOR is chosen
  at import by probing this numpy's string machinery (see the detector
  block below): numpy 2.5 fixed str_len's NUL handling, which silently
  blinded the 2.4-era detector while the isin miss remained - caught by
  the property net on 2.5.2 (Linux leg + fresh-venv probe, 2026-08-24).

assume_unique= is ACCEPTED here and REFUSED by intersect_sorted, which is
one rule and not two. The keyword is a promise numpy does not verify, and
membership is idempotent under duplication, so a false promise cannot
change this answer - verified with the promise deliberately false
(duplicates in both operands), where dispatched matches stock exactly for
True and False alike. intersect1d's answer DOES change under a false
promise, so it refuses. See intersect_sorted's contract.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=isin_string_hash or
pyoverdrive.disable_path("isin_string_hash").
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath

SIZE_FLOOR = 300  # combined elements; smallest measured winning case

# TEST_FLOOR: see isin_object_hash for the mechanism, which is the same one -
# stock's in1d answers a small test_elements with a chain of vectorized
# equalities, one full pass each, while this route pays a Python-level hash
# per element of the big operand. The four-axis sweep caught this path at
# 0.05x, the worst reading in the package, by moving the operand ratio at
# constant combined size, which the shipped gate could not see.
#
# The number differs from the object path's 12, and NOT for a tidy reason:
# this crossing MOVES with the element count where the object one does not
# (idle Intel box, fp 9bbe7063c555, 2026-08-25, tools/probe_isin_ratio.py):
#
#   elements     8     12     16     24     32     48     64     96
#      300     0.58x  0.81x  1.02x  1.42x  1.90x  2.60x  3.33x  4.73x
#    1,000     0.44x  0.59x  0.75x  1.09x  1.41x  2.03x  2.62x  3.91x
#   10,000     0.36x  0.48x  0.63x  0.90x  1.10x  1.62x  2.15x  3.26x
#  100,000     0.31x  0.43x  0.56x  0.79x  1.07x  1.53x  1.94x  2.90x
#
# So the crossing walks 16 -> 24 -> 32 -> 32 as the array grows, and 32 is
# only 1.05-1.10x at the top - a wash, not a promise. 48 is the first column
# every measured element count clears with a margin, worst 1.53x, which is
# the same margin the object floor carries.
#
# A fitted curve would keep more of the small-array region (24 already wins
# 1.42x at 300 elements) and is deliberately not used: four points do not
# justify an exponent, and a flat number every measured cell clears is a
# better promise than a curve that is right on average. The forfeited region
# is small arrays with small vocabularies, where the whole call is fast.
TEST_FLOOR = 48

# stock's bug class (measured on numpy 2.4.5 AND 2.5.2): isin misses
# strings consisting ONLY of NUL characters ("\x00", "\x00\x00"), while
# embedded NULs ("a\x00b") match fine. The DETECTOR for that class is
# itself version-sensitive: on 2.4.x str_len reports 0 for pure-NUL
# strings (so pure-NUL == zero-len and != ""), but numpy 2.5 fixed
# str_len (it now counts NULs) while leaving the isin miss in place -
# which silently blinded the 2.4-style detector and let this path
# diverge from stock (caught by a 2.5.2 probe, 2026-08-24). So the
# detector is now CHOSEN AT IMPORT by running every candidate against a
# probe array with known answers and keeping the first one that
# classifies it exactly; if no vectorized detector survives on some
# future numpy, a plain Python scan (slow but unfoolable) is the
# fallback. Never trust string machinery here without probing it first.

_NUL = "\x00"
# a Python "\x00" scalar C-truncates to "" inside the string ufuncs on
# BOTH 2.4.5 and 2.5.2 (probed: count(a, "\x00") behaves as count(a, ""));
# a 0-d StringDType array carries the NUL through intact
_NUL_SCALAR = np.array(_NUL, dtype=np.dtypes.StringDType())


def _detector_count(a: np.ndarray) -> np.ndarray:
    # numpy >= 2.5: str_len counts NULs and count() accepts an np-scalar
    # NUL needle correctly
    lengths = np.strings.str_len(a)
    return (lengths > 0) & (np.strings.count(a, _NUL_SCALAR) == lengths)


def _detector_zerolen(a: np.ndarray) -> np.ndarray:
    # numpy 2.4.x: pure-NUL reads as length 0 yet compares != ""
    return (np.strings.str_len(a) == 0) & (a != "")


def _detector_python(a: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (len(s) > 0 and s.count(_NUL) == len(s) for s in a.tolist()),
        dtype=bool,
        count=a.size,
    )


def _pick_detector():
    probe = np.array(
        [_NUL, _NUL * 2, "a" + _NUL + "b", "", "x"], dtype=np.dtypes.StringDType()
    )
    expected = np.array([True, True, False, False, False])
    for det in (_detector_count, _detector_zerolen):
        try:
            if bool(np.array_equal(det(probe), expected)):
                return det
        except Exception:
            continue
    return _detector_python


_PURE_NUL_MASK = _pick_detector()


def _has_pure_nul(a: np.ndarray) -> bool:
    return bool(_PURE_NUL_MASK(a).any())


def _string_default_dtype(a) -> bool:
    return (
        type(a) is np.ndarray
        and a.ndim == 1
        and a.dtype.kind == "T"
        and not hasattr(a.dtype, "na_object")
    )


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 2:
        return False
    extra = set(kwargs) - {"assume_unique", "invert", "kind"}
    if extra:
        return False
    if kwargs.get("kind") is not None:
        return False
    if not isinstance(kwargs.get("invert", False), (bool, np.bool_)):
        return False
    element, test = args
    if not (_string_default_dtype(element) and _string_default_dtype(test)):
        return False
    if test.size < TEST_FLOOR:
        return False
    if element.size + test.size < SIZE_FLOOR:
        return False
    # stock numpy (2.4.5 measured) has a genuine bug here: isin misses
    # strings made only of NUL characters even when present in
    # test_elements. The hash route answers correctly, which is still a
    # divergence from stock - and stock's behavior on this input class is
    # a bug likely to change under us. Bug-for-bug faithfulness wins:
    # refuse inputs containing pure-NUL strings so stock always answers.
    return not _has_pure_nul(element) and not _has_pure_nul(test)


def _run(element, test_elements, assume_unique=False, invert=False, kind=None):
    lookup = set(test_elements.tolist())
    mask = np.fromiter(
        (s in lookup for s in element.tolist()), dtype=bool, count=element.size
    )
    return ~mask if invert else mask


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="isin_string_hash",
            op="numpy.isin",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000023",
                "source": "https://github.com/numpy/numpy/issues/32161",
                "license": "hash-set membership, standard technique; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
