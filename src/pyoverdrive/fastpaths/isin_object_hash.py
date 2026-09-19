"""Fast path: numpy.isin on object arrays via a Python hash set.

Provenance (OPP-000036): numpy/numpy#14997 - object dtype
unconditionally takes in1d's O(n*m) broadcast-equality path (the
reporter's comparison used a Python set instead). This is the same
hash-membership mechanism as isin_string_hash (OPP-000023), adapted to
object operands. Combined-size and test-set-size floors amortize conversion
and hashing without assuming every operand ratio benefits.

Hazards, all handled INSIDE the run (measured as part of its cost):
- NaN-like objects (x != x): Python's `in` matches them by IDENTITY
  where stock's equality path says nan != nan - the battery's same-NaN
  probe measured the divergence. The run scans both operands
  (tolist once, reused for the set) and hands the call to stock via
  stock_fn when any NaN-like is present. The result is therefore
  ALWAYS stock's result.
- Unhashable elements (lists, dicts): the set build raises TypeError;
  caught, handed to stock the same way (never through the dispatcher's
  warning path).
- Objects whose __eq__ disagrees with __hash__ violate Python's own
  contract; inputs honoring it (str/int/float/bool mixes measured) are
  exact. This is the same trust Python dict/set place in user objects.

Correctness contract:
- Applies only to isin(element, test_elements[, assume_unique, invert])
  where both operands are plain 1-D object-dtype ndarrays, kind absent
  or None, invert (if given) a bool, combined size >= 300. The result
  is bit-identical to stock by construction: hazard inputs are answered
  BY stock, clean inputs by set membership, which equals stock's
  broadcast == for hash/eq-consistent objects.

assume_unique= is ACCEPTED here and REFUSED by intersect_sorted, which is
one rule and not two. The keyword is a promise numpy does not verify, and
membership is idempotent under duplication, so a false promise cannot
change this answer - verified with the promise deliberately false
(duplicates in both operands), where dispatched matches stock exactly for
True and False alike. intersect1d's answer DOES change under a false
promise, so it refuses. See intersect_sorted's contract.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=isin_object_hash or
pyoverdrive.disable_path("isin_object_hash").

Historical calibration ratios are omitted because the NumPy version was not
recorded. See docs/research/2026-09-19-burndown.md for current measured
evidence and its version, hardware and load qualifications.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

SIZE_FLOOR = 300  # combined input size; conversion/hash setup amortization

# Combined size alone cannot express the few-test-elements corner. Stock
# can evaluate a short chain of vectorized equalities in C, while this route
# pays Python hashing and lookup per input element. The number of test
# elements therefore needs its own floor, independent of total input size.
# A numeric-dtype crossover formula does not represent Python-object costs.
# Reproducer: tools/probe_isin_ratio.py.
TEST_FLOOR = 12


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 2:
        return False
    if set(kwargs) - {"assume_unique", "invert", "kind"}:
        return False
    if kwargs.get("kind") is not None:
        return False
    if not isinstance(kwargs.get("invert", False), (bool, np.bool_)):
        return False
    element, test = args
    for a in (element, test):
        if type(a) is not np.ndarray or a.ndim != 1 or a.dtype != object:
            return False
    if test.size < TEST_FLOOR:
        return False
    return element.size + test.size >= SIZE_FLOOR


def _is_nanlike(x) -> bool:
    try:
        return bool(x != x)
    except Exception:
        return True  # exotic comparison behavior: let stock answer


def _run(element, test_elements, assume_unique=False, invert=False, kind=None):
    stock = GEARBOX.stock_fn("numpy.isin")
    te = test_elements.tolist()
    el = element.tolist()
    try:
        if any(_is_nanlike(x) for x in te) or any(_is_nanlike(x) for x in el):
            return stock(element, test_elements, assume_unique=assume_unique, invert=invert)
        lookup = set(te)
        mask = np.fromiter((s in lookup for s in el), dtype=bool, count=element.size)
    except TypeError:  # unhashable member: stock's semantics, stock's answer
        return stock(element, test_elements, assume_unique=assume_unique, invert=invert)
    return ~mask if invert else mask


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="isin_object_hash",
            op="numpy.isin",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000036",
                "source": "https://github.com/numpy/numpy/issues/14997",
                "license": "hash-set membership, standard technique; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
