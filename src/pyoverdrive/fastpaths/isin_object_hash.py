"""Fast path: numpy.isin on object arrays via a Python hash set.

Provenance (OPP-000036): numpy/numpy#14997 - object dtype
unconditionally takes in1d's O(n*m) broadcast-equality path (the
reporter: ~10-15 s where a set does ~0.1 s). Sibling of the shipped
isin_string_hash (OPP-000023, 2317x): identical mechanism, object
operands.

Measured (OPP-000036 + BATCH5-CAL batteries, fp 9bbe7063c555, idle box,
0-1% load), for the route AS SHIPPED (hazard scan included): 262x at
30_000 x 3_000 strings, 169x ints, 11.2x at 1_000 x 100, 5.8x at 550
combined, 2.96x at 300 combined; 105 combined straddles (1.32x), hence
the floor.

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

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=isin_object_hash or
pyoverdrive.disable_path("isin_object_hash").
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

SIZE_FLOOR = 300  # combined; 2.96x measured there, 105 straddles


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
