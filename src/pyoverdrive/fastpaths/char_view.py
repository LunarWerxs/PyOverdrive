"""Fast paths: numpy.sort / numpy.unique on single-character U1/S1 arrays
via an integer reinterpretation of the same buffer.

Provenance (OPP-000024): numpy/numpy#24821 - numpy's string sort kernel is
2-3x slower than its int kernel on identical data (ngoldbaum's py-spy
analysis, quoted in the issue), so any U1/S1 call that sorts internally
pays it. The route is the reporter's own three-line snippet: view the
buffer as integers, run the operation there, view the values back.

Why the view is exact, not approximate: a U1 element is one UCS4 code
unit, numpy orders U1 by that unit, and every valid codepoint
(<= 0x10FFFF) is non-negative in int32 - so the int32 view is a strictly
monotone bijection. S1 orders as unsigned bytes, so its view target is
uint8 and NEVER int8 (a signed view would misorder bytes >= 0x80; the
CHARVIEW-CAL hibyte case pins this). Sort order, equality classes, and
element positions are all preserved, so sorted output, unique values,
counts, indices, and inverse are identical by construction - and both
batteries (benchmarks/results/OPP-000024/, CHARVIEW-CAL/) checked exact
equality on every cell, high-byte S1, Greek U1, and 0x00 included.
('' and '\\x00' share buffer value 0 in these dtypes, so the bijection is
untouched by that equivalence - both routes see the same value.)

Measured wins (fp 9bbe7063c555, idle box, 0% load, numpy 2.5.2):
- sort U1: 3.06x at n=1000, 14.1x at 3000, 33.1x at 10_000, 25.1x at 1e6
  (n=100 is a wash, 1.05x, hence the floor)
- sort S1: 2.06-2.53x at n=1000..100_000
- unique return_counts: U1 1.50x at n=300 up to 26.6x at 10_000;
  S1 2.61-2.75x from n=1000
- unique return_index/inverse: U1 1.42-2.00x, S1 2.38-8.72x from n=1000
- plain unique (no flags): U1 1.43-1.55x and S1 1.54-1.73x from
  n=10_000 (OPP-000024 battery; 1.31x at n=1000 straddles the min-win,
  so the plain floor sits at the clean 10_000 measurement)

Correctness contract:
- sort_char_view: sort(a[, axis]) on a plain 1-D native-byte-order U1 or
  S1 ndarray, axis absent/None/0/-1, kind/order/stable absent or None,
  n >= 1000. Multi-char strings (U>1/S>1), byte-swapped dtypes, 2-D+,
  and explicit kind all stay on stock (the view identity does not even
  hold for multi-char).
- unique_char_view: unique(ar[, return_index, return_inverse,
  return_counts]) on the same array class, axis absent/None, no other
  kwargs. Floors per route, from the cells above: any index/inverse ->
  1000; else counts -> 300 (U1) / 1000 (S1); else plain -> 10_000.
  return_index forces a stable argsort in stock on both representations,
  and inverse/counts depend only on equality classes, so flag outputs are
  identical, not merely equivalent.

Comparison mode: bit-identical (spec section 9). Kill switches:
PYOVERDRIVE_DISABLE=sort_char_view / unique_char_view, or
pyoverdrive.disable_path(...).

Implementation note: never calls np.sort / np.unique by name (both are
patched); stock comes from GEARBOX.stock_fn.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

# view targets, keyed by (dtype.kind, itemsize); values from the record
_INT_VIEW = {("U", 4): np.dtype(np.int32), ("S", 1): np.dtype(np.uint8)}

SORT_FLOOR = 1_000
UNIQUE_PLAIN_FLOOR = 10_000
UNIQUE_COUNTS_FLOOR = {"U": 300, "S": 1_000}
UNIQUE_IDXINV_FLOOR = 1_000

_BOOL_TYPES = (bool, np.bool_)


def _char_view_dtype(a):
    """The integer view dtype for a supported array, else None."""
    if type(a) is not np.ndarray or a.ndim != 1:
        return None
    dt = a.dtype
    if not dt.isnative:
        return None
    return _INT_VIEW.get((dt.kind, dt.itemsize))


# -- numpy.sort -------------------------------------------------------------


def _sort_applicable(args: tuple, kwargs: dict) -> bool:
    if not 1 <= len(args) <= 2:
        return False
    if set(kwargs) - {"axis", "kind", "order", "stable"}:
        return False
    if len(args) == 2 and "axis" in kwargs:
        return False  # duplicate axis: stock raises TypeError
    axis = args[1] if len(args) == 2 else kwargs.get("axis", -1)
    if axis not in (None, 0, -1):
        return False
    if kwargs.get("kind") is not None:
        return False
    if kwargs.get("order") is not None:
        return False
    if kwargs.get("stable") is not None:
        return False
    a = args[0]
    return _char_view_dtype(a) is not None and a.size >= SORT_FLOOR


def _sort_run(a, axis=-1, kind=None, order=None, *, stable=None):
    view_dt = _char_view_dtype(a)
    return GEARBOX.stock_fn("numpy.sort")(a.view(view_dt)).view(a.dtype)


# -- numpy.unique -----------------------------------------------------------


def _unique_normalize(args: tuple, kwargs: dict):
    """(a, ri, rv, rc) for an in-contract call, else None."""
    if not 1 <= len(args) <= 4:
        return None
    allowed = {"return_index", "return_inverse", "return_counts", "axis"}
    if set(kwargs) - allowed:
        return None
    names = ("return_index", "return_inverse", "return_counts")
    flags = list(args[1:]) + [None] * (4 - len(args))
    for i, name in enumerate(names):
        if flags[i] is not None and name in kwargs:
            return None  # duplicate: stock raises TypeError
        if flags[i] is None:
            flags[i] = kwargs.get(name, False)
    if kwargs.get("axis") is not None:
        return None
    if not all(isinstance(f, _BOOL_TYPES) for f in flags):
        return None
    return args[0], bool(flags[0]), bool(flags[1]), bool(flags[2])


def _unique_applicable(args: tuple, kwargs: dict) -> bool:
    norm = _unique_normalize(args, kwargs)
    if norm is None:
        return False
    a, ri, rv, rc = norm
    if _char_view_dtype(a) is None:
        return False
    if ri or rv:
        floor = UNIQUE_IDXINV_FLOOR
    elif rc:
        floor = UNIQUE_COUNTS_FLOOR[a.dtype.kind]
    else:
        floor = UNIQUE_PLAIN_FLOOR
    return a.size >= floor


def _unique_run(
    ar,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
):
    view_dt = _char_view_dtype(ar)
    res = GEARBOX.stock_fn("numpy.unique")(
        ar.view(view_dt),
        return_index=bool(return_index),
        return_inverse=bool(return_inverse),
        return_counts=bool(return_counts),
    )
    if isinstance(res, tuple):
        return (res[0].view(ar.dtype),) + res[1:]
    return res.view(ar.dtype)


_PROVENANCE = {
    "opportunity": "OPP-000024",
    "source": "https://github.com/numpy/numpy/issues/24821",
    "license": "reporter's own pure-numpy snippet from the public issue body",
    "comparison_mode": "bit-identical",
}


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="sort_char_view",
            op="numpy.sort",
            applicable=_sort_applicable,
            run=_sort_run,
            provenance=_PROVENANCE,
        )
    )
    gearbox.register(
        FastPath(
            name="unique_char_view",
            op="numpy.unique",
            applicable=_unique_applicable,
            run=_unique_run,
            provenance=_PROVENANCE,
        )
    )
