"""Fast path: numpy.apply_along_axis with a KNOWN reducer, served as the
equivalent axis= reduction.

Provenance (OPP-000054): apply_along_axis is documented as a convenience
wrapper, and its implementation is a Python loop over an ndindex of the
non-axis dimensions - one Python-level call to func1d per 1-D slice.
When func1d is one of NumPy's own reductions, that loop reproduces, one
slice at a time and in Python, exactly what the reduction's own axis=
argument does in one vectorized C call. The gap is not a rounding
detail or a cache effect: it is the interpreter, and it grows with the
number of slices.

Measured (both benchmark machines): mean 103.9-292.1x (dev box, fp
8f8198d9abab) / 109.4-179.2x (idle box, fp 9bbe7063c555, 0-1% load,
numpy 2.5.2); sum 80.9-234.9x; max 54.7-190.7x; median 8.4-20.3x. The
margin scales with slice COUNT, so the floor is a slice count, not an
element count.

WHY THIS IS SAFE, AND EXACTLY WHERE IT IS NOT. The values agree by
construction (same reduction, same slices), so the risk lives entirely
in the wrapper's edges. Each was read out of numpy's implementation and
then tested; each is a refusal here, not an assumption:

- Zero-size dimensions. With a zero-length non-axis dimension stock
  raises ValueError("Cannot apply_along_axis when any iteration
  dimensions are zero"); the axis= form returns an empty array. With a
  zero-length AXIS dimension stock calls func1d on empty slices, whose
  warning behavior (one per slice, deduplicated by the warnings
  registry) the single vectorized call cannot reproduce. Any zero
  dimension therefore refuses to stock.
- Subclasses. On np.matrix, stock's own answer is shape-wrong (matrix
  slices never become 1-D, a long-standing numpy quirk) - so the axis=
  form is not "equivalent", it is different-and-arguably-better, which
  is exactly what a transparent accelerator must not do. Masked arrays
  and other subclasses carry their own reduction semantics. Only
  type(arr) is exactly np.ndarray is served.
- 1-D input. Stock returns a 0-d ndarray where the axis= form returns a
  bare NumPy scalar - same value, different type. A 1-D array is also
  exactly ONE slice, so it is below the slice floor by construction and
  there is nothing to win; the path refuses ndim < 2 outright rather
  than carrying a re-wrap branch that can never run.
- func1d identity. Matching is by object identity against a fixed table
  (not by name, not by callable heuristics), and any extra *args or
  **kwargs refuse: those change what func1d computes.

Correctness contract: exactly apply_along_axis(func1d, axis, arr) with
func1d one of the served reducers, no extra arguments, arr a plain
ndarray of a listed dtype with no zero-length dimension and a valid
axis - and, for the order-sensitive reducers, the LAST axis (see the
_EXACT_ANY_AXIS / _ORDER_SENSITIVE_LAST_AXIS split, which is a measured
floating-point-associativity result, not a stylistic one). Results are
bit-identical to stock for every served call.

Comparison mode: bit-identical (spec section 9). Kill switch:
apply_along_axis_reduce.
"""

from __future__ import annotations

import math

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath, StockRaised

# Reducers served, by NumPy name. Matching is by object IDENTITY against
# numpy's own attribute (see _lookup): a user function merely named
# "mean" is never served.
#
# THE SPLIT IS A MEASURED NUMERICS RESULT, not tidiness. Summing a 1-D
# slice and reducing the same elements with axis= are the same arithmetic
# only when they accumulate in the same ORDER. For the LAST axis each
# slice is contiguous and the two routes agree bit-for-bit. For any other
# axis NumPy's reduction walks whole rows across the buffer instead of
# one strided column at a time, and floating-point addition is not
# associative: measured last-ulp disagreement on (40, 500) axis=0 for
# sum, mean, std and var. So order-sensitive reducers are served on the
# last axis only; the rest are order-INDEPENDENT (exact integer/boolean
# comparisons, or a partition over the same set of values) and are served
# on any axis. test_apply_along_axis_reduce_differential re-proves this
# split wherever the suite runs.
_EXACT_ANY_AXIS = ("max", "min", "median", "any", "all", "ptp", "argmax", "argmin")
_ORDER_SENSITIVE_LAST_AXIS = ("mean", "sum", "std", "var", "prod")
_NAMES = _EXACT_ANY_AXIS + _ORDER_SENSITIVE_LAST_AXIS

# id(callable) -> the STOCK reducer to run. Rebuilt whenever the gearbox
# patches or unpatches, because while PyOverdrive is enabled the caller's
# np.mean IS our wrapper, and an id-table built before patching would
# stop matching (silently disabling this path, which is exactly how the
# first build of it failed its own selfcheck).
#
# The value is deliberately the STOCK reducer, never the live wrapper:
# it makes this path's own result exactly stock-apply_along_axis's, with
# no second dispatch, and keeps the bit-identical claim independent of
# whatever other paths the user has enabled.
_LOOKUP: dict[int, object] = {}
_LOOKUP_GENERATION = -1


def _lookup() -> dict:
    global _LOOKUP, _LOOKUP_GENERATION
    gen = GEARBOX.generation
    if gen != _LOOKUP_GENERATION:
        table: dict[int, object] = {}
        for name in _NAMES:
            op = f"numpy.{name}"
            try:
                stock = GEARBOX.stock_fn(op)
            except (AttributeError, ValueError):  # pragma: no cover - numpy shape
                continue
            entry = (stock, name in _EXACT_ANY_AXIS)
            table[id(stock)] = entry
            live = getattr(np, name, None)
            if live is not None:
                table[id(live)] = entry  # our wrapper resolves to stock
        _LOOKUP, _LOOKUP_GENERATION = table, gen
    return _LOOKUP

# dtypes whose reductions are plain and well-understood here; anything
# else (object, complex, datetime, StringDType, ...) stays on stock
_DTYPES = frozenset(
    np.dtype(t)
    for t in (
        np.float64, np.float32, np.int64, np.int32, np.int16, np.int8,
        np.uint64, np.uint32, np.uint16, np.uint8, np.bool_,
    )
)

# Below this many 1-D slices the Python loop is short enough that the
# dispatch tax (ADR-0003, ~300ns) is a real fraction of the win.
# Measured: the margin is already 50x+ at a few hundred slices, so the
# floor is set for tax safety, not for the crossing.
SLICE_MIN = 16


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 3 or kwargs:
        return False  # extra *args/**kwargs change what func1d computes
    func1d, axis, arr = args
    entry = _lookup().get(id(func1d))
    if entry is None:
        return False
    if type(arr) is not np.ndarray:  # subclasses have their own semantics
        return False
    if arr.dtype not in _DTYPES:
        return False
    if type(axis) is not int:
        return False
    nd = arr.ndim
    if nd < 2:
        return False  # one slice: below the floor, and stock's 0-d return type
    if not -nd <= axis < nd:
        return False  # stock raises its own AxisError
    if 0 in arr.shape:
        return False  # zero-length dims: see the module docstring
    if not entry[1] and axis % nd != nd - 1:
        return False  # order-sensitive reducer off the last axis
    slices = math.prod(arr.shape) // arr.shape[axis]
    return slices >= SLICE_MIN


def _run(func1d, axis, arr):
    reducer = _lookup()[id(func1d)][0]
    try:
        return reducer(arr, axis=axis)
    except Exception as exc:  # noqa: BLE001 - the reducer IS stock
        # This body does exactly one thing: call the STOCK reducer. If it
        # refuses this dtype (np.ptp on bool, say), stock's own
        # apply_along_axis would refuse it too, one slice at a time. That
        # is stock's behavior, not a fast-path failure, so it must not be
        # branded with the fast-path-failure warning.
        raise StockRaised(exc) from None


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="apply_along_axis_reduce",
            op="numpy.apply_along_axis",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000054",
                "source": "numpy.lib._shape_base_impl.apply_along_axis (Python loop over ndindex)",
                "license": "routes to numpy's own reductions; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
