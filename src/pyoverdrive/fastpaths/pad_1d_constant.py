"""Fast path: numpy.pad on small 1-D arrays in constant mode.

Provenance (OPP-000057): np.pad is a Python driver - it normalizes
pad_width through _as_pairs, broadcasts the constant, allocates with
np.empty and then writes each pad region through generic N-d slice
machinery. On a small array that fixed preamble is the entire cost:
stock takes ~6.4 us at n=64 regardless of dtype or size, while the same
data movement written directly takes well under a microsecond.

Measured (BATCH13-CAL, fp 9bbe7063c555, idle box, 0% load, numpy 2.5.2),
driving the SHIPPED route with its predicate priced in, and quoting the
CONSUMED margin - the padded array is summed before the clock stops, for
the reason in the next paragraph:

    output length      14      70     262    1006    4006    8006   16006
    no constant      4.57x   4.46x   4.23x   3.54x   2.50x   1.88x   1.45x
    constant given   2.56x   2.58x   2.47x   2.35x   2.07x   1.85x   1.61x

Every measured dtype lands in the same band (3.6-4.7x at n=64), which is
the expected shape: the saving is fixed Python machinery, not element
work.

THE BARE NUMBER IS NOT THE NUMBER, and on this path the gap is enormous.
Timing only the pad call reports 14.8x at n=64, and 342x for a small
array with a huge zero pad, because np.zeros hands back calloc pages the
caller has not faulted in yet. Consume the result and that 342x becomes
0.86x - a REGRESSION dressed as the largest speedup in the project.
Hence OUTPUT_CAP, and hence it being set on the length of the RESULT
rather than of the input: a tiny array with an enormous pad is exactly
the shape that looks best and performs worst. BATCH13-CAL keeps those
refused cells, measured both ways, so the crossing is standing evidence
rather than a claim.

Correctness contract:
- Applies to pad(a, pad_width) and pad(a, pad_width, "constant"), with
  constant_values optional, where a is a plain 1-D ndarray (exactly
  np.ndarray - subclasses refused) of a measured dtype, both pad widths
  are non-negative Python/numpy integers, and the resulting length is at
  most OUTPUT_CAP. Everything else stays on stock.
- The constant is normalized to a NUMPY SCALAR, exactly as stock's
  _as_pairs does (it returns x.ravel()[i], so np.float64(nan) rather than
  the Python float or a 0-d array). All three spellings behave
  differently and only the middle one is right:

      out[k:] = -1                    OverflowError   (Python scalar)
      out[k:] = np.int64(-1)          wraps to 255    (numpy scalar, stock)
      out[k:] = np.asarray(-1)        wraps to 255    (0-d array)

      out[k:] = float('nan')          ValueError      (Python scalar)
      out[k:] = np.float64(nan)       ValueError      (numpy scalar, stock)
      out[k:] = np.asarray(np.nan)    writes INT_MIN  (0-d array!)

  The 0-d array agrees with stock on the uint wrap and DISAGREES on NaN
  into an integer array, where it silently writes INT_MIN under a mere
  RuntimeWarning instead of raising. An early version of this path used
  np.asarray and turned that stock exception into a wrong answer;
  hypothesis caught it. Using numpy's own normalization inherits its
  casting rules rather than reimplementing them.
- The np.zeros route is used ONLY when constant_values is absent. It is
  not merely an optimization for a zero constant: pad(a, 1,
  constant_values=-0.0) fills with NEGATIVE zero, np.zeros fills with
  positive zero, and np.array_equal cannot tell them apart. An explicit
  constant always goes through the assignment route, so signed zero, NaN
  and every other bit pattern come out exactly as stock produces them.

Refused, each because stock does something the direct route does not:
- ndim != 1: a 0-d input returns a value and a 2-D input pads every axis.
- non-integer pad_width, including integral floats like 2.0, where stock
  raises TypeError.
- any mode other than "constant", and any unknown keyword (stock raises).
- non-numeric dtypes. A string array is padded by stock with the STRING
  "0", not with the empty string np.zeros produces - shapes and dtypes
  match, so only a value comparison catches it.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=pad_1d_constant or
pyoverdrive.disable_path("pad_1d_constant").
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath

_DTYPES = frozenset(
    np.dtype(t)
    for t in (
        np.float64,
        np.float32,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
        np.bool_,
        np.complex128,
        np.complex64,
    )
)

# On the LENGTH OF THE RESULT, not of the input. See the module docstring:
# the calloc-page effect makes a small array with a huge pad look like the
# best cell in the project while actually being slower once used.
OUTPUT_CAP = 16_384

# constant_values may be any numeric kind; strings, objects and voids are
# refused because their casting is not the assignment we perform
_CONST_KINDS = frozenset("biufc")


def _pair(value, integral: bool):
    """Normalize a pad_width / constant_values spelling to (before, after).

    Returns None when the spelling is one stock accepts and we do not, or
    one stock rejects outright - either way the call belongs on stock.
    """
    if isinstance(value, bool):
        return None

    # The whole win here is that stock's preamble is the entire cost, so
    # this normalization must not become a preamble of its own. It runs
    # TWICE per call - once for the predicate, once inside run - and a
    # single np.asarray at n=64 costs about as much as the pad itself.
    # These two spellings are almost all real traffic and skip numpy
    # entirely; everything else falls through to the general path below.
    if integral:
        if type(value) is int:
            return value, value
        if type(value) is tuple and len(value) == 2:
            first, second = value
            if type(first) is int and type(second) is int:
                return first, second

    arr = np.asarray(value)
    if integral:
        # stock raises TypeError on a float pad_width, 2.0 included, so
        # matching that by refusing is exactly right
        if arr.dtype.kind not in ("i", "u"):
            return None
    elif arr.dtype.kind not in _CONST_KINDS:
        return None

    # NUMPY SCALARS, never 0-d arrays. Stock's _as_pairs returns x.ravel()[i],
    # i.e. np.float64(nan) / np.int64(-1), and the distinction is not
    # cosmetic: assigning np.float64(nan) into an int array RAISES
    # ValueError the way stock does, while assigning the equivalent 0-d
    # ARRAY silently writes INT_MIN with only a RuntimeWarning. Getting
    # this wrong turns a stock exception into a wrong answer.
    shape = arr.shape
    if shape == ():
        first = second = arr.ravel()[0]
    elif shape == (1,):
        first = second = arr[0]
    elif shape == (2,):
        first, second = arr[0], arr[1]
    elif shape == (1, 1):
        first = second = arr[0, 0]
    elif shape == (1, 2):
        first, second = arr[0, 0], arr[0, 1]
    else:
        return None
    return first, second


def _parse(args: tuple, kwargs: dict):
    """(array, before, after, constant) or None when this call is stock's.

    ``constant`` is None when no constant_values was supplied, which is
    the only case the np.zeros route may serve.
    """
    if not 1 <= len(args) <= 3:
        return None
    if set(kwargs) - {"array", "pad_width", "mode", "constant_values"}:
        return None
    if "array" in kwargs:
        return None  # positional-or-keyword collision territory; leave it

    array = args[0]
    if len(args) >= 2:
        if "pad_width" in kwargs:
            return None  # duplicate: stock raises TypeError
        pad_width = args[1]
    elif "pad_width" in kwargs:
        pad_width = kwargs["pad_width"]
    else:
        return None  # stock raises: pad_width is required

    if len(args) == 3:
        if "mode" in kwargs:
            return None
        mode = args[2]
    else:
        mode = kwargs.get("mode", "constant")
    # a callable mode, or any other named mode, is stock's business
    if not isinstance(mode, str) or mode != "constant":
        return None

    if type(array) is not np.ndarray:
        return None
    if array.ndim != 1 or array.dtype not in _DTYPES:
        return None

    widths = _pair(pad_width, integral=True)
    if widths is None:
        return None
    before, after = int(widths[0]), int(widths[1])
    if before < 0 or after < 0:
        return None  # stock raises ValueError; let it
    if array.shape[0] + before + after > OUTPUT_CAP:
        return None

    if "constant_values" in kwargs:
        constant = _pair(kwargs["constant_values"], integral=False)
        if constant is None:
            return None
        # A zero-width pad on BOTH sides writes the constant only into empty
        # slices, and an empty-slice assignment never performs the cast - so
        # pad(int_array, 0, constant_values=nan) would quietly succeed here
        # while stock raises ValueError. Found by hypothesis, not by hand.
        # The case is degenerate (a pad that pads nothing, with an explicit
        # fill), so refusing it costs nothing and needs no extra work on the
        # hot path. A zero width on ONE side is still served: the other side
        # is non-empty, so the cast happens and any error surfaces there.
        if before == 0 and after == 0:
            return None
    else:
        constant = None
    return array, before, after, constant


def _applicable(args: tuple, kwargs: dict) -> bool:
    return _parse(args, kwargs) is not None


def _run(*args, **kwargs):
    array, before, after, constant = _parse(args, kwargs)
    n = array.shape[0]

    if constant is None:
        out = np.zeros(n + before + after, dtype=array.dtype)
        out[before:before + n] = array
        return out

    out = np.empty(n + before + after, dtype=array.dtype)
    # constant[i] is already a NUMPY SCALAR from _pair, which is exactly what
    # stock assigns. Do not wrap it in np.asarray here: that turns a raised
    # ValueError into a silently wrong INT_MIN. See _pair.
    out[:before] = constant[0]
    out[before:before + n] = array
    out[before + n:] = constant[1]
    return out


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="pad_1d_constant",
            op="numpy.pad",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000057",
                "source": "shortlist panel, batch 11 (docs/research/batch11-shortlist.md)",
                "license": "own implementation; no third-party snippet",
                "comparison_mode": "bit-identical",
            },
        )
    )
