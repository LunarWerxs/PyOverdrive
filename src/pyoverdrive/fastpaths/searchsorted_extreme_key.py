"""Fast path: numpy.searchsorted with a Python int key outside the array
dtype's range - an O(1) provably-identical answer for a call stock
computes in milliseconds.

Provenance (OPP-000039): numpy/numpy#29719 reported Python-int keys
~32000x slower than dtype-matched keys. The battery showed numpy
2.4/2.5 already fixed the IN-RANGE path (stock bisects in ~900 ns;
a cast route is pure overhead there, 0.77-0.81x, so it is NOT shipped),
but the OUT-OF-RANGE remnant is alive: np.searchsorted(int64_array,
2**70) measured 163 ms at n=100_000 on the idle box - a per-element
arbitrary-precision comparison walk - where the answer is provable in
O(1): every representable value compares below a key above the dtype's
max, so the insertion point is a.size for either side; symmetrically 0
for a key below the dtype's min. No value inspection, no tolerance:
the result equals stock's BY PROOF for any array contents, sorted or
not (both routes' outputs depend only on elementwise comparisons that
are uniformly decided). uint64-with-in-range-int stays on stock too:
stock's own answer there is WRONG (float64 promotion, the spun-off
numpy/numpy#29727) and this project replicates stock, not truth.

Correctness contract:
- Applies only to searchsorted(a, v[, side]) where a is a plain 1-D
  integer-dtype ndarray, v is a Python int (bool excluded) strictly
  outside a.dtype's representable range, side absent/'left'/'right',
  and no sorter. Returns np.intp(a.size) for v above the range,
  np.intp(0) below - the same scalar type stock returns.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=searchsorted_extreme_key or
pyoverdrive.disable_path("searchsorted_extreme_key").
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath


_BOUNDS_CACHE: dict = {}


def _bounds(dtype):
    got = _BOUNDS_CACHE.get(dtype)
    if got is None:  # np.iinfo construction costs ~a microsecond per call
        info = np.iinfo(dtype)
        got = _BOUNDS_CACHE[dtype] = (int(info.min), int(info.max))
    return got


def _applicable(args: tuple, kwargs: dict) -> bool:
    if not 2 <= len(args) <= 3:
        return False
    if set(kwargs) - {"side", "sorter"}:
        return False
    if kwargs.get("sorter") is not None:
        return False
    side = args[2] if len(args) == 3 else kwargs.get("side", "left")
    if side not in ("left", "right"):
        return False
    if len(args) == 3 and "side" in kwargs:
        return False
    a, v = args[0], args[1]
    if type(a) is not np.ndarray or a.ndim != 1 or a.dtype.kind not in "iu":
        return False
    if isinstance(v, bool) or not isinstance(v, int):
        return False
    lo, hi = _bounds(a.dtype)
    return v < lo or v > hi


def _run(a, v, side="left"):
    lo, hi = _bounds(a.dtype)
    return np.intp(a.size if v > hi else 0)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="searchsorted_extreme_key",
            op="numpy.searchsorted",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000039",
                "source": "https://github.com/numpy/numpy/issues/29719",
                "license": "order-theoretic identity; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
