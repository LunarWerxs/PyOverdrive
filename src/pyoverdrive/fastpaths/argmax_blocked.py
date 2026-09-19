"""Fast path (CALIBRATION-GATED, off by default): numpy.argmax(a, axis=0)
on large C-order 2-D arrays via a cache-blocked transpose then a
fast-axis argmax.

Provenance (OPP-000034): numpy/numpy#9182 - argmax has no strided
kernel, so reducing along the non-contiguous axis of a C-order array
makes a hidden contiguous copy through a cache-hostile transpose walk
(juliantaylor's root cause). Tiling the transpose into 128x128 blocks
keeps source and destination tiles cache-resident. The benefit depends on
CPU cache behavior and stock's strided-copy implementation, so this path
ships disabled. ``pyoverdrive.calibrate()`` enables it only when the
local probe clears its required margin at the regime edges. Calibration
is stored per machine fingerprint; stale hardware/stack records are
ignored. See src/pyoverdrive/calibration.py.

Correctness contract (unchanged from the Intel-validated build):
- Applies only to argmax(a, axis) where a is a plain C-contiguous 2-D
  float64/float32/int64 ndarray, axis is 0 or -2, no out/keepdims,
  rows >= ROWS_MIN and a.size >= SIZE_MIN (Intel-measured cell edges;
  calibration may store tighter floors per machine). Other axes
  (axis=1 is already the fast axis - measured parity), other dtypes,
  F-order, 1-D/3-D+, and kwargs stay on stock.
- The blocked transpose is an exact permutation and the final argmax
  scans each column's elements in the same increasing order stock does,
  so first-occurrence ties and first-NaN semantics are preserved
  exactly; output is bit-identical (intp indices). NaN-salted battery
  cells checked exact.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=argmax_blocked_transpose or
pyoverdrive.disable_path("argmax_blocked_transpose"); note the path is
already off unless calibration enabled it.

Implementation note: the in-run argmax goes through stock_fn, never the
patched numpy.argmax name.

Historical calibration ratios are omitted because the NumPy version was not
recorded. See docs/research/2026-09-19-burndown.md for current measured
evidence and its version, hardware and load qualifications.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

_DTYPES = frozenset(np.dtype(t) for t in (np.float64, np.float32, np.int64))
# Intel-measured defaults; calibration.apply() may overwrite per machine.
ROWS_MIN = 3_000
SIZE_MIN = 9_000_000
_BLOCK = 128


def _applicable(args: tuple, kwargs: dict) -> bool:
    if not 1 <= len(args) <= 2:
        return False
    if set(kwargs) - {"axis"}:
        return False
    if len(args) == 2 and "axis" in kwargs:
        return False  # duplicate: stock raises TypeError
    axis = args[1] if len(args) == 2 else kwargs.get("axis", None)
    if axis not in (0, -2) or isinstance(axis, bool):
        return False
    a = args[0]
    if type(a) is not np.ndarray or a.dtype not in _DTYPES:
        return False
    if a.ndim != 2 or not a.flags.c_contiguous:
        return False
    return a.shape[0] >= ROWS_MIN and a.size >= SIZE_MIN


def _blocked_argmax_axis0(a, argmax_fn):
    """The route itself, argmax injected so the calibration probe can time
    it against plain stock without touching the dispatch layer."""
    rows, cols = a.shape
    t = np.empty((cols, rows), dtype=a.dtype)
    for j0 in range(0, cols, _BLOCK):
        j1 = min(j0 + _BLOCK, cols)
        for i0 in range(0, rows, _BLOCK):
            i1 = min(i0 + _BLOCK, rows)
            t[j0:j1, i0:i1] = a[i0:i1, j0:j1].T
    return argmax_fn(t, axis=1)


def _run(a, axis=0):
    return _blocked_argmax_axis0(a, GEARBOX.stock_fn("numpy.argmax"))


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="argmax_blocked_transpose",
            op="numpy.argmax",
            applicable=_applicable,
            run=_run,
            enabled=False,  # calibration-gated: see module docstring
            provenance={
                "opportunity": "OPP-000034",
                "source": "https://github.com/numpy/numpy/issues/9182",
                "license": "blocked-transpose technique from first principles; no third-party code",
                "comparison_mode": "bit-identical",
                "calibration_gated": True,
            },
        )
    )
