"""Fast path: numpy.ascontiguousarray of a transposed 2-D array, blocked + threaded.

Provenance (OPP-000004): numpy/numpy#21655 reports stock's order-conversion
copy benefiting from cache-blocked, threaded tiling (the reporter's
``fast_relayout_array2d``). The shipped mechanism uses the persistent
PyRallel pool rather than creating an executor per call. Calibration
inputs remain in benchmarks/micro/bench_relayout_calibration.py and
benchmarks/results/RELAYOUT-CAL/.

Correctness contract:
- Applies only to ``np.ascontiguousarray(x)`` with no keywords, where ``x``
  is a plain 2-D ndarray that is F-contiguous and NOT C-contiguous (the
  ``a.T`` of a C array, or a genuinely Fortran-ordered matrix) with a
  supported dtype. Everything else (already C-contiguous input, which stock
  returns as-is without copying; dtype= or like= keywords; other ndims;
  subclasses) stays on stock.
- Returns a fresh C-contiguous copy, bit-identical to stock's (every block
  is a stock slice assignment; no arithmetic happens here).

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=relayout_blocked or
pyoverdrive.disable_path("relayout_blocked").

Historical calibration ratios are omitted because the NumPy version was not
recorded. See docs/research/2026-09-19-burndown.md for current measured
evidence and its version, hardware and load qualifications.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath
from ..parallel import pyrallel
from ..parallel.relayout import blocked_transpose_copy

# Tiling balances locality against the overhead of many slice assignments.
# Keep the chosen block size separate from the overall array-size gate.
TILE = 256
# The array-size floor leaves only the upper thread-count regime of the
# original schedule reachable; pool configuration still caps this request.
THREADS = 16
# Power-of-two leading dimensions can amplify stock's cache-conflict costs.
# A floor inferred only from those shapes would admit neighboring layouts
# without comparable evidence. The conservative shared floor incorporates
# non-power-of-two shapes; keep one-element and aspect-ratio neighbors in
# tools/probe_relayout_floor.py and the public-API sweep.
# dtype -> minimum element count (n * m) at which the path dispatches.
_FLOOR_ELEMENTS = 4 * 1024 * 1024
SUPPORTED: dict[np.dtype, int] = {
    np.dtype(np.float64): _FLOOR_ELEMENTS,
    np.dtype(np.float32): _FLOOR_ELEMENTS,
    np.dtype(np.int64): _FLOOR_ELEMENTS,
}


_FLOOR = min(SUPPORTED.values())


def _applicable(args: tuple, kwargs: dict) -> bool:
    # np.ascontiguousarray on an already-C array is a no-copy stock operation
    # and a very common library idiom, so the refusal path is ordered by cost:
    # size first (most calls are small), one flags object, then the dtype table.
    if len(args) != 1 or kwargs:
        return False
    x = args[0]
    if type(x) is not np.ndarray or x.size < _FLOOR or x.ndim != 2:
        return False
    flags = x.flags
    if flags.c_contiguous or not flags.f_contiguous:
        return False
    threshold = SUPPORTED.get(x.dtype)
    return threshold is not None and x.size >= threshold and pyrallel.available()


def _run(x: np.ndarray) -> np.ndarray:
    return blocked_transpose_copy(x, tile=TILE, threads=THREADS)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="relayout_blocked",
            op="numpy.ascontiguousarray",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000004",
                "source": "https://github.com/numpy/numpy/issues/21655",
                "license": "tiling idea from the issue text, reimplemented; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
