"""Fast path: numpy.ascontiguousarray of a transposed 2-D array, blocked + threaded.

Provenance (OPP-000004): numpy/numpy#21655 reports stock's order-conversion
copy of large 2-D arrays being 6.6x slower than a cache-blocked, threaded
pure-NumPy tiling (the reporter's ``fast_relayout_array2d``). Dyno
reproduced the mechanism at 1.1-2.1x with the issue's per-call executor
(benchmarks/results/OPP-000004/); the shipped mechanism
(pyoverdrive.parallel.relayout) runs the same tiling on the persistent
PyRallel pool and is calibrated by benchmarks/micro/bench_relayout_calibration.py
(benchmarks/results/RELAYOUT-CAL/).

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
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath
from ..parallel import pyrallel
from ..parallel.relayout import blocked_transpose_copy

# CALIBRATION (fp 8f8198d9abab, benchmarks/results/RELAYOUT-CAL/, 2026-08-23,
# taken at 83-99% foreign CPU load so every number below is understated):
# 21 cases x 9 (tile, threads) variants, bit-identical on all of them.
# - 256x256 LOSES at every setting (0.75-0.90x best): below the floor.
# - tile 256 is best or within noise of best from 512x512 up, for all three
#   dtypes; 64 is consistently worst (too many tiny slice assignments).
# - 8 threads wins at 512x512-1024x1024, 16 from 2048x2048 up.
# float64: 512x512 2.0x, 1024x1024 2.9x, 2048x2048 3.8x, 4096x4096 3.6-3.8x,
#   8192x1024 2.6x, 1024x8192 2.9-3.1x.
# float32: 512x512 1.7-1.9x, 1024x1024 4.4x, 2048x2048 4.2x, 4096x4096 5.6x,
#   8192x1024 6.6x, 1024x8192 4.9x.
# int64: 512x512 only 1.0-1.3x (excluded), 1024x1024 2.0x, 2048x2048 2.9x,
#   4096x4096 3.3x, 8192x1024 2.5x, 1024x8192 3.1x.
TILE = 256
_THREADS_LARGE = (4 * 1024 * 1024, 16)  # >= 4M elements (2048x2048): 16 threads
_THREADS_DEFAULT = 8
# dtype -> minimum element count (n * m) at which the path dispatches
SUPPORTED: dict[np.dtype, int] = {
    np.dtype(np.float64): 512 * 512,
    np.dtype(np.float32): 512 * 512,
    np.dtype(np.int64): 1024 * 1024,
}


def threads_for(size: int) -> int:
    return _THREADS_LARGE[1] if size >= _THREADS_LARGE[0] else _THREADS_DEFAULT


_FLOOR = min(SUPPORTED.values())


def _applicable(args: tuple, kwargs: dict) -> bool:
    # np.ascontiguousarray on an already-C array is a ~70 ns no-op in stock
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
    return blocked_transpose_copy(x, tile=TILE, threads=threads_for(x.size))


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
