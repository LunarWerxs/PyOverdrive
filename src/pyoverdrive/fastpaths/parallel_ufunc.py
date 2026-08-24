"""Fast path family: PyRallel chunked dispatch for unary elementwise ufuncs.

Provenance (OPP-000008): numpy/numpy#8208 (2016, closed same day as out of
scope for NumPy itself) reported ~3x for np.sin over 4 threads on 1e7
float64 elements by splitting the array into contiguous chunks and calling
np.sin(chunk, out=out_chunk) on a ThreadPoolExecutor, which works because the
compute-bound ufunc loops release the GIL. Reproduced and exceeded on the
first calibrated machine: 5.7x at 1e6 (8 threads), 7.2x at 1e7 (16 threads).
Mechanism reimplemented from the issue's description; no upstream code.

Correctness contract:
- Applies to ``np.<op>(x)`` and ``np.<op>(x, out=o)`` only: exactly one
  positional plain-ndarray argument, and at most the ``out`` keyword, whose
  value must be a plain C-contiguous ndarray of identical shape and dtype
  (``o is x``, the in-place idiom from the source issue, is allowed: each
  chunk touches only its own index range). Every other keyword (where=,
  dtype=, casting=, order=, subok=, signature=) and any out= needing a cast,
  broadcast, or non-contiguous write stays on stock untouched.
- C-contiguous input only (chunks are flat views; anything else falls back).
- The fresh-allocation form pays first-touch page faults inside the worker
  threads, measured 25-40% slower than the out= form at 1e6 float64 on the
  calibration machine; the thresholds below are calibrated on the
  allocating form (the conservative one), so out= calls only do better.
- Only (op, dtype) pairs with committed Dyno evidence of a win, and only at
  sizes at or above the measured crossover for that pair. Sizes below the
  crossover run on stock: that IS the "automatic single-thread selection"
  the Phase 4 gate asks for, and it is table-driven, not guessed.
- The caller's np.errstate is mirrored into every chunk (it is thread-local
  in NumPy); callers running under raise/call/log modes stay on stock so no
  FloatingPointError or callback ever fires from a worker thread.
- Result is bit-identical to stock (elementwise kernel, no cross-element
  data flow; the differential suite asserts it, the calibration battery
  checks it on every measured case).

Comparison mode: bit-identical (spec section 9).

Kill switches: PYOVERDRIVE_DISABLE=pyrallel_sin (per op),
pyoverdrive.disable_path("pyrallel_sin"), or PYOVERDRIVE_THREADS=1 (whole
PyRallel core off: every predicate here goes False).
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath
from ..parallel import pyrallel
from . import _pyrallel_common as _common

_F64 = np.dtype(np.float64)
_F32 = np.dtype(np.float32)

# ---------------------------------------------------------------------------
# CALIBRATION TABLE. Every entry below is a measured crossover, fingerprint
# 8f8198d9abab (Zen 4 AVX-512, 16C/32T, numpy 2.4.5), evidence in
# benchmarks/results/PYRALLEL-CAL/ and benchmarks/results/OPP-000008/.
# An (op, dtype) pair absent from this table has either no evidence or
# measured no win, and stays on stock. Regenerate with
# benchmarks/micro/bench_pyrallel_calibration.py, then edit by hand with the
# numbers in front of you; never extrapolate a row.
# ---------------------------------------------------------------------------

# op name -> {dtype: minimum element count at which the path dispatches}.
# Derived by lab/cli/calibrate_pyrallel.py (min win 1.3x at the scheduled
# thread count, at that size and every larger measured size) from the
# PYRALLEL-CAL battery of 2026-08-23 rerun on a QUIET box (5-11% foreign
# load, the first uncontended calibration; log:
# benchmarks/results/_scratch/baseline-20260823-131020.log). The quiet run
# confirmed every contended threshold except two float64 rows: log dropped
# 300k -> 100k (1.41x there) and log10 rose 100k -> 300k (only 1.16x at
# 100k once the baseline ran unloaded; the contended run had understated
# stock, not the candidate).
#
# Measured at the threshold / at 1e7 (16 threads), float64:
#   sin 1.63x / 4.63x   cos 1.33x / 5.26x   tan 1.70x / 4.92x
#   exp 1.42x / 4.60x   log 1.41x / 6.07x   log10 1.92x / 5.97x
#   tanh 1.46x / 6.90x  sqrt 1.60x / 2.26x (memory bound, late crossover)
# float32 kernels are ~2-3x faster per element, so the crossover sits 3-10x
# higher in elements; sqrt float32 only pays from 3M elements (1.77x).
SUPPORTED: dict[str, dict[np.dtype, int]] = {
    "sin": {_F64: 100_000, _F32: 300_000},
    "cos": {_F64: 100_000, _F32: 1_000_000},
    "tan": {_F64: 100_000, _F32: 300_000},
    "exp": {_F64: 300_000, _F32: 1_000_000},
    "log": {_F64: 100_000, _F32: 1_000_000},
    "log10": {_F64: 300_000, _F32: 300_000},
    "tanh": {_F64: 100_000, _F32: 1_000_000},
    "sqrt": {_F64: 1_000_000, _F32: 3_000_000},
}

# THREAD_SCHEDULE / threads_for live in _pyrallel_common (shared with the
# binary family); re-exported here because this module is the documented
# home of the unary calibration.
THREAD_SCHEDULE = _common.THREAD_SCHEDULE
threads_for = _common.threads_for


def _make_applicable(table: dict[np.dtype, int]):
    floor = min(table.values())

    def applicable(args: tuple, kwargs: dict) -> bool:
        # Cheapest, most selective checks first (small arrays dominate refusals).
        if len(args) != 1:
            return False
        x = args[0]
        if type(x) is not np.ndarray or x.size < floor:
            return False
        if kwargs:
            if len(kwargs) != 1 or "out" not in kwargs:
                return False
            if not _common.out_ok(kwargs["out"], x.shape, x.dtype):
                return False
        threshold = table.get(x.dtype)
        return (
            threshold is not None
            and x.size >= threshold
            and x.flags.c_contiguous
            and _common.core_ready()  # last: costs ~1 us
        )

    return applicable


def _make_run(gearbox, op: str):
    def run(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        stock = gearbox.stock_fn(op)  # the real ufunc, never the patched name
        return pyrallel.parallel_unary(stock, x, threads_for(x.nbytes), out=out)

    return run


_PROVENANCE = {
    "opportunity": "OPP-000008",
    "source": "https://github.com/numpy/numpy/issues/8208",
    "license": "mechanism reimplemented from the issue text; no third-party code",
    "comparison_mode": "bit-identical",
}


def register(gearbox) -> None:
    for op_name, table in SUPPORTED.items():
        op = f"numpy.{op_name}"
        gearbox.register(
            FastPath(
                name=f"pyrallel_{op_name}",
                op=op,
                applicable=_make_applicable(table),
                run=_make_run(gearbox, op),
                provenance=dict(_PROVENANCE, op=op),
            )
        )
