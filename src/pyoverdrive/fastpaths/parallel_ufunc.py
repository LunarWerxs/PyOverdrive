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
# 9bbe7063c555 (i7-12700K, 8P+4E, numpy 2.5.2), evidence in
# benchmarks/results/PYRALLEL-DISPATCH-CAL/.
# An (op, dtype) pair absent from this table has either no evidence or
# measured no win, and stays on stock. Regenerate with
# tools/calibrate_dispatch.py, then edit by hand with the numbers in front
# of you; never extrapolate a row.
# ---------------------------------------------------------------------------

# op name -> {dtype: minimum element count at which the path dispatches}.
#
# RE-DERIVED 2026-08-24 by tools/calibrate_dispatch.py, and the table it
# replaced was wrong in 15 of its 16 rows. The old one came from
# benchmarks/micro/bench_pyrallel_calibration.py, which times the CANDIDATE
# (parallel_unary against the bare ufunc). Two things about that were fine
# and one was fatal:
#
#   - candidate vs dispatched: measured, and they agree; the predicate here
#     is cheap enough not to show up.
#   - blocked vs interleaved timing: measured, and they agree too.
#   - THE BASELINE WAS A COIN FLIP. This is a hybrid CPU (8 P-cores + 4
#     E-cores). A single-threaded process is placed on one class and stays
#     there, so asked for the same np.sin float64 n=1e5 in 25 fresh
#     processes the box answered 344 us fifteen times and 497 us ten times,
#     1.44x apart, with almost no spread inside either group. A THREADED
#     candidate spans cores and averages over the split, so the flip moves
#     the denominator of every ratio and never the numerator - up to 1.44x,
#     always in our favour. The old evidence for sin float64 1e5 (stock
#     490.8 us, pyrallel_4t 307.8 us, 1.59x) is simply a run that drew an
#     E-core: the candidate is unchanged today at ~302 us and stock is
#     344 us, making the same cell 1.10x.
#     Full write-up: docs/research/hybrid-cpu-baseline-coin-flip.md.
#
# The numbers below are measured END TO END through the patched public name,
# one cell per process, both sides interleaved, only on processes that drew
# a fast core, and each threshold is the smallest size clearing 1.3x on the
# WORST of {sorted, shuffled} x {bare result, consumed result} at that size
# and every larger measured size. Sorted input is measured because it is not
# a corner case for these ops - np.sin(np.linspace(...)) is close to the
# canonical call - and because NumPy's own trig kernels run ~2.2x FASTER on
# sorted data (sin at n=1e5: 338 us sorted, 753 us shuffled), which leaves
# threading much less to win back there.
#
# Measured worst-case at the threshold / at 1e7, float64:
#   sin 1.36x / 1.94x   cos 1.35x / 1.90x   tan 1.37x / 1.90x
#   exp 1.41x / 2.04x   log 1.34x / 1.82x   log10 1.41x / 1.91x
#   tanh 1.63x / 2.16x
#
# sqrt is GONE, not merely raised: it clears 1.3x at no measured size on
# either dtype (best 1.19x at 1e7 float64, 1.17x at 1e7 float32), and at its
# old shipped floor of 1e6 float64 it ran at 1.05x. It is memory-bandwidth
# bound, so threads cannot help it; it stays on stock.
#
# Three float32 rows (cos, exp, log) do clear 1.3x, but only at 1e7, the
# LARGEST size measured - one point, with nothing above it to show the win
# persists. House rule is that hardware decides and nobody extrapolates, so
# they stay on stock until the sweep is extended past 1e7. All three ran at
# 1.06-1.15x at their old thresholds, so nothing of value is lost.
SUPPORTED: dict[str, dict[np.dtype, int]] = {
    "sin": {_F64: 300_000, _F32: 3_000_000},
    "cos": {_F64: 300_000},
    "tan": {_F64: 1_000_000, _F32: 1_000_000},
    "exp": {_F64: 300_000},
    "log": {_F64: 1_000_000},
    "log10": {_F64: 1_000_000, _F32: 1_000_000},
    "tanh": {_F64: 3_000_000, _F32: 3_000_000},
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
