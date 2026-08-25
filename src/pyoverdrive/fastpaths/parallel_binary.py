"""Fast path family: PyRallel chunked dispatch for BINARY elementwise ufuncs.

Provenance (OPP-000008, numpy/numpy#8208): the issue reporter predicted
that simple arithmetic like np.add would NOT scale across threads (memory
bandwidth bound). The Dyno reproducer measured otherwise on the first
calibrated machine: np.add at 1e7 float64 gains 2.4-2.75x on 4-16 threads,
because one core cannot saturate dual-channel DDR5 by itself. That is a
bandwidth effect, far more machine dependent than the compute-bound
transcendental family (parallel_ufunc.py), so it lives in its own module
with its own calibration table and its own battery
(benchmarks/micro/bench_pyrallel_binary_calibration.py).

Correctness contract:
- Applies to ``np.<op>(a, b)`` and ``np.<op>(a, b, out=o)`` only: two plain
  C-contiguous ndarrays of IDENTICAL shape and of one supported dtype
  (NO broadcasting, NO mixed dtypes, NO scalars: all of those stay on
  stock, whose broadcasting and casting rules are not reimplemented here).
- ``out`` must be a plain, writeable, C-contiguous ndarray of the same
  shape and of the dtype stock would produce (for every op in this table
  that is the operand dtype). ``out`` aliasing an operand
  (``np.add(x, y, out=x)``) is fine: chunks own disjoint index ranges.
- The caller's np.errstate is mirrored into every chunk; raise/call/log
  modes stay on stock (see parallel_ufunc.py).
- Result is bit-identical to stock (elementwise kernel, no cross-element
  data flow; the differential suite asserts it, the battery checks it).

Reach: only explicit ``np.add(a, b)`` style calls go through the patched
module-level name. The operator form ``a + b`` resolves to ``ndarray.__add__``
in C and never touches ``numpy.add``, so it is NOT accelerated by this family
(nor is NumPy's own internal use of the ufunc). Making operators adaptive
needs the extension-hook route the spec (10.5) reserves for later.

Comparison mode: bit-identical (spec section 9).

Kill switches: PYOVERDRIVE_DISABLE=pyrallel_add (per op),
pyoverdrive.disable_path("pyrallel_add"), or PYOVERDRIVE_THREADS=1.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath
from ..parallel import pyrallel
from . import _pyrallel_common as _common

_F64 = np.dtype(np.float64)
_F32 = np.dtype(np.float32)
_I64 = np.dtype(np.int64)

# ---------------------------------------------------------------------------
# CALIBRATION TABLE: op name -> {operand dtype: minimum element count}.
# Derived by `tools/calibrate_dispatch.py --family binary` (fingerprint
# 9bbe7063c555), evidence in benchmarks/results/PYRALLEL-DISPATCH-CAL/.
# A pair absent here measured no >= 1.3x win at any size and stays on stock.
# Bandwidth-bound wins are the most machine dependent numbers in the
# project: recalibrate on every new box before trusting this table there.
# ---------------------------------------------------------------------------
# RE-DERIVED 2026-08-24 by tools/calibrate_dispatch.py --family binary, and
# almost nothing of the old table survived. The previous rows came from
# benchmarks/micro/bench_pyrallel_binary_calibration.py, whose
# single-threaded baseline is a per-process coin flip on this hybrid CPU
# (P-core or E-core, 1.44x apart, always inflating a threaded candidate);
# the "quiet rerun" that moved six rows DOWN to 1e6 was reading that flip.
# Measured end to end at those 1e6 floors the family actually delivered
# 1.04-1.20x against a promised 1.3x, and subtract float32 at its 3e6 floor
# ran at 0.97x - a dispatched loss.
# Full write-up: docs/research/hybrid-cpu-baseline-coin-flip.md.
#
# THESE FLOORS COME FROM TWO INDEPENDENT SWEEPS WITH THE WORSE READING KEPT
# PER CELL, which for this family is not pedantry. Bandwidth-bound wins here
# cross the 1.3x bar somewhere between 1e7 and 3e7 elements and the
# run-to-run spread is about as wide as the margin: one sweep read
# subtract float64 at 1.23x, 1.14x, 1.33x on consecutive sizes - non
# monotone. A threshold read off a single sweep is fitting noise, so a row
# ships only if it cleared 1.3x TWICE at its floor and at every larger size.
#
# What that leaves, and what it removed:
#   - EVERY float32 row is gone. Best any of them managed on the worse
#     reading is 1.28x (add, multiply at 2e7); float32 halves the bytes per
#     element, so a given element count carries half the bandwidth pressure.
#   - divide is gone entirely: float64 peaks at 1.32x but dips to 1.29x at
#     2e7, so no size clears the bar and stays clear above it.
#   - multiply float64 misses by a hair (1.31x at 2e7, 1.2996x at 3e7).
#     Kept out rather than rounded in.
# Below ~1e6 a binary op finishes in tens of microseconds and threading
# LOSES by 4-10x (0.50-0.92x at 3e5 here), which is what the floors prevent.
#
# This family is the strongest candidate in the project for per-machine
# calibration (src/pyoverdrive/calibration.py) rather than a shipped table:
# the numbers are bandwidth, and bandwidth is the least transferable thing
# measured here. Doing that needs the probe cells run in re-drawn
# subprocesses first - see the warning in calibration.py.
SUPPORTED: dict[str, dict[np.dtype, int]] = {
    "add": {_F64: 20_000_000, _I64: 20_000_000},
    "subtract": {_I64: 20_000_000},
    "multiply": {_I64: 10_000_000},
    "maximum": {_F64: 20_000_000, _I64: 20_000_000},
    "minimum": {_F64: 20_000_000, _I64: 20_000_000},
}


# The table AS SHIPPED, captured before anything can edit SUPPORTED.
# `pyoverdrive --calibrate` may drop rows that do not pay on the host
# machine, and it rewrites SUPPORTED in place to do it. Without a
# pristine copy that would be one-way: a row dropped by one calibration
# could never be re-probed, because the probe would no longer find it.
SHIPPED: dict[str, dict[np.dtype, int]] = {
    op: dict(row) for op, row in SUPPORTED.items()
}


def _make_applicable(table: dict[np.dtype, int]):
    floor = min(table.values())

    def applicable(args: tuple, kwargs: dict) -> bool:
        # Cheapest, most selective checks first: almost every refused call is
        # a small array, and it must leave here in a few hundred ns.
        if len(args) != 2:
            return False
        a, b = args
        if type(a) is not np.ndarray or a.size < floor or type(b) is not np.ndarray:
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if kwargs:
            if len(kwargs) != 1 or "out" not in kwargs:
                return False
            if not _common.out_ok(kwargs["out"], a.shape, a.dtype):
                return False
        threshold = table.get(a.dtype)
        return (
            threshold is not None
            and a.size >= threshold
            and a.flags.c_contiguous
            and b.flags.c_contiguous
            and _common.core_ready()  # last: costs ~1 us
        )

    return applicable


def _make_run(gearbox, op: str):
    def run(a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        stock = gearbox.stock_fn(op)  # the real ufunc, never the patched name
        return pyrallel.parallel_elementwise(stock, (a, b), _common.threads_for(a.nbytes), out=out)

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
