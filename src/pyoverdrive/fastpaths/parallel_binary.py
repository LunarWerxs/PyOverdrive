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
# Derived by `lab/cli/calibrate_pyrallel.py --suite PYRALLEL-BIN-CAL` from
# benchmarks/results/PYRALLEL-BIN-CAL/ (fingerprint 8f8198d9abab). A pair
# absent here measured no >= 1.3x win at any size and stays on stock.
# Bandwidth-bound wins are the most machine dependent numbers in the
# project: recalibrate on every new box before trusting this table there.
# ---------------------------------------------------------------------------
# 2026-08-23 battery rerun on a QUIET box (5-7% foreign load, the first
# uncontended calibration; log:
# benchmarks/results/_scratch/baseline-20260823-131020.log): wins at the
# threshold are 1.4-1.6x and top out at 1.6-2.1x at 1e7 on 16 threads. The
# quiet run moved six rows DOWN from the contended table (add/multiply f32
# 10M -> 3M; add/multiply/divide f64 3M -> 1M; maximum i64 3M -> 1M): the
# contended run had slowed stock as much as the candidate at mid sizes.
# Below ~1e6 a binary op finishes in tens of microseconds and threading
# LOSES by 4-10x (0.10-0.24x at 1e5), which is why every threshold here is
# >= 1e6.
SUPPORTED: dict[str, dict[np.dtype, int]] = {
    "add": {_F64: 1_000_000, _F32: 3_000_000, _I64: 1_000_000},
    "subtract": {_F64: 1_000_000, _F32: 3_000_000, _I64: 1_000_000},
    "multiply": {_F64: 1_000_000, _F32: 3_000_000, _I64: 1_000_000},
    "divide": {_F64: 1_000_000, _F32: 3_000_000},
    "maximum": {_F64: 1_000_000, _F32: 3_000_000, _I64: 1_000_000},
    "minimum": {_F64: 1_000_000, _F32: 3_000_000, _I64: 1_000_000},
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
