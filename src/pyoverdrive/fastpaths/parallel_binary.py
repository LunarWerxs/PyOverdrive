"""Fast path family: PyRallel chunked dispatch for BINARY elementwise ufuncs.

Provenance (OPP-000008, numpy/numpy#8208): the issue reporter predicted
that simple arithmetic like np.add would NOT scale across threads (memory
bandwidth bound). Chunking can use multiple cores to supply memory
bandwidth, but the outcome depends on the host memory system and thread
scheduling. The family therefore has its own dtype/size table and battery
(benchmarks/micro/bench_pyrallel_binary_calibration.py); it does not inherit
transcendental-ufunc crossovers.

Correctness contract:
- Applies to ``np.<op>(a, b)`` and ``np.<op>(a, b, out=o)`` only: two plain
  C-contiguous ndarrays of IDENTICAL shape and of one supported dtype
  (NO broadcasting, NO mixed dtypes, NO scalars: all of those stay on
  stock, whose broadcasting and casting rules are not reimplemented here).
- ``out`` must be a plain, writeable, C-contiguous ndarray of the same
  shape and of the dtype stock would produce (for every op in this table
  that is the operand dtype). ``out`` aliasing an operand
  (``np.add(x, y, out=x)``) is fine only for exact elementwise aliases:
  shifted overlapping views stay on stock for safe input buffering.
- Float64 add with an exact in-place output stays on stock after repeated
  measured losses; disjoint float64 outputs and integer add retain dispatch.
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

Historical calibration ratios are omitted because the NumPy version was not
recorded. See docs/research/2026-09-19-burndown.md for current measured
evidence and its version, hardware and load qualifications.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath, StockRaised
from ..parallel import pyrallel
from . import _pyrallel_common as _common

_F64 = np.dtype(np.float64)
_F32 = np.dtype(np.float32)
_I64 = np.dtype(np.int64)

# Operation/dtype -> minimum element count. Missing pairs stay on stock.
# Thread submission and output traffic make small binary calls unsuitable
# for this route. Floors must be measured through public dispatch, with
# stock on a fast core of a hybrid CPU; a slow-core baseline flatters the
# threaded candidate. Repeated independent sweeps guard against noise near
# a crossover. Float32 and divide rows were withdrawn, and only selected
# float64/integer rows remain; do not extrapolate across dtype or operation.
# Historical inputs: benchmarks/results/PYRALLEL-DISPATCH-CAL/.
# Regeneration tool: tools/calibrate_dispatch.py --family binary.
# Host calibration may remove rows, never lower these shipped floors.
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


def _make_applicable(table: dict[np.dtype, int], *, op_name: str | None = None):
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
            if not _common.out_ok(kwargs["out"], a.shape, a.dtype, args):
                return False
            # Only disjoint buffers or exact elementwise aliases survive
            # out_ok. The latter lose for float64 add on the measured Intel
            # host; this operation-specific gate leaves the core unchanged.
            if op_name == "add" and a.dtype == _F64 and (
                np.shares_memory(kwargs["out"], a) or np.shares_memory(kwargs["out"], b)
            ):
                return False
        threshold = table.get(a.dtype)
        return (
            threshold is not None
            and a.size >= threshold
            and a.flags.c_contiguous
            and b.flags.c_contiguous
            and _common.core_ready()  # last: thread and error-state checks
        )

    return applicable


def _make_run(gearbox, op: str):
    def run(a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        stock = gearbox.stock_fn(op)  # the real ufunc, never the patched name
        try:
            return pyrallel.parallel_elementwise(stock, (a, b), _common.threads_for(a.nbytes), out=out)
        except RuntimeWarning as exc:
            # Preserve stock's warning-as-error without replaying mutated inputs.
            raise StockRaised(exc) from exc

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
                applicable=_make_applicable(table, op_name=op_name),
                run=_make_run(gearbox, op),
                provenance=dict(_PROVENANCE, op=op),
            )
        )
