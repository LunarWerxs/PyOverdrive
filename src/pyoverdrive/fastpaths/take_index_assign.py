"""Fast path: numpy.take with out= via fancy-index assignment.

Provenance (OPP-000051): numpy/numpy#28636 (open) - np.take runs
measurably SLOWER when out= is provided than without it. seberg's
in-thread diagnosis: "take ensures that the output is unchanged when
an out-of-bounds access happens", i.e. stock pays for a buffered
write. The equivalent out[...] = a[indices] runs the plain fancy-index
gather plus one copy, wins at every measured size, and KEEPS that
guarantee: the gather materializes (and raises on any bad index)
before out is touched. numpy's own triage (mattip, 2025-08-20) wants
the same avoid-the-copy idea upstream.

CALIBRATION (fp 9bbe7063c555, idle box, 0-1% load, numpy 2.5.2,
benchmarks/results/BATCH8-CAL/): 3.34x at 1000 gathered elements,
2.17x at 3000, 2.15x at 10_000, 1.82x at 100_000 (1.80x int64), 1.58x
at 1M, 1.30x at 10M - a win at every measured cell; floor 1000, the
smallest measured cell (below it the ~300 ns dispatch tax approaches
the microsecond-scale call itself).

Correctness contract:
- Applies only to take(a, indices, out=<ndarray>): exactly two
  positional arguments plus exactly the out kwarg. a is a plain 1-D
  float64/int64 ndarray; indices a plain 1-D intp ndarray of size >=
  SIZE_MIN; out a plain 1-D ndarray with out.dtype == a.dtype and
  out.shape == indices.shape. axis=, mode= and every other form stay
  on stock.
- Negative indices wrap exactly as stock's default mode='raise' does.
- Out-of-bounds indices: the gather raises before out is touched; the
  path then reruns STOCK so the caller sees stock's own IndexError
  (StockRaised pattern), with out left exactly as stock leaves it.
- Bit-identical, and returns the same `out` object stock returns.

Comparison mode: bit-identical (spec section 9). Kill switch:
take_index_assign.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath, StockRaised

_DTYPES = frozenset((np.dtype(np.float64), np.dtype(np.int64)))
_INTP = np.dtype(np.intp)
SIZE_MIN = 1_000  # smallest measured winning cell (3.34x)


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 2 or set(kwargs) != {"out"}:
        return False
    a, indices = args
    if type(a) is not np.ndarray or a.ndim != 1 or a.dtype not in _DTYPES:
        return False
    if type(indices) is not np.ndarray or indices.ndim != 1 or indices.dtype != _INTP:
        return False
    if indices.size < SIZE_MIN:
        return False
    out = kwargs["out"]
    return (
        type(out) is np.ndarray
        and out.dtype == a.dtype
        and out.shape == indices.shape
    )


def _run(a, indices, out):
    try:
        gathered = a[indices]
    except IndexError:
        # out-of-bounds: let STOCK produce its own error (and any side
        # effects) so the caller sees exactly stock behavior
        stock = GEARBOX.stock_fn("numpy.take")
        try:
            return stock(a, indices, out=out)
        except Exception as exc:  # noqa: BLE001 - stock's raise is the contract
            raise StockRaised(exc) from None
    out[...] = gathered
    return out


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="take_index_assign",
            op="numpy.take",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000051",
                "source": "https://github.com/numpy/numpy/issues/28636",
                "license": "fancy-index gather plus assignment, trivial standard technique",
                "comparison_mode": "bit-identical",
            },
        )
    )
