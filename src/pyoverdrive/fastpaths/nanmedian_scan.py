"""Fast path: numpy.nanmedian 2-D via one isnan scan + vectorized median.

Provenance (OPP-000043): nanmedian with an axis loops per-slice in
Python; numpy/numpy#4683 documents the mechanism class for np.ma.median
(same per-slice looping around median). For NaN-free 2-D input, one
O(n) scan plus plain np.median(axis=...) (vectorized partition) returns
the bit-identical result.

CALIBRATION (fp 9bbe7063c555, idle box, 0% load, numpy 2.5.2,
benchmarks/results/BATCH6-CAL/): (500,500) 2.59x, (2000,200) 2.42x,
(100,2000) 1.42x, (1000,1000) 1.96x - and the anti-regime (200,10000)
0.94x: few LONG slices amortize stock's Python loop while the
vectorized partition pays for the whole width. Wasted-scan (1% NaN)
0.99-1.01x. The guards below are that measured envelope: reduced
length <= 2000, total size >= 200000 (the smallest winning cell), and
axis=1/-1 on C-contiguous input only (every measured cell reduced the
contiguous axis; layout flips regimes, the nanquantile lesson).

Correctness contract:
- plain 2-D C-contiguous float64 ndarray; axis 1 or -1 as the only
  argument beyond the array (out/keepdims/overwrite_input force stock).
- clean input: bit-identical to stock.
- NaN present (all-NaN slices included, with stock's RuntimeWarning):
  stock's own result, via internal fallback.

Comparison mode: bit-identical. Kill switch: nanmedian_scan.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

_F64 = np.dtype(np.float64)
SIZE_FLOOR = 200_000
REDUCED_LEN_CAP = 2_000


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) == 1:
        if set(kwargs) != {"axis"}:
            return False
        axis = kwargs["axis"]
    elif len(args) == 2:
        if kwargs:
            return False
        axis = args[1]
    else:
        return False
    a = args[0]
    if type(a) is not np.ndarray or a.dtype != _F64 or a.ndim != 2:
        return False
    if isinstance(axis, bool) or axis not in (1, -1):
        return False
    if not a.flags.c_contiguous:
        return False
    if a.size < SIZE_FLOOR or a.shape[1] > REDUCED_LEN_CAP:
        return False
    return a.shape[1] > 0


def _run(a, axis=None):
    if np.isnan(np.min(a)):
        # NaN somewhere: stock's masking/warnings are the semantics
        return GEARBOX.stock_fn("numpy.nanmedian")(a, axis=axis)
    return GEARBOX.stock_fn("numpy.median")(a, axis=axis)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="nanmedian_scan",
            op="numpy.nanmedian",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000043",
                "source": "https://github.com/numpy/numpy/issues/4683",
                "license": "isnan scan + vectorized median, standard technique; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
