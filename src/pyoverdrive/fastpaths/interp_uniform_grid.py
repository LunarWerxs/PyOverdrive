"""Fast path: numpy.interp on a UNIFORMLY SPACED sample grid via direct
index arithmetic instead of per-query binary search.

Provenance (OPP-000050): numpy/numpy#2513 (Trac #1920) asked for a
better-than-binary-search interp back in 2012; upstream kept the
general-grid searchsorted route. When xp is uniform by construction
(linspace/arange grids - the resampling regime), the bin of every query
is one subtract-multiply away, and the whole lookup vectorizes: no
per-element bisection at all.

CALIBRATION (fp 9bbe7063c555, idle box, 0-1% load, numpy 2.5.2,
benchmarks/results/BATCH8-CAL/, guard scans included in the candidate):
3.37x at nq=3000/grid 1000, 5.95x at 10_000/1000, 3.32x at 30_000,
2.81x at 100_000, 3.19x at 1M/10_000, 1.46x at the 1M/100 arange-grid
cell; the earlier probe's nq=1000 cell measured 1.06x, under the bar -
floor 3000, the smallest measured winning cell.

Guards, all part of the measured candidate cost:
- xp must be uniform: every np.diff(xp) within 1e-9 relative of the
  first, which is loose enough to admit linspace/arange grids (their
  diffs wobble in the last ulp) and strict enough that the arithmetic
  bin can disagree with bisection only at a bin boundary, where the
  lerp is continuous and the two answers differ by rounding only. The
  same check enforces dx > 0, i.e. strictly increasing xp.
- fp must be all-finite (the lerp form lo + frac*(hi-lo) differs from
  stock's arithmetic around inf/nan), and x must be all-finite (a NaN
  query would hit undefined intp-cast behavior).

Correctness contract:
- Applies only to interp(x, xp, fp) with exactly three positional
  arguments and no kwargs (left/right/period: the user chose different
  edge semantics, stock handles them); all three plain 1-D float64
  ndarrays; xp.size >= 2, fp.size == xp.size, x.size >= NQ_MIN.
- Below xp[0] the result is fp[0], at/above xp[-1] it is fp[-1],
  exactly stock's default edge policy.
- Different arithmetic, different rounding: numeric mode (measured
  1e-14..1e-12 relative on the probe grid).

Comparison mode: numeric (spec section 9). Kill switch:
interp_uniform_grid.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath

_F64 = np.dtype(np.float64)
NQ_MIN = 3_000  # smallest measured winning cell (3.37x); 1000 measured 1.06x
UNIFORM_RTOL = 1e-9


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 3 or kwargs:
        return False
    x, xp, fp = args
    for a in (x, xp, fp):
        if type(a) is not np.ndarray or a.ndim != 1 or a.dtype != _F64:
            return False
    if xp.size < 2 or fp.size != xp.size or x.size < NQ_MIN:
        return False
    d = np.diff(xp)
    dx = d[0]
    if not dx > 0:
        return False
    if not bool((np.abs(d - dx) <= UNIFORM_RTOL * dx).all()):
        return False
    if not bool(np.isfinite(fp).all()):
        return False
    return bool(np.isfinite(x).all())


def _run(x, xp, fp):
    dx = (xp[-1] - xp[0]) / (xp.size - 1)
    pos = (x - xp[0]) / dx
    idx = pos.astype(np.intp)
    np.clip(idx, 0, fp.size - 2, out=idx)
    frac = pos - idx
    lo = fp[idx]
    out = lo + frac * (fp[idx + 1] - lo)
    out[x <= xp[0]] = fp[0]
    out[x >= xp[-1]] = fp[-1]
    return out


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="interp_uniform_grid",
            op="numpy.interp",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000050",
                "source": "https://github.com/numpy/numpy/issues/2513",
                "license": "direct-index linear interpolation from first principles",
                "comparison_mode": "numeric",
            },
        )
    )
