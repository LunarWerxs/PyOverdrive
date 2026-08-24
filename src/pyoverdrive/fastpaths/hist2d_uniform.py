"""Fast path: numpy.histogram2d with uniform-by-construction bins via
direct index computation.

Provenance (OPP-000038): numpy/numpy#17676 - histogramdd builds edge
arrays and runs a per-dimension searchsorted over every sample even
when bins are uniform by construction (int bin counts plus an explicit
range); numpy's own 1-D histogram takes the direct-index shortcut
internally, which is the reporter's precedent (4-5x in their use case).

The route computes floor bin indices, then CORRECTS them against the
actual rounded linspace edges (one vectorized compare down, one up) -
the same trick numpy's 1-D path uses - so values exactly ON an edge
land exactly where stock puts them. The battery's decisive edge-salted
cell (half the samples exactly on interior/outer edges) was
BIT-IDENTICAL at 1.74x. Rightmost edge inclusive; out-of-range samples
dropped; weights ride through bincount.

Measured (OPP-000038 + BATCH5-CAL batteries, fp 9bbe7063c555, idle box,
0-1% load): 1.65x at the reporter's 5e6-sample 100x100 case, 2.41x at
1000x1000 bins, 1.74x edge-salted, 1.73x weighted, 1.67x at 50x50,
1.42x at 30x30; 10x10 measured 1.09x (below min-win), hence the
total-bins floor of 900. histogramdd 3-D measured a wash (1.03x) and
is NOT shipped.

Correctness contract:
- Applies only to histogram2d(x, y, bins=..., range=...) where x and y
  are plain 1-D float64 ndarrays of equal length, bins is an int or a
  pair of ints each >= 2 with product >= 900, range is a pair of
  finite (lo, hi) pairs with lo < hi, weights is absent or a plain 1-D
  float64 ndarray of the same length, and density/normed are absent.
  Everything else - array-valued bins (explicit edges), missing range,
  other dtypes, density - stays on stock.
- Bit-identical: counts (float64, exactly as stock returns), edge
  arrays (the same np.linspace), and weighted sums.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=hist2d_uniform or
pyoverdrive.disable_path("hist2d_uniform").
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath

_F64 = np.dtype(np.float64)
BINS_MIN_EACH = 2
BINS_MIN_TOTAL = 900  # 30x30 measured 1.42x; 10x10 measured 1.09x


def _norm_bins(bins):
    if isinstance(bins, bool):
        return None
    if isinstance(bins, (int, np.integer)):
        return int(bins), int(bins)
    if isinstance(bins, (list, tuple)) and len(bins) == 2:
        out = []
        for b in bins:
            if isinstance(b, bool) or not isinstance(b, (int, np.integer)):
                return None
            out.append(int(b))
        return tuple(out)
    return None


def _norm_range(range_):
    if not isinstance(range_, (list, tuple)) or len(range_) != 2:
        return None
    out = []
    for pair in range_:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        try:
            lo, hi = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            return None
        out.append((lo, hi))
    return tuple(out)


def _applicable(args: tuple, kwargs: dict) -> bool:
    if not 2 <= len(args) <= 4:
        return False
    if set(kwargs) - {"bins", "range", "weights", "density"}:
        return False
    if kwargs.get("density") is not None:
        return False
    bins = args[2] if len(args) >= 3 else kwargs.get("bins", 10)
    if len(args) >= 3 and "bins" in kwargs:
        return False
    range_ = args[3] if len(args) == 4 else kwargs.get("range", None)
    if len(args) == 4 and "range" in kwargs:
        return False
    nb = _norm_bins(bins)
    if nb is None or min(nb) < BINS_MIN_EACH or nb[0] * nb[1] < BINS_MIN_TOTAL:
        return False
    if _norm_range(range_) is None:
        return False
    x, y = args[0], args[1]
    for a in (x, y):
        if type(a) is not np.ndarray or a.ndim != 1 or a.dtype != _F64:
            return False
    if x.size != y.size:
        return False
    w = kwargs.get("weights", None)
    if w is not None:
        if type(w) is not np.ndarray or w.ndim != 1 or w.dtype != _F64:
            return False
        if w.size != x.size:
            return False
    return True


def _indices(x, lo, hi, nbins, edges):
    idx = np.floor((x - lo) * (nbins / (hi - lo))).astype(np.intp)
    np.clip(idx, 0, nbins - 1, out=idx)
    idx[x < edges[idx]] -= 1
    idx2 = idx + 1
    np.clip(idx2, 0, nbins, out=idx2)
    idx[x >= edges[idx2]] += 1
    np.clip(idx, 0, nbins - 1, out=idx)
    keep = (x >= edges[0]) & (x <= edges[-1])
    return idx, keep


def _run(x, y, bins=10, range=None, weights=None, density=None):
    nx, ny = _norm_bins(bins)
    (xlo, xhi), (ylo, yhi) = _norm_range(range)
    ex = np.linspace(xlo, xhi, nx + 1)
    ey = np.linspace(ylo, yhi, ny + 1)
    ix, keepx = _indices(x, xlo, xhi, nx, ex)
    iy, keepy = _indices(y, ylo, yhi, ny, ey)
    keep = keepx & keepy
    flat = ix[keep] * ny + iy[keep]
    w = None if weights is None else weights[keep]
    h = np.bincount(flat, weights=w, minlength=nx * ny).reshape(nx, ny)
    # unconditional: unweighted bincount returns int64, and so does a
    # bincount whose weights array is EMPTY (probed) - stock is float64
    # in every case
    h = h.astype(np.float64, copy=False)
    return h, ex, ey


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="hist2d_uniform",
            op="numpy.histogram2d",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000038",
                "source": "https://github.com/numpy/numpy/issues/17676",
                "license": "direct-index binning reimplemented from first principles",
                "comparison_mode": "bit-identical",
            },
        )
    )
