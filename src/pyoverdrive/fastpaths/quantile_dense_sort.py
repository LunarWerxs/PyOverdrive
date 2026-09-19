"""Fast path: numpy.quantile with an ARRAY of quantiles, via sort + exact lerp.

Provenance (OPP-000022): numpy/numpy#32187 - stock quantile's
partition-based route can become costly when many order statistics are
requested. Sorting each slice once and interpolating all requested values
avoids repeated selection work. Quantile-count and reduced-length windows
bound the regime; scalar quantiles and unsupported layouts stay on stock.

Correctness contract:
- Applies only to quantile(a, q[, axis]) where a is a plain float64
  ndarray (1-D, or 2-D reduced along its LAST axis: axis -1 or 1;
  for 1-D, axis absent, None, 0 or -1 all mean the same reduction), q is
  a plain 1-D float64 ndarray with 4 <= q.size <= 16384, every q in
  [0, 1] (stock raises outside; the scan is tiny), the reduced length is
  in [512, 65536] (the measured range; outside it stays on stock), and
  the only other argument is method='linear' (absent or explicit).
  out/overwrite_input/keepdims/weights/interpolation, other dtypes,
  other axes, scalar q, and 3-D+ inputs all stay on stock.
- The route sorts along the reduced axis once and then performs numpy's
  own virtual-index + lerp arithmetic (including the above-bounds -1
  substitution and the gamma >= 0.5 stability rewrite), which the
  battery checked BIT-EXACT against stock everywhere, NaN-salted slices
  included (after a sort, a slice's NaN is last; the route stamps those
  slices NaN exactly as stock's arithmetic yields).

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=quantile_dense_sort or
pyoverdrive.disable_path("quantile_dense_sort").

Historical calibration ratios are omitted because the NumPy version was not
recorded. See docs/research/2026-09-19-burndown.md for current measured
evidence and its version, hardware and load qualifications.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

_F64 = np.dtype(np.float64)

# Bounds on requested quantiles and reduced length limit the sort route.
# Slice count is not independently gated: both routes repeat work per slice,
# but shape/count neighbors still need public-API measurements on each host.
NQ_MIN, NQ_MAX = 4, 16_384
M_MIN, M_MAX = 512, 65_536


def _normalize(args: tuple, kwargs: dict):
    if not 2 <= len(args) <= 3:
        return None
    a, q = args[0], args[1]
    axis = args[2] if len(args) == 3 else kwargs.get("axis", None)
    allowed = {"axis", "method"}
    if set(kwargs) - allowed:
        return None
    if len(args) == 3 and "axis" in kwargs:
        return None  # duplicate axis: stock raises TypeError
    if kwargs.get("method", "linear") != "linear":
        return None
    return a, q, axis


def _applicable(args: tuple, kwargs: dict) -> bool:
    norm = _normalize(args, kwargs)
    if norm is None:
        return False
    a, q, axis = norm
    if type(a) is not np.ndarray or a.dtype != _F64:
        return False
    if type(q) is not np.ndarray or q.dtype != _F64 or q.ndim != 1:
        return False
    if not NQ_MIN <= q.size <= NQ_MAX:
        return False
    if a.ndim == 1:
        if axis not in (None, 0, -1):
            return False
    elif a.ndim == 2:
        if axis not in (1, -1):
            return False
    else:
        return False
    m = a.shape[-1]
    if not M_MIN <= m <= M_MAX:
        return False
    if not bool(((q >= 0.0) & (q <= 1.0)).all()):
        return False
    return True


def _run(a, q, axis=None, method="linear"):
    moved = a if a.ndim == 1 else np.moveaxis(a, -1, 0)
    n = moved.shape[0]
    # stock_fn, not np.sort: that name is patched too (sort_char_view)
    srt = GEARBOX.stock_fn("numpy.sort")(moved, axis=0)
    virt = (n - 1) * q
    prev_i = np.floor(virt).astype(np.intp)
    next_i = prev_i + 1
    above = virt >= n - 1
    prev_i[above] = -1
    next_i[above] = -1
    gamma = (virt - prev_i).reshape(virt.shape + (1,) * (srt.ndim - 1))
    below_v = srt[prev_i]
    above_v = srt[next_i]
    diff = above_v - below_v
    result = np.where(gamma >= 0.5, above_v - diff * (1 - gamma), below_v + diff * gamma)
    # after the sort, any slice containing NaN has NaN last
    has_nan = np.isnan(srt[-1, ...])
    if a.ndim == 1:
        if has_nan:
            result[...] = np.nan
    elif has_nan.any():
        result[..., has_nan] = np.nan
    return result


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="quantile_dense_sort",
            op="numpy.quantile",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000022",
                "source": "https://github.com/numpy/numpy/issues/32187",
                "license": "sort + numpy's own interpolation arithmetic replicated; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
