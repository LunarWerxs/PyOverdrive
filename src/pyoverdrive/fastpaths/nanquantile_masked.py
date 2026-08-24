"""Fast path: numpy.nanquantile axis-reduction without the per-slice Python loop.

Provenance (OPP-000013): numpy/numpy#16575 reports np.nanquantile(a, q,
axis=...) tens to hundreds of times slower than np.quantile on the same
call, because nanquantile loops slice-by-slice in Python (maintainer
seberg: "these are functions written in Python... overheads associated to
each individual slice"). Dyno reproduced 53-70x at the reporter's own
(27, 100) shape and 102x at (50, 100, 100), with zero losing cases
anywhere measured (benchmarks/results/OPP-000013/). Two routes, both from
the record:

- zero NaNs anywhere: one O(n) isnan scan, then numpy's OWN np.quantile,
  which is already vectorized across slices. The scan cost is part of the
  measured win.
- NaNs present: sort along the reduced axis (float NaNs sort to the end),
  per-slice valid counts from the isnan mask, then the same virtual-index
  plus linear-interpolation arithmetic numpy's quantile performs,
  vectorized over all slices at once with take_along_axis.

Correctness contract:
- Applies only to nanquantile(a, q[, axis]) where a is a plain 2-D+
  float64 ndarray, q is a single Python int/float in [0, 1] (no q
  sequences), axis is a single int, and NO other argument is passed (out,
  keepdims, overwrite_input, method, weights, interpolation all force
  stock, as do float32/other dtypes and axis=None: every one of those is
  either semantics this route does not replicate or a regime without its
  own measurement).
- The interpolation replicates numpy's _lerp exactly, including its
  gamma >= 0.5 rewrite (b - diff * (1 - gamma)) that exists for numerical
  stability, so per-element results match stock's to the bit in the cases
  the differential battery pins; the declared comparison mode stays
  numeric (ULP-scale tolerance) rather than bit-identical because stock
  reaches the same arithmetic through a per-slice compress-and-partition
  path this module deliberately does not follow.
- All-NaN slices produce NaN AND the same "All-NaN slice encountered"
  RuntimeWarning stock emits.

Comparison mode: numeric (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=nanquantile_masked or
pyoverdrive.disable_path("nanquantile_masked").
"""

from __future__ import annotations

import warnings

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

# CALIBRATION (fp 8f8198d9abab, benchmarks/results/FASTNANQ-CAL/,
# 2026-08-23, 10-11% foreign load). The vectorized route wins EVERY
# many-slice case measured, down to tiny inputs: 22.6x at (10, 30), 229x
# at (5, 500), 51.9-63.4x at the reporter's (27, 100), 43.5-75.1x at
# (50, 100, 100), 5.5-10.9x at (500, 500). It LOSES only in the few-long-
# slices anti-regime the issue thread predicted (cakedev0): (10000, 3)
# 0.88x with NaNs, (100000, 5) 0.85-0.93x, where stock's Python loop runs
# 3-5 iterations and the candidate sorts the whole array. The guard below
# admits a call when the reduced length is modest (every measured
# reduced_len <= 500 wins by >= 5.4x) OR the slice count is at least the
# reduced length (cakedev0's own crossover model); both loss shapes fail
# both arms. (3, 10000) reduced along its contiguous axis measured a real
# 1.40-1.50x win but is refused by this rule: layout flips that regime's
# sign, marginally, and no simple measured rule separates it.
SIZE_FLOOR = 300  # elements; smallest measured winning input (10, 30)
_REDUCED_LEN_CAP = 1_000  # above this, require n_slices >= reduced_len

_F64 = np.dtype(np.float64)


def _normalize(args: tuple, kwargs: dict):
    """Return (a, q, axis) for an admissible call shape, else None."""
    if not 2 <= len(args) <= 3:
        return None
    a, q = args[0], args[1]
    if len(args) == 3:
        if kwargs:
            return None
        axis = args[2]
    else:
        if len(kwargs) != 1 or "axis" not in kwargs:
            return None
        axis = kwargs["axis"]
    return a, q, axis


def _applicable(args: tuple, kwargs: dict) -> bool:
    norm = _normalize(args, kwargs)
    if norm is None:
        return False
    a, q, axis = norm
    if type(a) is not np.ndarray or a.dtype != _F64:
        return False
    if a.ndim < 2 or a.size < SIZE_FLOOR:
        return False
    if isinstance(axis, bool) or not isinstance(axis, int):
        return False
    if not -a.ndim <= axis < a.ndim:
        return False
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        return False
    if not 0.0 <= q <= 1.0:
        return False
    reduced = a.shape[axis]
    return reduced <= _REDUCED_LEN_CAP or a.size >= reduced * reduced


def _run(a: np.ndarray, q, axis=None) -> np.ndarray:
    am = np.moveaxis(a, axis, 0)
    n = am.shape[0]
    nan_mask = np.isnan(am)
    if not nan_mask.any():
        # numpy's quantile is already vectorized across slices; with no
        # NaNs present it is nanquantile's exact answer (stock_fn, not
        # np.quantile: that name is patched too, by quantile_dense_sort)
        return GEARBOX.stock_fn("numpy.quantile")(a, q, axis=axis)
    valid = n - nan_mask.sum(axis=0)
    if (valid == 0).any():
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
    # stock_fn, not np.sort: that name is patched too (sort_char_view)
    s = GEARBOX.stock_fn("numpy.sort")(am, axis=0)  # float NaNs sort to the end
    vc = np.maximum(valid, 1)
    virt = q * (vc - 1.0)
    lo = np.floor(virt).astype(np.intp)
    np.clip(lo, 0, vc - 1, out=lo)
    hi = np.minimum(lo + 1, vc - 1)
    gamma = virt - lo
    a_lo = np.take_along_axis(s, lo[None, ...], axis=0)[0]
    a_hi = np.take_along_axis(s, hi[None, ...], axis=0)[0]
    diff = a_hi - a_lo
    res = a_lo + diff * gamma
    # numpy _lerp's stability rewrite: approach the upper point from above
    # once gamma passes one half, exactly as stock computes it
    np.subtract(a_hi, diff * (1.0 - gamma), out=res, where=gamma >= 0.5)
    return np.where(valid == 0, np.nan, res)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="nanquantile_masked",
            op="numpy.nanquantile",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000013",
                "source": "https://github.com/numpy/numpy/issues/16575",
                "license": "vectorized masked route from the issue thread's candidates, reimplemented; no third-party code",
                "comparison_mode": "numeric (float64, ULP-scale: same interpolation arithmetic as stock)",
            },
        )
    )
