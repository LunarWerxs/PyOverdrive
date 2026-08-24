"""Fast path: numpy.nan_to_num default-args float64 via copy + in-place
masking.

Provenance (OPP-000046): numpy/numpy#23140 - stock nan_to_num builds
its result through a chain of np.where allocations. A single copy
followed by in-place np.copyto masks (NaN -> 0.0, +inf -> float64 max,
-inf -> float64 min: exactly stock's documented defaults) produces the
identical array with fewer temporaries; the isinf pass is skipped
entirely when no infinity exists (an .any() probe, part of the
measured cost).

CALIBRATION (fp 9bbe7063c555, idle box, 0% load, numpy 2.5.2,
benchmarks/results/BATCH6-CAL/): 2.21x at n=1e4 (1% NaN), 1.87x at
n=1e6 (1% NaN), 2.02x clean, 1.20x with a NaN+inf mix - a win in
every measured cell, floor at the smallest measured n.

Correctness contract:
- nan_to_num(x) plus the scalar-override forms nan_to_num(x, nan=v,
  posinf=v, neginf=v): plain float64 ndarray, size >= SIZE_FLOOR;
  override values must be Python int/float/bool (or np.float64) - the
  fills stock assigns by plain scalar store, replicated bit-for-bit by
  copyto (probe: 2.03x at n=1e6 with overrides). Array-like fills,
  np.float32 fills, an explicit copy= (either value), other dtypes,
  complex input: all unmeasured or different semantics, all refuse to
  stock.
- bit-identical to stock, always a fresh copy like stock's copy=True.
  All three replacement masks are computed from the ORIGINAL values
  (stock computes idx_nan/idx_posinf/idx_neginf before any store), so
  a non-finite nan= override cannot leak into the inf replacements.

Comparison mode: bit-identical. Kill switch: nan_to_num_where.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath

_F64 = np.dtype(np.float64)
SIZE_FLOOR = 10_000  # smallest measured winning cell (2.21x)


def _fill_ok(v) -> bool:
    # Python int/float only (np.float64 subclasses float; np.float32 and
    # array-likes refuse: stock broadcasts those, an unmeasured regime).
    return type(v) in (int, float, bool) or isinstance(v, float)


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 1:
        return False
    if kwargs:
        if set(kwargs) - {"nan", "posinf", "neginf"}:
            return False  # copy= or anything unknown: stock
        if "nan" in kwargs and not _fill_ok(kwargs["nan"]):
            return False
        for k in ("posinf", "neginf"):
            if k in kwargs and kwargs[k] is not None and not _fill_ok(kwargs[k]):
                return False
    a = args[0]
    return type(a) is np.ndarray and a.dtype == _F64 and a.size >= SIZE_FLOOR


def _run(a, nan=0.0, posinf=None, neginf=None):
    # All three masks read the ORIGINAL values, exactly like stock: with
    # a non-finite nan= override, a former-NaN slot must NOT be caught by
    # the inf replacement afterwards.
    out = a.copy()
    np.copyto(out, nan, where=np.isnan(a))
    isinf = np.isinf(a)
    if isinf.any():
        info = np.finfo(out.dtype)
        np.copyto(out, info.max if posinf is None else posinf, where=isinf & (a > 0))
        np.copyto(out, info.min if neginf is None else neginf, where=isinf & (a < 0))
    return out


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="nan_to_num_where",
            op="numpy.nan_to_num",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000046",
                "source": "https://github.com/numpy/numpy/issues/23140",
                "license": "copy + copyto masking, trivial standard technique; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
