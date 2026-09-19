"""Fast path: numpy.isclose without its per-call wrapper overhead.

Provenance (OPP-000012): numpy/numpy#16160 reports np.isclose paying a
fixed per-call cost (asanyarray conversions, an errstate context manager,
two full isfinite reductions) that can dominate arithmetic on small
inputs. Fusing the finite-input expression removes that wrapper work;
size caps limit the route as array arithmetic becomes the main cost.

Correctness contract:
- Applies only to isclose(a, b[, rtol, atol]) where rtol/atol (positional
  or keyword) are FINITE Python ints/floats and equal_nan is absent or
  False, and the operands are either two Python int/float scalars or two
  plain same-shape same-dtype float64/float32 ndarrays below the size cap
  CONTAINING ONLY FINITE VALUES - the predicate scans isfinite and
  refuses otherwise. Non-finite inputs need stock's masked handling,
  and non-finite tolerances need its error-state suppression
  (WarrenWeckesser's observation in-thread). equal_nan=True also stays
  on stock. The refusal still pays for the finite-value scan, so it
  belongs in fallback-overhead measurements.
- With those guarantees the dispatched computation is the pure fused
  expression abs(a - b) <= atol + rtol * abs(b), which for all-finite
  operands and finite tolerances is exactly stock's within_tol
  arithmetic: bit-identical output (a bool array, or np.bool_ for scalar
  input, matching stock's return type).

The dtype-specific size caps bound an overhead-saving regime; removing
fixed wrapper work cannot imply a benefit for arbitrarily large arrays.
Scalar handling preserves stock's result type.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=isclose_fused or
pyoverdrive.disable_path("isclose_fused").

Historical calibration ratios are omitted because the NumPy version was not
recorded. See docs/research/2026-09-19-burndown.md for current measured
evidence and its version, hardware and load qualifications.
"""

from __future__ import annotations

import math

import numpy as np

from ..dispatcher.gearbox import FastPath

_CAPS = {
    np.dtype(np.float64): 1_000,
    np.dtype(np.float32): 10_000,
}
_SCALARS = (int, float)


def _tolerances(args: tuple, kwargs: dict):
    """Return (rtol, atol) if the call shape is admissible, else None."""
    if not 2 <= len(args) <= 4:
        return None
    rtol, atol = 1e-05, 1e-08
    if len(args) >= 3:
        rtol = args[2]
    if len(args) == 4:
        atol = args[3]
    extra = set(kwargs) - {"rtol", "atol", "equal_nan"}
    if extra:
        return None
    if kwargs.get("equal_nan", False):
        return None
    if "rtol" in kwargs:
        if len(args) >= 3:
            return None  # duplicate: stock raises TypeError
        rtol = kwargs["rtol"]
    if "atol" in kwargs:
        if len(args) == 4:
            return None
        atol = kwargs["atol"]
    for tol in (rtol, atol):
        if isinstance(tol, bool) or not isinstance(tol, _SCALARS):
            return None
        if not math.isfinite(tol):
            return None
    return rtol, atol


def _applicable(args: tuple, kwargs: dict) -> bool:
    if _tolerances(args, kwargs) is None:
        return False
    a, b = args[0], args[1]
    if isinstance(a, _SCALARS) and isinstance(b, _SCALARS):
        return (
            not isinstance(a, bool)
            and not isinstance(b, bool)
            and math.isfinite(a)
            and math.isfinite(b)
        )
    if type(a) is not np.ndarray or type(b) is not np.ndarray:
        return False
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    cap = _CAPS.get(a.dtype)
    if cap is None or a.size > cap:
        return False
    # the finiteness scan makes the run branch-free; on a refusal it costs
    # ~a microsecond before stock's own (identical) reduction
    return bool(np.isfinite(a).all()) and bool(np.isfinite(b).all())


def _run(a, b, *args, **kwargs):
    rtol, atol = _tolerances((a, b) + args, kwargs)
    x = np.asarray(a, dtype=np.float64) if not isinstance(a, np.ndarray) else a
    y = np.asarray(b, dtype=np.float64) if not isinstance(b, np.ndarray) else b
    return (np.abs(x - y) <= atol + rtol * np.abs(y))[()]


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="isclose_fused",
            op="numpy.isclose",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000012",
                "source": "https://github.com/numpy/numpy/issues/16160",
                "license": "fused expression from the issue thread, reimplemented; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
