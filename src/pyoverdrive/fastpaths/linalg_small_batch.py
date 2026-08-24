"""Fast paths: numpy.linalg.det / slogdet / solve on 2x2/3x3 batches via
closed forms (cofactor expansion; Cramer's rule).

Provenance (OPP-000045): stock runs one LAPACK call per matrix, so
stacked small matrices pay per-call overhead thousands of times over -
the same vein as the SHIPPED inv_small_batch (OPP-000035, numpy#17166)
and eigvalsh_2x2_closed (OPP-000030). numpy/numpy#20052 documents
slogdet's own overhead. Closed forms vectorize across the whole stack.

CALIBRATION (fp 9bbe7063c555, idle box, 0% load, numpy 2.5.2,
benchmarks/results/BATCH6-CAL/): det 3x3 2.06x at batch 100 to 20.6x
at 5000; det 2x2 6.2x at 100 to 91.9x at 20000; slogdet 3x3 1.58x at
100 to 11.3x; slogdet 2x2 3.0x to 13.7x; solve 3x3 0.81x at 100
(loses), 1.76x at 300, 3.47x at 1000, 6.22x at 5000; solve 2x2 1.14x
at 30, 2.29x at 100, 4.65x at 300. The floors below sit ONE MEASURED
NOTCH ABOVE the 1.3x crossings because the predicate re-derives the
determinant for its conditioning guard (a cost the battery's
candidate-only timing does not carry); the end-to-end MVP rows verify
the dispatched win.

Conditioning guard: identical policy to inv_small_batch, DET_RTOL=1e-8
against |det| / scale^d (that module's battery measured the pass at
cond 1e6 and the fail at 1e8; Cramer shares the adjugate's cancellation
behavior, and det/slogdet themselves lose relative accuracy exactly
where |det| collapses). Exactly-singular input is refused by the same
test: for solve, stock raises LinAlgError; for det, stock returns 0.0
exactly, which a rounded closed form will not; for slogdet the SIGN
would be unstable. Non-finite input refused by an isfinite scan.

Correctness contract:
- det/slogdet: plain float64 ndarray, shape (..., d, d), d in {2, 3},
  ndim >= 3, batch >= floor, finite, well-conditioned per the guard;
  no kwargs. slogdet returns stock's result type (probed at import)
  with signs EXACTLY equal to stock's on admitted input.
- solve: additionally b must be a plain float64 ndarray of shape
  batch + (d, 1) (the measured column form); everything else refuses.
- Different algorithm, different rounding: numeric mode. det/solve
  battery-checked at rtol 1e-9/1e-8; slogdet signs exact, logdet
  rtol 1e-9.

Comparison mode: numeric (spec section 9). Kill switches:
det_small_batch, slogdet_small_batch, solve_small_batch.
"""

from __future__ import annotations

import math

import numpy as np

from ..dispatcher.gearbox import FastPath
from .inv_small_batch import DET_RTOL, _det_and_scale

_F64 = np.dtype(np.float64)

# (op-kind, d) -> minimum batch: one measured notch above the 1.3x edge
_FLOORS = {
    ("det", 2): 100,
    ("det", 3): 300,
    ("slogdet", 2): 100,
    ("slogdet", 3): 300,
    ("solve", 2): 300,
    ("solve", 3): 1_000,
}

# stock's return type for slogdet (a namedtuple in modern numpy)
_SLOGDET_RESULT = type(np.linalg.slogdet(np.eye(2)))


def _stack_ok(a, kind: str) -> bool:
    if type(a) is not np.ndarray or a.dtype != _F64 or a.ndim < 3:
        return False
    d = a.shape[-1]
    if a.shape[-2] != d:
        return False
    floor = _FLOORS.get((kind, d))
    if floor is None or math.prod(a.shape[:-2]) < floor:
        return False
    if not bool(np.isfinite(a).all()):
        return False
    det, scale = _det_and_scale(a)
    return bool((np.abs(det) >= DET_RTOL * np.maximum(scale, 1e-100) ** d).all())


def _applicable_det(args: tuple, kwargs: dict) -> bool:
    return len(args) == 1 and not kwargs and _stack_ok(args[0], "det")


def _applicable_slogdet(args: tuple, kwargs: dict) -> bool:
    return len(args) == 1 and not kwargs and _stack_ok(args[0], "slogdet")


def _applicable_solve(args: tuple, kwargs: dict) -> bool:
    if len(args) != 2 or kwargs:
        return False
    a, b = args
    if not _stack_ok(a, "solve"):
        return False
    if type(b) is not np.ndarray or b.dtype != _F64:
        return False
    return b.shape == a.shape[:-2] + (a.shape[-1], 1)


def _run_det(a):
    det, _ = _det_and_scale(a)
    return det


def _run_slogdet(a):
    det, _ = _det_and_scale(a)
    return _SLOGDET_RESULT(np.sign(det), np.log(np.abs(det)))


def _run_solve(a, b):
    d = a.shape[-1]
    b1 = b[..., 0, 0]
    b2 = b[..., 1, 0]
    out = np.empty_like(b)
    if d == 2:
        a11 = a[..., 0, 0]; a12 = a[..., 0, 1]
        a21 = a[..., 1, 0]; a22 = a[..., 1, 1]
        inv_det = 1.0 / (a11 * a22 - a12 * a21)
        out[..., 0, 0] = (b1 * a22 - b2 * a12) * inv_det
        out[..., 1, 0] = (a11 * b2 - a21 * b1) * inv_det
        return out
    a11 = a[..., 0, 0]; a12 = a[..., 0, 1]; a13 = a[..., 0, 2]
    a21 = a[..., 1, 0]; a22 = a[..., 1, 1]; a23 = a[..., 1, 2]
    a31 = a[..., 2, 0]; a32 = a[..., 2, 1]; a33 = a[..., 2, 2]
    b3 = b[..., 2, 0]
    c11 = a22 * a33 - a23 * a32
    c12 = a23 * a31 - a21 * a33
    c13 = a21 * a32 - a22 * a31
    inv_det = 1.0 / (a11 * c11 + a12 * c12 + a13 * c13)
    out[..., 0, 0] = (
        b1 * c11 + b2 * (a13 * a32 - a12 * a33) + b3 * (a12 * a23 - a13 * a22)
    ) * inv_det
    out[..., 1, 0] = (
        b1 * c12 + b2 * (a11 * a33 - a13 * a31) + b3 * (a13 * a21 - a11 * a23)
    ) * inv_det
    out[..., 2, 0] = (
        b1 * c13 + b2 * (a12 * a31 - a11 * a32) + b3 * (a11 * a22 - a12 * a21)
    ) * inv_det
    return out


def register(gearbox) -> None:
    common = {
        "opportunity": "OPP-000045",
        "source": "https://github.com/numpy/numpy/issues/20052",
        "license": "cofactor expansion and Cramer's rule, textbook formulas; no third-party code",
        "comparison_mode": "numeric",
    }
    gearbox.register(
        FastPath(
            name="det_small_batch",
            op="numpy.linalg.det",
            applicable=_applicable_det,
            run=_run_det,
            provenance=dict(common),
        )
    )
    gearbox.register(
        FastPath(
            name="slogdet_small_batch",
            op="numpy.linalg.slogdet",
            applicable=_applicable_slogdet,
            run=_run_slogdet,
            provenance=dict(common),
        )
    )
    gearbox.register(
        FastPath(
            name="solve_small_batch",
            op="numpy.linalg.solve",
            applicable=_applicable_solve,
            run=_run_solve,
            provenance=dict(common),
        )
    )
