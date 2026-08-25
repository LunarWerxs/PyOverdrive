"""Fast path: numpy.linalg.inv on batches of 2x2/3x3 matrices via the
vectorized adjugate.

Provenance (OPP-000035): numpy/numpy#17166 - stock inv runs one LAPACK
call per matrix, so stacked small matrices pay per-call overhead
thousands of times; the reporter's hand-vectorized adjugate was 213x
faster on a 4000-stack (88x independently confirmed in-thread), and
thrasibule named the mechanism: reorder the loops so every arithmetic
op vectorizes across the batch. Same per-matrix-LAPACK-overhead vein as
the shipped eigvalsh_2x2_closed (OPP-000030, 31x).

Measured (OPP-000035 + BATCH5-CAL batteries, fp 9bbe7063c555, idle box,
0-1% load): 3x3 float64 2.95x at batch 300, 5.24x at 1000, 8.20x at
4000, 3.27x at 10_000, 1.77x at 100_000; 2x2 float64 12.1x at 1000,
4.5x at 100_000; 3x3 float32 16.5x at 10_000. Batch 1 loses 0.16x and
100 straddles 1.39x, hence the floors.

The CONDITION CEILING, measured not guessed: LAPACK's pivoted LU stays
accurate where the adjugate's naive cancellation does not. The battery's
scaled rtol-1e-9 check PASSED at condition 1e3 and 1e6 and FAILED at
1e8 (and at 1e10). The predicate therefore refuses any matrix whose
|det| falls below DET_RTOL times scale^d (scale = that matrix's max
|entry|): the passing cond-1e6 batch sits at |det|/scale^3 ~ 3e-8, the
failing cond-1e8 at ~3e-10, and the threshold 1e-8 lies between them,
conservative toward refusal for the unmeasured middle. Exactly-singular
input (stock raises LinAlgError) is refused by the same test, and
non-finite input is refused by an isfinite scan (stock raises or
propagates per LAPACK; the adjugate would silently emit garbage).

Correctness contract:
- Applies only to inv(a) where a is a plain ndarray, shape (..., d, d)
  with d in {2, 3}, ndim >= 3, dtype/batch in the measured table
  (float64: batch >= 300 for 3x3, >= 1000 for 2x2; float32: 3x3 only,
  batch >= 10_000), every element finite, and every |det| above the
  scale-relative floor. Everything else - single matrices, other
  dtypes/sizes, near-singular batches - stays on stock.
- Different algorithm, different rounding: numeric mode, battery-checked
  at rtol 1e-9 (f64) / 1e-3 (f32) scaled per matrix.

Comparison mode: numeric (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=inv_small_batch or
pyoverdrive.disable_path("inv_small_batch").
"""

from __future__ import annotations

import math

import numpy as np

from ..dispatcher.gearbox import FastPath

# (d, dtype) -> minimum batch (measured winning cells only)
_FLOORS = {
    (2, np.dtype(np.float64)): 1_000,
    (3, np.dtype(np.float64)): 300,
    (3, np.dtype(np.float32)): 10_000,
}
DET_RTOL = 1e-8  # |det| >= DET_RTOL * scale^d, between the measured
#                  cond-1e6 pass (~3e-8) and cond-1e8 fail (~3e-10)


def _det4(a):
    """4x4 determinant by Laplace expansion on complementary 2x2 minors.

    The six ways to choose two columns for the top row-pair, each paired
    with the complementary two columns on the bottom row-pair. Twelve 2x2
    minors rather than the twenty-four terms of a full cofactor expansion,
    and every operation is a whole-array multiply, so the batch never
    leaves vectorized code.
    """
    def m2(r0, r1, c0, c1):
        return (a[..., r0, c0] * a[..., r1, c1]
                - a[..., r0, c1] * a[..., r1, c0])

    return (
        m2(0, 1, 0, 1) * m2(2, 3, 2, 3)
        - m2(0, 1, 0, 2) * m2(2, 3, 1, 3)
        + m2(0, 1, 0, 3) * m2(2, 3, 1, 2)
        + m2(0, 1, 1, 2) * m2(2, 3, 0, 3)
        - m2(0, 1, 1, 3) * m2(2, 3, 0, 2)
        + m2(0, 1, 2, 3) * m2(2, 3, 0, 1)
    )


def _det_and_scale(a):
    # Explicit per-dimension, NOT an else branch. This helper is shared by
    # det, slogdet and inv, and an else that quietly means "3" would return
    # a 3x3 determinant for a 4x4 stack the moment any caller widened its
    # floors - a silently wrong answer rather than a refusal.
    d = a.shape[-1]
    if d == 2:
        det = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    elif d == 3:
        det = (
            a[..., 0, 0] * (a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1])
            + a[..., 0, 1] * (a[..., 1, 2] * a[..., 2, 0] - a[..., 1, 0] * a[..., 2, 2])
            + a[..., 0, 2] * (a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0])
        )
    elif d == 4:
        det = _det4(a)
    else:
        raise ValueError(f"_det_and_scale has no closed form for d={d}")
    scale = np.abs(a).max(axis=(-2, -1))
    return det, scale


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 1 or kwargs:
        return False
    a = args[0]
    if type(a) is not np.ndarray or a.ndim < 3:
        return False
    d = a.shape[-1]
    if a.shape[-2] != d:
        return False
    floor = _FLOORS.get((d, a.dtype))
    if floor is None or math.prod(a.shape[:-2]) < floor:
        return False
    if not bool(np.isfinite(a).all()):
        return False
    det, scale = _det_and_scale(a)
    # scale clamped away from zero: an all-zero matrix has det == 0 AND
    # scale == 0, and 0 >= 0 would slip through to a silent 1/0 where
    # stock raises LinAlgError (caught by a dispatch probe)
    return bool((np.abs(det) >= DET_RTOL * np.maximum(scale, 1e-100) ** d).all())


def _inv2(a):
    m00 = a[..., 0, 0]; m01 = a[..., 0, 1]
    m10 = a[..., 1, 0]; m11 = a[..., 1, 1]
    inv_det = 1.0 / (m00 * m11 - m01 * m10)
    out = np.empty_like(a)
    out[..., 0, 0] = m11 * inv_det
    out[..., 0, 1] = -m01 * inv_det
    out[..., 1, 0] = -m10 * inv_det
    out[..., 1, 1] = m00 * inv_det
    return out


def _inv3(a):
    m00 = a[..., 0, 0]; m01 = a[..., 0, 1]; m02 = a[..., 0, 2]
    m10 = a[..., 1, 0]; m11 = a[..., 1, 1]; m12 = a[..., 1, 2]
    m20 = a[..., 2, 0]; m21 = a[..., 2, 1]; m22 = a[..., 2, 2]
    c00 = m11 * m22 - m12 * m21
    c10 = m12 * m20 - m10 * m22
    c20 = m10 * m21 - m11 * m20
    inv_det = 1.0 / (m00 * c00 + m01 * c10 + m02 * c20)
    out = np.empty_like(a)
    out[..., 0, 0] = c00 * inv_det
    out[..., 0, 1] = (m02 * m21 - m01 * m22) * inv_det
    out[..., 0, 2] = (m01 * m12 - m02 * m11) * inv_det
    out[..., 1, 0] = c10 * inv_det
    out[..., 1, 1] = (m00 * m22 - m02 * m20) * inv_det
    out[..., 1, 2] = (m02 * m10 - m00 * m12) * inv_det
    out[..., 2, 0] = c20 * inv_det
    out[..., 2, 1] = (m01 * m20 - m00 * m21) * inv_det
    out[..., 2, 2] = (m00 * m11 - m01 * m10) * inv_det
    return out


def _run(a):
    return _inv2(a) if a.shape[-1] == 2 else _inv3(a)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="inv_small_batch",
            op="numpy.linalg.inv",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000035",
                "source": "https://github.com/numpy/numpy/issues/17166",
                "license": "classical adjugate formulas; no third-party code",
                "comparison_mode": "numeric",
            },
        )
    )
