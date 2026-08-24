"""Fast path: numpy.linalg.eigvalsh on batches of 2x2 symmetric matrices
via the closed-form quadratic.

Provenance (OPP-000030): numpy/numpy#22158 - batched small-matrix eigh
routes each 2x2 through a per-matrix LAPACK call whose setup overhead
dominates (seberg's analysis). The values-only surface eigvalsh is the
shippable one: ascending eigenvalues are mathematically UNIQUE for a
given matrix (no sign or basis freedom outside exact degeneracy), unlike
eigenvectors, whose LAPACK sign convention blocks a comparison contract
(full eigh stays unshipped for the same reason np.partition did).

Route: for real symmetric [[a, b], [b, d]] read from the LOWER triangle
(stock's default UPLO='L' reads a[..., 1, 0] and ignores a[..., 0, 1]),
the eigenvalues are (a+d)/2 -/+ sqrt(((a-d)/2)^2 + b^2), ascending.

Measured (OPP-000030 + BATCH4-CAL batteries, fp 9bbe7063c555, idle box,
0% load): 2.51-2.55x at batch 100, 12.6x at 1000, 31.2x at 10_000, 6.6x
at 100_000, 5.9x at 1_000_000 (float64); 38.8x at 10_000 float32. Batch
10 loses (0.67x) and 30 straddles (1.11x), hence the floor of 100. The
1e12-condition witness passed the scaled tolerance at 30.7x.

Correctness contract:
- Applies only to eigvalsh(a) / eigvalsh(a, UPLO='L') where a is a plain
  float64/float32 ndarray shaped (..., 2, 2) with ndim >= 3, at least
  BATCH_MIN matrices, and every element finite. UPLO='U' (a different
  read triangle, unmeasured), 2-D single matrices, complex Hermitian
  input, other dtypes, and non-finite values all stay on stock -
  the last because LAPACK raises LinAlgError on non-convergence where
  the closed form would silently return NaNs.
- Agreement with stock is numeric at the absolute-error-vs-||A|| standard
  LAPACK itself promises; every battery cell passed at rtol 1e-9 (f64) /
  1e-3 (f32) scaled per matrix.

Comparison mode: numeric (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=eigvalsh_2x2_closed or
pyoverdrive.disable_path("eigvalsh_2x2_closed").
"""

from __future__ import annotations

import math

import numpy as np

from ..dispatcher.gearbox import FastPath

_DTYPES = frozenset((np.dtype(np.float64), np.dtype(np.float32)))
BATCH_MIN = 100


def _applicable(args: tuple, kwargs: dict) -> bool:
    if not 1 <= len(args) <= 2:
        return False
    if set(kwargs) - {"UPLO"}:
        return False
    if len(args) == 2 and "UPLO" in kwargs:
        return False  # duplicate: stock raises TypeError
    uplo = args[1] if len(args) == 2 else kwargs.get("UPLO", "L")
    if uplo != "L":
        return False
    a = args[0]
    if type(a) is not np.ndarray or a.dtype not in _DTYPES:
        return False
    if a.ndim < 3 or a.shape[-2:] != (2, 2):
        return False
    if math.prod(a.shape[:-2]) < BATCH_MIN:
        return False
    return bool(np.isfinite(a).all())


def _run(a, UPLO="L"):
    a00 = a[..., 0, 0]
    a10 = a[..., 1, 0]
    a11 = a[..., 1, 1]
    half = a.dtype.type(0.5)
    mid = half * (a00 + a11)
    disc = np.sqrt((half * (a00 - a11)) ** 2 + a10 * a10)
    return np.stack([mid - disc, mid + disc], axis=-1)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="eigvalsh_2x2_closed",
            op="numpy.linalg.eigvalsh",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000030",
                "source": "https://github.com/numpy/numpy/issues/22158",
                "license": "closed-form quadratic from first principles; no third-party code",
                "comparison_mode": "numeric",
            },
        )
    )
