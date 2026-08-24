"""Fast path: numpy.inner via a single tensordot GEMM when ndim > 2.

Provenance (OPP-000002): numpy/numpy#12778 (open since 2019, duplicate of
#619 from 2012). Maintainer-confirmed mechanism: PyArray_MatrixProduct2 makes
one BLAS call only when both operands have ndim <= 2 and otherwise loops 2-D
BLAS calls; np.tensordot reshapes to a single GEMM, reported ~10x faster.

Correctness contract:
- Applies only when at least one operand has ndim > 2 (the 1-D/2-D regimes
  already take the single-BLAS route and MUST stay on stock; rerouting them
  would be a regression).
- Plain ndarrays only, matching float dtypes, matching contraction length
  (mismatched shapes fall back so stock raises the familiar error).

Comparison mode: NUMERIC, not bit-identical (spec section 9). The single GEMM
sums in a different order than the looped per-slice GEMMs, so results agree
to tight relative tolerance but may differ in final ulps. np.inner makes no
bit-stability promise across versions or BLAS builds, and upstream's own
blessed fix is this same reroute; still, this is the documented trade of the
path. Kill switch: PYOVERDRIVE_DISABLE=inner_tensordot or
pyoverdrive.disable_path("inner_tensordot").

Peak memory differs: tensordot materializes reshaped/transposed copies.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath

_FLOAT_DTYPES = frozenset(np.dtype(t) for t in (np.float32, np.float64))


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 2 or kwargs:
        return False
    a, b = args
    return (
        type(a) is np.ndarray
        and type(b) is np.ndarray
        and (a.ndim > 2 or b.ndim > 2)
        and a.ndim >= 1
        and b.ndim >= 1
        and a.dtype in _FLOAT_DTYPES
        and a.dtype == b.dtype
        and a.shape[-1] == b.shape[-1]
        and a.size > 0
        and b.size > 0
    )


def _inner_tensordot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.tensordot(a, b, axes=(-1, -1))


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="inner_tensordot",
            op="numpy.inner",
            applicable=_applicable,
            run=_inner_tensordot,
            provenance={
                "opportunity": "OPP-000002",
                "source": "https://github.com/numpy/numpy/issues/12778",
                "license": "reroute to numpy's own tensordot; no third-party code",
                "comparison_mode": "numeric (rtol ~1e-9 float64, ~1e-4 float32)",
            },
        )
    )
