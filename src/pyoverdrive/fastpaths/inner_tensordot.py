"""Fast path: numpy.inner via a single tensordot GEMM when ndim > 2.

Provenance (OPP-000002): numpy/numpy#12778 (open since 2019, duplicate of
#619 from 2012). Maintainer-confirmed mechanism: PyArray_MatrixProduct2 makes
one BLAS call only when both operands have ndim <= 2 and otherwise loops 2-D
BLAS calls; np.tensordot reshapes to a single GEMM, reported ~10x faster.
That ~10x is the ISSUE's number, not ours - see the measured regime below,
which is where this path is allowed to run.

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


# MEASURED REGIME, added 2026-08-25 after this path was found dispatching
# into a 0.38x LOSS - 2.6x slower than stock - on ordinary small operands.
#
# It had no size gate at all: any ndim>2 pair was accepted, and the only
# speedup ever quoted for it (~10x) was the upstream issue's own number,
# never re-measured through the predicate and run. Measured end to end on
# the idle box over rows_a x rows_b x k = {4,8,20,50} x {4,16,64,256} x
# {8,32,128,512}, the path ranges from 0.38x to 6.98x and the wins and
# losses INTERLEAVE - (4, 256, 512) is 0.43x while (20, 16, 512) is 1.23x,
# so no single function of volume, output size or contraction length
# separates them.
#
# When a regime cannot be separated, the gate has to be restrictive rather
# than clever: admit only the corner where every measured cell wins, and
# leave everything else on stock. That forfeits real wins (the k=512 corner
# reaches 3.42x, mixed in with 0.43x losses) and that is the correct trade -
# dispatching into 0.38x is worse than declining a 3x.
#
# Admitted region, every measured cell 1.27x-6.98x:
#   rows_a >= 8, rows_b >= 64, rows_a*rows_b >= 1024, k <= 128
# where rows_a = prod(a.shape[:-1]) is the number of separate 2-D BLAS calls
# stock would make, and k is the contracted length whose transpose the
# single GEMM has to pay for.
_ROWS_A_MIN = 8
_ROWS_B_MIN = 64
_OUT_MIN = 1_024
_K_MAX = 128


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 2 or kwargs:
        return False
    a, b = args
    if (
        type(a) is not np.ndarray
        or type(b) is not np.ndarray
        or not (a.ndim > 2 or b.ndim > 2)
        or a.ndim < 1
        or b.ndim < 1
        or a.dtype not in _FLOAT_DTYPES
        or a.dtype != b.dtype
        or a.shape[-1] != b.shape[-1]
        or not a.size
        or not b.size
    ):
        return False
    k = a.shape[-1]
    rows_a = a.size // k
    rows_b = b.size // k
    return (
        k <= _K_MAX
        and rows_a >= _ROWS_A_MIN
        and rows_b >= _ROWS_B_MIN
        and rows_a * rows_b >= _OUT_MIN
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
