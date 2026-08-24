"""Fast paths: integer numpy.matmul / numpy.dot via exact float64 BLAS.

Provenance (OPP-000044): numpy has no BLAS path for integer matmul
(numpy/numpy#14556, open); int64/int32 matrix multiply runs a naive
loop, tens of times slower than float64 GEMM at the same size. When
k * max|A| * max|B| < 2**53 every float64 partial sum along the
contraction is an exactly representable integer, so cast -> BLAS ->
round-trip cast is BIT-EXACT, provably: products are bounded by
max|A|*max|B|, partial sums by k times that, and every integer of
magnitude < 2**53 is exact in float64. For int32 the bound is 2**31,
which simultaneously proves stock could not have wrapped its int32
accumulator, so results agree exactly there too.

The bound check needs max magnitudes; those come from min/max
reductions converted to Python ints (np.abs on INT64_MIN would wrap).
Both O(n^2) scans and both casts are inside the measured win.

CALIBRATION (fp 9bbe7063c555, idle box, 0% load, numpy 2.5.2,
benchmarks/results/BATCH6-CAL/, entries in [-1000, 1000)): int64
2.62x at n=50 rising to 28.48x at n=800; int32 2.69x at n=50 to
23.51x at n=800; n=30 measured 1.15-1.21x, under the 1.3x bar, so the
floor below is min dimension >= 50. Every cell bit-identical.

Correctness contract:
- both operands plain 2-D C-order-or-F-order ndarrays of the SAME
  dtype, int64 or int32; shapes (m, k) x (k, n) with min(m, k, n) >=
  MIN_DIM; the exactness bound must hold (checked per call). Anything
  else - mixed dtypes, stacks, vectors, out=, casting kwargs - forces
  stock. The operator form a @ b and the a.dot(b) method bypass the
  patch entirely (documented reach limitation, as with
  matmul_split_complex).
- within the bound: bit-identical to stock, including dtype.

Comparison mode: bit-identical. Kill switches: matmul_int_blas,
dot_int_blas.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

MIN_DIM = 50  # smallest measured winning square size
_BOUNDS = {
    np.dtype(np.int64): 2**53,
    np.dtype(np.int32): 2**31,
}


def _max_abs(a: np.ndarray) -> int:
    # Python-int magnitudes: np.abs(INT64_MIN) wraps, int() never does
    return max(-int(np.min(a)), int(np.max(a)))


def _admissible(args: tuple, kwargs: dict):
    if kwargs or len(args) != 2:
        return None
    x, y = args
    if type(x) is not np.ndarray or type(y) is not np.ndarray:
        return None
    if x.dtype != y.dtype or x.dtype not in _BOUNDS:
        return None
    if x.ndim != 2 or y.ndim != 2:
        return None
    m, k = x.shape
    k2, n = y.shape
    if k != k2 or min(m, k, n) < MIN_DIM:
        return None
    return x, y


def _applicable(args: tuple, kwargs: dict) -> bool:
    adm = _admissible(args, kwargs)
    if adm is None:
        return False
    x, y = adm
    bound = _BOUNDS[x.dtype]
    return x.shape[1] * _max_abs(x) * _max_abs(y) < bound


def _run(x, y):
    r = x.astype(np.float64) @ y.astype(np.float64)
    return r.astype(x.dtype)


def register(gearbox) -> None:
    for op, path_name in (
        ("numpy.matmul", "matmul_int_blas"),
        ("numpy.dot", "dot_int_blas"),
    ):
        gearbox.register(
            FastPath(
                name=path_name,
                op=op,
                applicable=_applicable,
                run=_run,
                provenance={
                    "opportunity": "OPP-000044",
                    "source": "https://github.com/numpy/numpy/issues/14556",
                    "license": "cast-to-BLAS with exactness bound, widely known technique; no third-party code",
                    "comparison_mode": "bit-identical",
                },
            )
        )
