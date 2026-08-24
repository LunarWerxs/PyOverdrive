"""Fast path: numpy.matmul(C complex 2-D, R real 2-D) with few C rows,
via two real GEMMs instead of stock's upcast-then-complex-GEMM.

Provenance (OPP-000029): numpy/numpy#24565 proposes exploiting the
real/imaginary structure of C @ R instead of promoting R to complex.
Stock upcasts R (a full copy at 2x the memory) and runs a complex GEMM
whose multiply-accumulate does roughly twice the real work the split
route needs; the split writes out.real = C.real @ R and
out.imag = C.imag @ R (C.real / C.imag are stride-2 views, consumed by
BLAS natively, no copy anywhere).

Measured regime (OPP-000029 + BATCH4-CAL batteries, fp 9bbe7063c555,
idle box, 0% load) - the regime is SHAPE-INVERTED from naive
expectation: for square/tall C the strided real GEMMs LOSE to
OpenBLAS's complex GEMM (0.36-0.84x measured at m in {200, 1000, 2000}
with small q), but when C has few rows against a large R the upcast
copy dominates stock and the split wins at EVERY measured cell:
m in {16, 64, 256} x n in {1000, 2000, 4000} x q in {500 .. 4000},
complex128 1.55-7.4x and complex64 1.60-7.5x (worst cell 1.55x at
(256, 1000, 1000); best 7.5x at (16, 2000, 2000)). The reverse
direction R @ C measured only 1.17-1.18x (below min-win) and stays on
stock, matching OPP-000027's vector-case finding.

Correctness contract:
- Applies only to matmul(C, R) with no kwargs, C a plain 2-D complex128
  (with R float64) or complex64 (with R float32) ndarray, R plain 2-D,
  inner dimensions matching, m <= M_MAX, n >= N_MIN, q >= Q_MIN, and
  BOTH operands all-finite: a complex multiply mixes real/imaginary
  cross terms exactly where the split discards them, so non-finite
  propagation differs (the shipped dot_mixed_view path measured this
  same hazard; OPP-000027).
- Two real GEMMs and one complex GEMM accumulate in different orders:
  agreement is numeric at BLAS-rounding scale (battery-checked at rtol
  1e-12 scaled for complex128, 1e-5 for complex64).

Comparison mode: numeric (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=matmul_split_complex or
pyoverdrive.disable_path("matmul_split_complex").

Implementation note: the in-run GEMMs go through stock_fn, never the
patched numpy.matmul name.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

_PAIRS = {
    np.dtype(np.complex128): np.dtype(np.float64),
    np.dtype(np.complex64): np.dtype(np.float32),
}
M_MAX = 256
N_MIN = 1_000
Q_MIN = 500


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 2 or kwargs:
        return False
    c, r = args
    if type(c) is not np.ndarray or type(r) is not np.ndarray:
        return False
    want_r = _PAIRS.get(c.dtype)
    if want_r is None or r.dtype != want_r:
        return False
    if c.ndim != 2 or r.ndim != 2 or c.shape[1] != r.shape[0]:
        return False
    m, n = c.shape
    q = r.shape[1]
    if m > M_MAX or n < N_MIN or q < Q_MIN:
        return False
    return bool(np.isfinite(c).all()) and bool(np.isfinite(r).all())


def _run(c, r):
    stock_matmul = GEARBOX.stock_fn("numpy.matmul")
    out = np.empty((c.shape[0], r.shape[1]), dtype=c.dtype)
    stock_matmul(c.real, r, out=out.real)
    stock_matmul(c.imag, r, out=out.imag)
    return out


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="matmul_split_complex",
            op="numpy.matmul",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000029",
                "source": "https://github.com/numpy/numpy/issues/24565",
                "license": "identity decomposition from first principles; no third-party code",
                "comparison_mode": "numeric",
            },
        )
    )
