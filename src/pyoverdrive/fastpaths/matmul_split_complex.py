"""Fast path: numpy.matmul(C complex 2-D, R real 2-D) with few C rows,
via two real GEMMs instead of stock's upcast-then-complex-GEMM.

Provenance (OPP-000029): numpy/numpy#24565 proposes exploiting the
real/imaginary structure of C @ R instead of promoting R to complex.
Stock upcasts R (a full copy at 2x the memory) and runs a complex GEMM
whose multiply-accumulate does roughly twice the real work the split
route needs; the split writes out.real = C.real @ R and
out.imag = C.imag @ R.

C.real and C.imag are STRIDE-2 VIEWS, and what numpy does with a
non-contiguous matmul operand is the whole ballgame here. numpy 2.3.0
"enable[d] using BLAS for matmul even when operands are non-contiguous
by copying if needed" (gh-23752). Before that release the two real
matmuls did not reach BLAS AT ALL - they fell to numpy's own loop - and
this path was 20-100x SLOWER than the stock route it replaces: 0.05x,
0.02x and 0.01x measured on numpy 2.0.2, 2.1.3 and 2.2.6, against
2.31x/1.36x on 2.3.5. That is why the package floor is numpy>=2.3 (see
pyproject); it was 2.0, and on 2.0 through 2.2 this path was a defect,
not an optimization.

An earlier version of this note claimed BLAS consumed the stride-2 views
"natively, no copy anywhere". That was never true on any numpy: 2.3+
copies them contiguous first and then calls BLAS, which is exactly what
the release note says it added. The win is real and the copy is part of
it - two real copies plus two real GEMMs still beat upcasting R to
complex and running a complex GEMM - but the mechanism as written would
have led someone to generalise this path to strided cases where no such
copy is affordable.

Historical regime (OPP-000029 + BATCH4-CAL batteries, fp 9bbe7063c555,
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

The complex64/float32 pair is now withdrawn globally: the quiet AMD
NumPy 2.4.5 public-API sweep confirmed 0.8871x and 0.9973x at the shape
boundary. See BURNDOWN-20260919/amd-complex-hist.json and the burndown
report. Historical wins on another machine do not justify that pair.

Correctness contract:
- Applies only to matmul(C, R) with no kwargs, C a plain 2-D complex128
  ndarray with R a plain 2-D float64 ndarray,
  inner dimensions matching, m <= M_MAX, n >= N_MIN, q >= Q_MIN, and
  BOTH operands all-finite: a complex multiply mixes real/imaginary
  cross terms exactly where the split discards them, so non-finite
  propagation differs (the shipped dot_mixed_view path measured this
  same hazard; OPP-000027).
- Two real GEMMs and one complex GEMM accumulate in different orders:
  agreement is numeric at BLAS-rounding scale (battery-checked at rtol
  1e-12 scaled for complex128).

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
}
# Quiet NumPy 2.3.0 lost at m=255 on the old boundary. The fixed row grid
# measured m=63/64 and n/q immediate inside neighbors winning at the tighter
# cap: BURNDOWN-20260919/intel-floor-matmul-grid.json (1.57-1.42% load).
M_MAX = 64
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
