"""Fast path: numpy.dot(real 2-D, complex 1-D) via a view-as-real GEMV.

Provenance (OPP-000027): numpy/numpy#10468 (open since 2018) - a real
matrix times a complex vector runs 10-50x slower than the all-complex
BLAS path because numpy routes it through a non-BLAS loop. The thread's
own suggestion (upcast the real operand) measures ~1x on numpy 2.5 (the
upcast happens internally now); the winning route, found by this
project's reproducer, is different: view the contiguous complex128
vector as an (m, 2) float64 matrix, run ONE real matmul, and view the
(n, 2) result back as complex128. No complex arithmetic, no copy of the
large matrix. Dyno: 43.84x at 200x100, 12.05x at 10000x1000 (idle box,
0% load; benchmarks/results/OPP-000027/9bbe7063c555.json). The reverse
direction (complex matrix @ real vector) measured no gap (0.96-1.05x)
and stays on stock.

Correctness contract:
- Applies only to dot(A, b) with no kwargs, where A is a plain 2-D
  float64 ndarray, b a plain 1-D C-contiguous complex128 ndarray (the
  view needs contiguity), and A.shape[1] == b.size (a mismatched call
  raises on stock; refusing it keeps stock's own exception). Floor:
  A.size >= 20_000 (the smallest measured winning shape, 200x100).
- The route computes exactly the same two real accumulations stock's
  complex GEMV performs (result_re = A @ Re(b), result_im = A @ Im(b)),
  through the `@` operator - which resolves in C and cannot re-enter any
  patched name - so results agree to BLAS rounding: numeric mode, and
  the differential battery pins a tight scaled tolerance.
- Output: complex128, shape (n,), matching stock.

Comparison mode: numeric (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=dot_mixed_view or
pyoverdrive.disable_path("dot_mixed_view").
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath

SIZE_FLOOR = 20_000  # A.size at the smallest measured winning shape (200x100)

_F64 = np.dtype(np.float64)
_C128 = np.dtype(np.complex128)


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 2 or kwargs:
        return False
    a, b = args
    if type(a) is not np.ndarray or type(b) is not np.ndarray:
        return False
    if a.ndim != 2 or b.ndim != 1:
        return False
    if a.dtype != _F64 or b.dtype != _C128:
        return False
    if not b.flags.c_contiguous:
        return False
    if a.shape[1] != b.size or a.size < SIZE_FLOOR:
        return False
    # non-finite entries ANYWHERE diverge, both found by the differential
    # battery: in b, stock's complex GEMV multiplies by A's zero imaginary
    # part (0 * inf breeds NaN the clean real accumulations never produce);
    # in A, BLAS's complex multiply mixes components internally (a single
    # inf became NaN on stock where the real route keeps inf). Both scans
    # cost one pass against a 12-44x win regime.
    return bool(np.isfinite(b).all()) and bool(np.isfinite(a).all())


def _run(a, b):
    n = a.shape[0]
    bv = b.view(np.float64).reshape(-1, 2)
    # the @ operator resolves in C; no patched name is re-entered
    return (a @ bv).view(np.complex128).reshape(n)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="dot_mixed_view",
            op="numpy.dot",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000027",
                "source": "https://github.com/numpy/numpy/issues/10468",
                "license": "view-as-real decomposition, standard identity; no third-party code",
                "comparison_mode": "numeric (BLAS rounding; same two real accumulations)",
            },
        )
    )
