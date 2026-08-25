"""Fast paths: numpy.linalg.det / slogdet / solve on 2x2/3x3 batches via
closed forms (cofactor expansion; Cramer's rule), 4x4 for det+slogdet.

Provenance (OPP-000045): stock runs one LAPACK call per matrix, so
stacked small matrices pay per-call overhead thousands of times over -
the same vein as the SHIPPED inv_small_batch (OPP-000035, numpy#17166)
and eigvalsh_2x2_closed (OPP-000030). numpy/numpy#20052 documents
slogdet's own overhead. Closed forms vectorize across the whole stack.

CALIBRATION, RE-DERIVED 2026-08-25 (fp 9bbe7063c555, idle box, 0% load,
numpy 2.5.2). Measured END TO END through the public API with the result
consumed, which is the only number a user experiences:

    det      d=2  1.54x at 200 -> peak 4.31x at 10k -> 2.15x at 100k
             d=3  1.52x at 300 -> peak 3.00x at 3k  -> 1.59x at 100k
             d=4  1.55x at 500 -> peak 2.64x at 3k  -> 1.09x at 100k
    slogdet  d=2  1.51x at 300 -> peak 3.16x at 10k -> 1.79x at 100k
             d=3  1.64x at 500 -> peak 2.52x at 10k -> 1.43x at 100k
             d=4  1.82x at 1000 -> peak 2.52x at 3k -> 1.07x at 100k
    solve    d=2  1.55x at 300 -> peak 3.82x at 10k -> 1.60x at 300k
             d=3  1.62x at 1000 -> peak 2.13x at 3k -> capped at 10k

THE PREVIOUS FLOORS SHIPPED A REGRESSION. They were set from
candidate-only timings (the old docstring quoted "det 2x2 6.2x at 100 to
91.9x at 20000"), which measure the closed form WITHOUT the guard. The
guard was in the predicate and it recomputed the determinant, so the
dispatched route computed it twice and scanned the array for finiteness
on top: end to end, det 2x2 at its own floor of 100 measured 0.70x, and
slogdet 3x3 at 300 measured 0.96x. Both dispatched. The lesson is
batch 13's, one level along: a margin measured without the guard is not
the margin, exactly as a margin measured without consuming the result is
not the margin.

Fixed by fusing the guard into the run for ALL THREE - one determinant,
checked on its own numbers, handing the whole call to stock via
StockRaised on refusal (cholesky_small_batch's pattern). That roughly
doubled every cell, and every floor above is then the first measured size
at or above 1.5x.

solve had the same disease and was measured separately: it was 0.91x at
its own 2x2 floor of 300 and 0.94x at batch 100_000 for 3x3, both
dispatching. Cramer's rule needs the determinant to divide by and the
guard needs it to judge, so computing it once serves both - the floor
went from 0.91x to 1.55x on the identical input. Its 3x3 arm also gained
a CAP, because Cramer builds every cofactor as a full-array temporary and
past L2 that loses to LAPACK outright (0.66x at 300_000).

Conditioning guard: identical policy to inv_small_batch, DET_RTOL=1e-8
against |det| / scale^d (that module's battery measured the pass at
cond 1e6 and the fail at 1e8; Cramer shares the adjugate's cancellation
behavior, and det/slogdet themselves lose relative accuracy exactly
where |det| collapses). Exactly-singular input is refused by the same
test: for solve, stock raises LinAlgError; for det, stock returns 0.0
exactly, which a rounded closed form will not; for slogdet the SIGN
would be unstable. Non-finite input is refused too, by testing the
DETERMINANT for finiteness in the run rather than scanning the input,
which is an n-element check rather than a 16n one and is equivalent
because every entry appears in every term of the expansion.

Correctness contract:
- det/slogdet: plain float64 ndarray, shape (..., d, d), d in {2, 3, 4},
  ndim >= 3, batch inside the measured window; no kwargs. Finiteness and
  conditioning are checked IN THE RUN, not the predicate, so a stack that
  fails either is served by stock mid-call and still gets stock's exact
  behaviour. slogdet returns stock's result type (probed at import) with
  signs EXACTLY equal to stock's on admitted input.
- The 4x4 determinant is a Laplace expansion on complementary 2x2 minors:
  twelve 2x2 minors rather than the twenty-four terms of a full cofactor
  expansion, every operation a whole-array multiply. Measured max relative
  error 2.0e-14 at batch 100 rising to 4.4e-12 at 100_000, which is why
  the 4x4 window is capped rather than open-ended.
- solve: additionally b must be a plain float64 ndarray of shape
  batch + (d, 1) (the measured column form); everything else refuses. Its
  guard is fused in exactly as det/slogdet's is, so a refused stack is
  served by stock mid-call with stock's LinAlgError intact.
- Different algorithm, different rounding: numeric mode. det/solve
  battery-checked at rtol 1e-9/1e-8; slogdet signs exact, logdet
  rtol 1e-9.

Comparison mode: numeric (spec section 9). Kill switches:
det_small_batch, slogdet_small_batch, solve_small_batch.
"""

from __future__ import annotations

import math

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath, StockRaised
from .inv_small_batch import DET_RTOL, _det_and_scale

_F64 = np.dtype(np.float64)

# (op-kind, d) -> (minimum batch, maximum batch or None).
#
# RE-DERIVED 2026-08-25 end-to-end through the public API, consumed, on the
# idle box (fp 9bbe7063c555, numpy 2.5.2, 0% load), with the guard fused
# into the run. The previous floors came from a candidate-level measurement
# and were BELOW break-even: det 2x2 at its own floor of 100 measured 0.70x
# end to end, i.e. the path dispatched and made the call 30% slower. Each
# floor below is the first measured size at or above 1.5x, which leaves
# room for a busier machine; the 4x4 caps are where the margin decays back
# towards 1.0x (1.09x at 100_000).
_WINDOWS = {
    ("det", 2): (200, None),       # 1.08x at 100, 1.54x at 200, peak 4.31x
    ("det", 3): (300, None),       # 1.23x at 200, 1.52x at 300, peak 3.00x
    ("det", 4): (500, 30_000),     # 1.11x at 300, 1.55x at 500, peak 2.64x
    ("slogdet", 2): (300, None),   # 1.25x at 200, 1.51x at 300, peak 3.16x
    ("slogdet", 3): (500, None),   # 1.27x at 300, 1.64x at 500, peak 2.52x
    ("slogdet", 4): (1_000, 30_000),  # 1.37x at 500, 1.82x at 1000, peak 2.52x
    ("solve", 2): (300, None),        # 1.27x at 200, 1.55x at 300, peak 3.82x
    # CAPPED: Cramer's rule builds every cofactor as a full-array temporary,
    # so past L2 it loses to LAPACK outright - 1.36x at 10k, 1.13x at 30k,
    # 0.96x at 100k and 0.66x at 300k. The cap sits at the last size with
    # real headroom rather than at the crossing itself.
    ("solve", 3): (1_000, 10_000),    # 1.25x at 500, 1.62x at 1000, peak 2.13x
}

# stock's return type for slogdet (a namedtuple in modern numpy)
_SLOGDET_RESULT = type(np.linalg.slogdet(np.eye(2)))


def _shape_ok(a, kind: str) -> bool:
    """Metadata only: no pass over the data at all.

    All three paths use this. The expensive half of a guard here - the
    finiteness scan and the determinant itself - is exactly the work the
    run performs. Asking the predicate for it made
    the shipped route compute the determinant TWICE and scan the array a
    third time, which cost the whole margin: measured end to end, the
    2x2 path was 0.70x at its own floor, i.e. slower than stock while
    still dispatching. See _det_if_safe.
    """
    if type(a) is not np.ndarray or a.dtype != _F64 or a.ndim < 3:
        return False
    d = a.shape[-1]
    if a.shape[-2] != d:
        return False
    window = _WINDOWS.get((kind, d))
    if window is None:
        return False
    batch = math.prod(a.shape[:-2])
    lo, hi = window
    return batch >= lo and (hi is None or batch <= hi)


def _admissible(a, det) -> bool:
    """Finite and well-conditioned, judged from a determinant ALREADY computed.

    Takes the determinant rather than computing one, so a caller that needs
    it anyway - solve, which divides by it - pays for it exactly once.

    Non-finiteness is detected on the DETERMINANT rather than by scanning
    the input: every entry of the matrix appears in every term of the
    expansion, so a non-finite entry always reaches the determinant as inf
    or nan (inf times zero is nan, inf minus inf is nan). That turns a
    16n-element scan into an n-element one.
    """
    if not bool(np.isfinite(det).all()):
        return False
    scale = np.abs(a).max(axis=(-2, -1))
    d = a.shape[-1]
    return bool((np.abs(det) >= DET_RTOL * np.maximum(scale, 1e-100) ** d).all())


def _det_if_safe(a):
    """The determinant, computed ONCE and guarded, or None if refused."""
    det, _ = _det_and_scale(a)
    return det if _admissible(a, det) else None


def _hand_to_stock(op: str, *args):
    """Graceful mid-run fallback, as cholesky_small_batch does it: refused
    input gets stock's exact behaviour, its LinAlgError included."""
    stock = GEARBOX.stock_fn(op)
    try:
        return stock(*args)
    except Exception as exc:  # noqa: BLE001 - stock's raise is the contract
        raise StockRaised(exc) from None


def _applicable_det(args: tuple, kwargs: dict) -> bool:
    return len(args) == 1 and not kwargs and _shape_ok(args[0], "det")


def _applicable_slogdet(args: tuple, kwargs: dict) -> bool:
    return len(args) == 1 and not kwargs and _shape_ok(args[0], "slogdet")


def _applicable_solve(args: tuple, kwargs: dict) -> bool:
    if len(args) != 2 or kwargs:
        return False
    a, b = args
    if not _shape_ok(a, "solve"):
        return False
    if type(b) is not np.ndarray or b.dtype != _F64:
        return False
    return b.shape == a.shape[:-2] + (a.shape[-1], 1)


def _run_det(a):
    det = _det_if_safe(a)
    if det is None:
        return _hand_to_stock("numpy.linalg.det", a)
    return det


def _run_slogdet(a):
    det = _det_if_safe(a)
    if det is None:
        return _hand_to_stock("numpy.linalg.slogdet", a)
    return _SLOGDET_RESULT(np.sign(det), np.log(np.abs(det)))


def _run_solve(a, b):
    """Cramer's rule, with the guard FUSED IN.

    Cramer needs the determinant to divide by, and the conditioning guard
    needs the determinant to judge. Asking the predicate for it meant
    computing it twice and scanning the array for finiteness on top, which
    cost more than the closed form saved: end to end this path measured
    0.91x at its own 2x2 floor and 0.94x at batch 100_000 for 3x3. One
    determinant now serves both.
    """
    d = a.shape[-1]
    b1 = b[..., 0, 0]
    b2 = b[..., 1, 0]
    out = np.empty_like(b)
    if d == 2:
        a11 = a[..., 0, 0]; a12 = a[..., 0, 1]
        a21 = a[..., 1, 0]; a22 = a[..., 1, 1]
        det = a11 * a22 - a12 * a21
        if not _admissible(a, det):
            return _hand_to_stock("numpy.linalg.solve", a, b)
        inv_det = 1.0 / det
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
    det = a11 * c11 + a12 * c12 + a13 * c13
    if not _admissible(a, det):
        return _hand_to_stock("numpy.linalg.solve", a, b)
    inv_det = 1.0 / det
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
