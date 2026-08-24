"""Fast path: numpy.linalg.cholesky on 2x2/3x3 batches via the
Cholesky-Crout closed form, fused guard-in-run, chunked.

Provenance (OPP-000047): same per-matrix-LAPACK-dispatch vein as the
shipped linalg_small_batch (OPP-000045) and inv_small_batch
(OPP-000035): stock routes each matrix in a stack through one potrf
call, so small-matrix batches pay call overhead thousands of times
over. scipy/scipy#24474 documents live demand for exactly this batched
small-matrix decomposition surface. The closed form (explicit
Cholesky-Crout formulas for d=2/d=3) vectorizes across the stack and
never calls LAPACK.

ARCHITECTURE (batch 9 rewrite): the original path computed the pivot
guard in _applicable and then recomputed the same arithmetic in _run -
the guard pivots ARE the factorization's intermediates (p2 = a22 -
l21^2, p3 = a33 - l31^2 - l32^2), so the guard now rides the single
factorization pass, checked between stages, and _applicable keeps only
the cheap metadata tests. A guard failure anywhere in the stack
(non-finite entry, pivot under the band) hands the WHOLE call to stock
mid-run - the graceful-fallback pattern (StockRaised on a stock
raise), same as eigvalsh_3x3_trig's degeneracy bail - so refused input
still gets stock's exact behavior, including its LinAlgError. The pass
runs in cache-sized chunks so every temporary stays L2-resident;
the unfused route's full-stack temporaries are what caused the L2
cliff that forced the original 3x3 cap (batch 5000 bistable: 1.37x
min-of-9 vs 0.98-1.03x battery medians).

CALIBRATION (fp 9bbe7063c555, idle box, 0% load, numpy 2.5.2,
benchmarks/results/BATCH9-CAL/; dev box fp 8f8198d9abab re-probed at
1% load and agreeing at every cell): under the chunk policy below, 2x2
wins 1.86-2.30x at every batch 1000-1M (the unfused route's weakest
admitted cell, 1.29x at 20_000, is now 2.30x) and 3x3 wins 1.59-1.90x
at 1000-10_000 and 1.66-1.88x at 16_384-1M on both boxes - so BOTH
windows are floor 1000, no cap (the unfused route's 3x3 cap at 3000,
forced by its bistable 5000 cell, is gone: that cell now reads 1.90x
idle / 1.92x dev). Below the floor, 300 reads 1.10-1.30x. The chunk
policy is measured, not guessed: chunk 4096 dips to 1.11x idle /
1.23x dev in a narrow resonance around batch 10_000 for 3x3 only
(recovered by 16_384: 1.88x), while chunk 1024 is smooth there
(1.56-1.59x) but ~25% slower in the >=30_000 regime; 2x2 never dips
(2.28x at 10_000, 2.30x at 20_000). Hence: 2x2 always 4096; 3x3 uses
1024 below _CHUNK_SWITCH_3X3 = 30_000 (the smallest cell where 4096
is verified clean on BOTH machines) and 4096 from there up. Max
closed-form error measured 3.6e-17 of scale (unchanged from the
unfused route: same arithmetic, same order). BATCH7-CAL holds the
unfused route's history.

Positivity guard (unchanged semantics): the d leading pivots of the
factorization (a11, then each Schur complement diagonal) must each be
at least PIVOT_RTOL times the matrix's max-|entry| scale. This one
test does triple duty: it refuses non-positive-definite input (where
stock raises LinAlgError and a naive closed form would sqrt a
negative), it refuses the near-semidefinite boundary band (where
LAPACK's own rounding decides raise-vs-succeed and the two routes
could disagree), and it bounds the conditioning so the closed form's
rounding stays at the measured 1e-17-of-scale level. Non-finite input
refused by an isfinite scan, first in each chunk, so no stage ever
divides by or roots a value the guard has not cleared - the fused
route needs no errstate suppression at all.

Correctness contract:
- exactly cholesky(a): plain float64 ndarray, shape (..., d, d), d in
  {2, 3}, ndim >= 3, batch inside _WINDOWS[d]; no kwargs (numpy 2.x's
  upper= keyword refuses to stock, as does anything else). Non-finite
  or guard-refused stacks are served BY stock through the mid-run
  fallback (decision reads PATH; behavior is stock's own, unbranded).
- Only the LOWER triangle of each input matrix is read, exactly like
  stock (potrf UPLO='L'; numpy documents that the upper triangle is
  ignored). The strict upper triangle of the result is exactly 0.0,
  like stock's.
- Different algorithm, different rounding: numeric mode, checked by
  the battery at rtol 1e-9 scaled per matrix.

Comparison mode: numeric (spec section 9). Kill switch:
cholesky_small_batch.
"""

from __future__ import annotations

import math

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath, StockRaised

_F64 = np.dtype(np.float64)
PIVOT_RTOL = 1e-8

# d -> (min batch, max batch or None): the measured winning window, see
# the CALIBRATION section above
_WINDOWS = {2: (1_000, None), 3: (1_000, None)}

# chunk of the flattened stack processed per pass: keeps every
# temporary (six n-vectors for d=3) inside L2, which is what removes
# the old full-stack route's memory cliff. The sizes are measured, not
# guessed - see CALIBRATION: 4096 everywhere except the 3x3 mid-size
# band, where it hits a narrow resonance around batch 10_000 on both
# benchmark machines and 1024 is smooth.
_CHUNK_SWITCH_3X3 = 30_000


def _chunk_for(d: int, batch: int) -> int:
    if d == 3 and batch < _CHUNK_SWITCH_3X3:
        return 1024
    return 4096


def _applicable(args: tuple, kwargs: dict) -> bool:
    if len(args) != 1 or kwargs:
        return False
    a = args[0]
    if type(a) is not np.ndarray or a.dtype != _F64 or a.ndim < 3:
        return False
    d = a.shape[-1]
    if a.shape[-2] != d:
        return False
    window = _WINDOWS.get(d)
    if window is None:
        return False
    lo, hi = window
    batch = math.prod(a.shape[:-2])
    return batch >= lo and (hi is None or batch <= hi)


def _factor2(c, o):
    """Fused factor-and-guard for one 2x2 chunk; False = guard refused."""
    if not bool(np.isfinite(c).all()):
        return False
    a11 = c[:, 0, 0]
    bound = PIVOT_RTOL * np.maximum(np.abs(c).max(axis=(1, 2)), 1e-100)
    if not bool((a11 >= bound).all()):
        return False
    l11 = np.sqrt(a11)
    l21 = c[:, 1, 0] / l11
    p2 = c[:, 1, 1] - l21 * l21
    if not bool((p2 >= bound).all()):
        return False
    o[:, 0, 0] = l11
    o[:, 0, 1] = 0.0
    o[:, 1, 0] = l21
    o[:, 1, 1] = np.sqrt(p2)
    return True


def _factor3(c, o):
    """Fused factor-and-guard for one 3x3 chunk; False = guard refused."""
    if not bool(np.isfinite(c).all()):
        return False
    a11 = c[:, 0, 0]
    bound = PIVOT_RTOL * np.maximum(np.abs(c).max(axis=(1, 2)), 1e-100)
    if not bool((a11 >= bound).all()):
        return False
    l11 = np.sqrt(a11)
    l21 = c[:, 1, 0] / l11
    l31 = c[:, 2, 0] / l11
    p2 = c[:, 1, 1] - l21 * l21
    if not bool((p2 >= bound).all()):
        return False
    l22 = np.sqrt(p2)
    l32 = (c[:, 2, 1] - l31 * l21) / l22
    p3 = c[:, 2, 2] - l31 * l31 - l32 * l32
    if not bool((p3 >= bound).all()):
        return False
    o[:, 0, 0] = l11
    o[:, 0, 1] = 0.0
    o[:, 0, 2] = 0.0
    o[:, 1, 0] = l21
    o[:, 1, 1] = l22
    o[:, 1, 2] = 0.0
    o[:, 2, 0] = l31
    o[:, 2, 1] = l32
    o[:, 2, 2] = np.sqrt(p3)
    return True


def _run(a):
    d = a.shape[-1]
    factor = _factor2 if d == 2 else _factor3
    out = np.empty(a.shape, dtype=a.dtype)
    src = a.reshape(-1, d, d)
    dst = out.reshape(-1, d, d)  # out is fresh and contiguous: a view
    n = src.shape[0]
    chunk = _chunk_for(d, n)
    for s in range(0, n, chunk):
        if not factor(src[s : s + chunk], dst[s : s + chunk]):
            # guard refused somewhere in the stack: the whole call is
            # stock's (its LinAlgError on non-PD input is the contract)
            stock = GEARBOX.stock_fn("numpy.linalg.cholesky")
            try:
                return stock(a)
            except Exception as exc:  # noqa: BLE001 - stock's raise is the contract
                raise StockRaised(exc) from None
    return out


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="cholesky_small_batch",
            op="numpy.linalg.cholesky",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000047",
                "source": "https://github.com/scipy/scipy/issues/24474",
                "license": "Cholesky-Crout closed form, textbook formulas; no third-party code",
                "comparison_mode": "numeric",
            },
        )
    )
