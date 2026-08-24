"""Fast path: numpy.linalg.eigvalsh on batches of 3x3 symmetric matrices
via the trigonometric closed form.

Provenance (OPP-000048): the d=3 sibling of the shipped eigvalsh_2x2_closed
(OPP-000030, numpy/numpy#22158): batched small-matrix eigh routes each
matrix through a per-matrix LAPACK call whose setup overhead dominates
(seberg's analysis in-thread). For real symmetric 3x3 the characteristic
cubic has the classical trigonometric solution (Smith 1961 form): shift by
q = tr(A)/3, scale by p = sqrt(tr((A-qI)^2)/6), then the three roots are
q + 2p*cos(phi + 2k*pi/3) with phi = arccos(det(B)/2)/3 for the scaled
deviator B. Vectorizes across the whole stack; never calls LAPACK.

CALIBRATION (fp 9bbe7063c555, idle box, 0-1% load, numpy 2.5.2,
benchmarks/results/BATCH7-CAL/): f64 1.30x at batch 100, 2.67x at 300,
4.44x at 1000, 3.76x at 10_000, 2.86x at 100_000; f32 within a few
percent of the same curve. Floor 300: the 100 cell sits on the 1.3x
line for f64 and under it for f32. Dev-box probe agreed in direction
(1.41x at 100 rising to 7.06x at 1000, AMD Zen 4, numpy 2.4.5).

Route details that carry correctness:
- Lower-triangle reads only (stock's default UPLO='L' reads
  a[..., 1, 0], a[..., 2, 0], a[..., 2, 1] and ignores the upper
  triangle); UPLO='U' refuses to stock, exactly like the 2x2 path.
- float32 input is computed in float64 and cast back: the trig form
  loses about a decimal digit near clustered roots, which float64
  headroom absorbs; the result is at least as close to the true
  spectrum as stock's own float32 LAPACK path.
- arccos argument clamped to [-1, 1] (rounding can push det(B)/2 a few
  ulp outside); p == 0 (exact multiple of identity) handled by a safe
  divisor, all roots collapse to q exactly.
- NEAR-DEGENERATE SPLIT (batch 9; was a whole-stack bail): matrices
  whose scaled cubic has 1 - r^2 < DEGENERACY_MIN are gathered, served
  by stock, and scattered back into the trig result, so one coalesced
  pair no longer costs the whole stack its speedup. See the constants'
  comments for the derivation; without the test a coalescing
  eigenvalue pair degrades the trig form to ~sqrt(eps)*scale (measured
  1.6e-8 on diag(1, 1+1e-9, 5)) where LAPACK keeps eps-grade accuracy.
  Past DEGEN_FRAC_MAX degenerate cells the whole stack goes to stock
  in one batched call (the graceful-fallback pattern, StockRaised on a
  stock raise), because the split's per-matrix subset dispatch would
  cost more than it saves.
- A final 3-element sort enforces stock's documented ascending order:
  the roots emerge ordered mathematically, but rounding near exact
  degeneracy can invert a pair by an ulp.

Correctness contract:
- Applies only to eigvalsh(a) / eigvalsh(a, UPLO='L') where a is a
  plain float64/float32 ndarray shaped (..., 3, 3) with ndim >= 3, at
  least BATCH_MIN matrices, every element finite. UPLO='U', 2-D single
  matrices, complex Hermitian input, other dtypes, non-finite values
  all stay on stock (the last because LAPACK raises LinAlgError on
  non-convergence where the closed form would silently return NaNs).
- Agreement with stock is numeric at the absolute-error-vs-||A||
  standard, same as the 2x2 path: with the near-degenerate bail in
  place every admitted stack holds the 1e-9-scaled tolerance (battery
  cells measured ~1e-15 on random stacks). Clustered-pair input is not
  served less accurately - it is served by stock.

Comparison mode: numeric (spec section 9). Kill switch:
eigvalsh_3x3_trig.
"""

from __future__ import annotations

import math

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath, StockRaised

_DTYPES = frozenset((np.dtype(np.float64), np.dtype(np.float32)))
_F32 = np.dtype(np.float32)
# CALIBRATION (fp 9bbe7063c555, idle box, 0-1% load, numpy 2.5.2,
# benchmarks/results/BATCH7-CAL/): f64 0.48x at batch 30, 1.30x at 100,
# 2.67x at 300, 4.44x at 1000, 3.76x at 10_000, 2.86x at 100_000; f32
# 0.50x/1.26x/2.63x/4.34x/3.51x/2.83x on the same grid. 100 sits ON the
# 1.3x line for f64 and under it for f32, so the floor is the next
# measured notch.
BATCH_MIN = 300

# Near-degeneracy bail: the root separation enters through
# phi = arccos(r)/3, whose derivative 1/(3*sqrt(1 - r^2)) amplifies the
# ~eps error of r = det(B)/2 without bound as a pair of eigenvalues
# coalesces (r -> +/-1). At 1 - r^2 ~ eps the delivered accuracy decays
# to ~sqrt(eps)*scale - measured 1.6e-8 absolute on
# diag(1, 1+1e-9, 5) - where LAPACK stays at eps-grade. Requiring
# 1 - r^2 >= 1e-12 keeps the amplification below ~3e2, i.e. eigenvalue
# error under ~1e-13*p, a 10x margin against the 1e-9-scaled contract;
# cells violating it are served by stock through the split below.
DEGENERACY_MIN = 1e-12

# Split-and-recombine ceiling (batch 9): cells failing the degeneracy
# test are gathered, served by stock, and scattered back, so a stack is
# no longer punished whole for a few coalesced pairs. But the gathered
# subset pays stock's per-matrix LAPACK dispatch - the exact overhead
# this path exists to avoid - so past some fraction the one batched
# stock call on the whole stack is cheaper. Dev-box grid (n=10k/100k):
# split wins 3.2-3.8x at 0.1-10% degenerate, only 1.2-1.3x at 50%.
# Idle-box BATCH9-CAL cells (fp 9bbe7063c555, 0% load, n=10_000):
# 3.90x at 0% degenerate, 3.70x at 1%, 3.03x at 10%, 2.33x at 25%,
# 0.92x at 50% - and 2.86-2.92x at n=100_000 with 0-1%. The ceiling
# sits at the last measured fraction that clears the bar on both
# machines (dev box: 3.2-3.8x at 0.1-10%, ~1.2x at 50%).
DEGEN_FRAC_MAX = 0.25


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
    if a.ndim < 3 or a.shape[-2:] != (3, 3):
        return False
    if math.prod(a.shape[:-2]) < BATCH_MIN:
        return False
    return bool(np.isfinite(a).all())


def _run(a, UPLO="L"):
    f32 = a.dtype == _F32
    w = a.astype(np.float64) if f32 else a
    a11 = w[..., 0, 0]
    a22 = w[..., 1, 1]
    a33 = w[..., 2, 2]
    a21 = w[..., 1, 0]
    a31 = w[..., 2, 0]
    a32 = w[..., 2, 1]
    p1 = a21 * a21 + a31 * a31 + a32 * a32
    q = (a11 + a22 + a33) / 3.0
    d11 = a11 - q
    d22 = a22 - q
    d33 = a33 - q
    p2 = d11 * d11 + d22 * d22 + d33 * d33 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    safe = np.where(p == 0.0, 1.0, p)
    b11 = d11 / safe
    b22 = d22 / safe
    b33 = d33 / safe
    b21 = a21 / safe
    b31 = a31 / safe
    b32 = a32 / safe
    detb = (
        b11 * (b22 * b33 - b32 * b32)
        - b21 * (b21 * b33 - b32 * b31)
        + b31 * (b21 * b32 - b22 * b31)
    )
    r_raw = detb / 2.0
    bad = 1.0 - r_raw * r_raw < DEGENERACY_MIN
    nbad = int(np.count_nonzero(bad))
    if nbad > DEGEN_FRAC_MAX * bad.size:
        # near-coalesced pairs dominate the stack: the split's per-matrix
        # stock calls on the bad subset would cost more than handing the
        # whole call to stock's one batched pass (see DEGEN_FRAC_MAX)
        stock = GEARBOX.stock_fn("numpy.linalg.eigvalsh")
        try:
            return stock(a, UPLO)
        except Exception as exc:  # noqa: BLE001 - stock's raise is the contract
            raise StockRaised(exc) from None
    r = np.clip(r_raw, -1.0, 1.0)
    phi = np.arccos(r) / 3.0
    e_hi = q + 2.0 * p * np.cos(phi)
    e_lo = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    e_mid = 3.0 * q - e_hi - e_lo
    out = np.stack([e_lo, e_mid, e_hi], axis=-1)
    out.sort(axis=-1)
    if nbad:
        # split-and-recombine: only the near-degenerate cells go to stock
        # (their trig values above are the ones the accuracy contract cannot
        # hold); everyone else keeps the closed form. For float32 input the
        # subset is served by stock's own float32 route and the values
        # survive the f32->f64->f32 roundtrip exactly.
        stock = GEARBOX.stock_fn("numpy.linalg.eigvalsh")
        try:
            out[bad] = stock(a[bad], UPLO)
        except Exception as exc:  # noqa: BLE001 - stock's raise is the contract
            raise StockRaised(exc) from None
    return out.astype(np.float32) if f32 else out


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="eigvalsh_3x3_trig",
            op="numpy.linalg.eigvalsh",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000048",
                "source": "https://github.com/numpy/numpy/issues/22158",
                "license": "classical trigonometric cubic solution (Smith 1961 form), textbook math; no third-party code",
                "comparison_mode": "numeric",
            },
        )
    )
