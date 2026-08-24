"""Fast path: numpy.linalg.qr on 2x2/3x3 batches via unrolled Householder
reflectors in LAPACK's own sign convention.

Provenance (OPP-000053): numpy/numpy#7179 (jcrist 2016, retitled by
seberg: "linalg.qr should be a gufunc") documented qr as the odd one
out of the linalg family - not a gufunc, holding the GIL, with dask
measuring no thread scaling. The literal ask landed in numpy 1.22
(qr_reduced/qr_complete/qr_r_raw gufuncs exist in numpy 2.x), but the
gufunc inner loop still makes per-matrix LAPACK calls (dgeqrf, then
dorgqr for Q), whose setup overhead dominates at d in {2, 3} - the
same per-matrix-dispatch class as the shipped eigvalsh_2x2/3x3 and
cholesky_small_batch families (OPP-000030/000048/000047). The
Householder factorization for d=2 (one reflector) and d=3 (two
reflectors) unrolls into branch-free vectorized arithmetic across the
stack and never calls LAPACK.

THE SIGN CONTRACT IS THE WHOLE PATH. qr's factors are unique only up
to per-column signs, so serving qr requires reproducing LAPACK's
choices exactly, not just "a valid QR". dlarfg pins them: with a
nonzero below-diagonal part, beta = -sign(alpha) * hypot(alpha, xnorm)
and tau = (beta - alpha)/beta; with the below-diagonal part exactly
zero, tau = 0 and beta = alpha (identity reflector, column passes
through unchanged). This module unrolls exactly those formulas, so
signs match stock on every served input class: random, negative
leading entries, already-upper-triangular (the tau = 0 branch), zero
first columns, all-zero matrices (qr never raises on finite input).
Agreement measured at |dQ| <= 8.7e-15, |dR| <= 9.3e-16 of scale. For
d=3 the SECOND reflector's inputs are computed, not raw, and dlarfg's
discontinuities there (identity-vs-flip at b32 == 0, copysign at
b22 == 0) mean noise-grade trailing blocks - rank-deficient or
repeated-column input - can make two valid factorizations disagree at
full scale; those MATRICES are detected by the QR_RTOL band mid-run,
split out, served by stock, and scattered back (whole stack to stock
past QR_BAD_FRAC_MAX trippers - see both constants' comments). d=2 has
only the raw first reflector and needs no band.

CALIBRATION (fp 9bbe7063c555, idle box, 0-1% load, numpy 2.5.2,
benchmarks/results/BATCH10-CAL/; dev box fp 8f8198d9abab contended
25-28%, agreeing in direction at every cell): mode='reduced' 2x2
4.08x at batch 300, 7.7-11.0x at 1000-3000, 6.5-8.8x at 10k-1M; 3x3
2.21x at 300, 3.0-4.2x at 1000-1M (split arm live at the large cells:
random stacks trip the band ~1e-5 per matrix, and the scattered stock
calls cost the 30k-1M cells nothing visible). mode='r' 2x2 2.78-8.2x,
3x3 1.67-3.6x. Below the floor, 3x3 n=100 loses outright (0.83x) and
2x2 n=100 mode='r' fails the bar on the dev box - see _FLOORS.

Correctness contract:
- Applies to qr(a), qr(a, mode) / qr(a, mode=...) with mode in
  {'reduced', 'complete', 'r'} - for square input reduced and complete
  coincide, so both are served by the same factorization; 'raw'
  (reflector internals) and anything else stays on stock. Input must
  be a plain float64 ndarray shaped (..., d, d), d in {2, 3}, ndim >=
  3, at least _FLOORS[d] matrices, every element finite (non-finite
  stays on stock so stock's own behavior - garbage-in-garbage-out or
  raise - is preserved bit-for-bit).
- 'reduced'/'complete' return stock's QRResult namedtuple (Q, R);
  'r' returns the plain R ndarray, exactly like stock.
- Different rounding order than LAPACK: numeric mode, checked by the
  battery at rtol 1e-9 scaled; signs are exact by construction.

Comparison mode: numeric (spec section 9). Kill switch:
qr_small_batch.
"""

from __future__ import annotations

import math

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

_F64 = np.dtype(np.float64)
_MODES = frozenset({"reduced", "complete", "r"})

# Second-reflector determinism band. Reflector 1 reads RAW input bits,
# so both routes make bit-identical branch and sign decisions there.
# Reflector 2's column (b22, b32) is COMPUTED, carrying ~eps*scale
# absolute noise, and dlarfg is DISCONTINUOUS at b32 == 0 (identity vs
# near-flip) and at sign(b22) (copysign flips the row): input that
# cancels the trailing block to noise grade can make the two routes
# disagree at FULL scale in R-row signs and Q columns - both valid
# factorizations, but the contract here is agreement with STOCK. The
# safe set: tau1 == 0 (no arithmetic happened, reflector 2 sees raw
# bits too), or both |b22| and |b32| at least QR_RTOL times the
# per-matrix scale, which caps the routes' H2 divergence at
# ~eps/QR_RTOL = 2e-10, inside the 1e-9 contract band. Everything
# else (rank-deficient, repeated-column, contrived cancellation input)
# is served BY stock through the mid-run split below.
QR_RTOL = 1e-6

# Band-trippers are SPLIT OUT and served by stock, then scattered back
# (the eigvalsh_3x3 batch-9 pattern) - a whole-stack bail would gut the
# large-n regime, because a random Gaussian stack trips the band with
# probability ~1e-5 per matrix, i.e. almost surely somewhere in a
# million-matrix stack. Past this fraction of trippers the whole call
# goes to stock instead: the gathered subset pays stock's per-matrix
# LAPACK dispatch, so a mostly-degenerate stack is cheaper served whole.
QR_BAD_FRAC_MAX = 0.25

# stock's result type for the (Q, R) modes, captured from stock itself
# (at import time numpy is unpatched; a 2-D input would refuse anyway)
_QRRESULT = type(np.linalg.qr(np.eye(2)))

# Floors (BATCH10-CAL, both fingerprints): 300 for both shapes. The
# n=100 cell wins on the idle box (2x2 1.99x reduced / 1.46x mode='r';
# 3x3 loses outright, 0.83x) but 2x2 mode='r' never clears the 1.3x
# bar on the dev box there (1.04-1.29x across three runs), so under
# the two-machine law the floor sits at the first cell clearing every
# mode on BOTH boxes: n=300 (2.0-4.1x dev, 1.66-4.09x idle). CHUNK:
# 4096 uniform - a few mid-size reduced cells prefer 1024 by ~15-20%
# (the 10_000 resonance again, see cholesky_small_batch), but every
# such cell still reads 3-6x, nowhere near the bar, so the adaptive
# machinery is not warranted here.
_FLOORS = {2: 300, 3: 300}
BATCH_MIN = max(_FLOORS.values())  # sizing convenience for tests/batteries
CHUNK = 4096


def _mode_of(args: tuple, kwargs: dict):
    """The resolved mode string, or None when the call must stay on stock."""
    if not 1 <= len(args) <= 2:
        return None
    if set(kwargs) - {"mode"}:
        return None
    if len(args) == 2 and "mode" in kwargs:
        return None  # duplicate: stock raises TypeError
    mode = args[1] if len(args) == 2 else kwargs.get("mode", "reduced")
    return mode if mode in _MODES else None


def _applicable(args: tuple, kwargs: dict) -> bool:
    if _mode_of(args, kwargs) is None:
        return False
    a = args[0]
    if type(a) is not np.ndarray or a.dtype != _F64 or a.ndim < 3:
        return False
    d = a.shape[-1]
    if a.shape[-2] != d or d not in (2, 3):
        return False
    if math.prod(a.shape[:-2]) < _FLOORS[d]:
        return False
    return bool(np.isfinite(a).all())


def _reflector(alpha, xnorm):
    """dlarfg's reflector scalars for one column, vectorized. alpha:
    leading entry; xnorm: norm of the below-diagonal part. Returns
    beta, tau, and the 1/(alpha - beta) scale for the v entries."""
    zero = xnorm == 0.0
    nrm = np.hypot(alpha, xnorm)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(zero, alpha, -np.copysign(nrm, alpha))
        tau = np.where(zero, 0.0, (beta - alpha) / np.where(beta == 0.0, 1.0, beta))
        inv = np.where(zero, 0.0, 1.0 / np.where(alpha == beta, 1.0, alpha - beta))
    return beta, tau, inv


def _qr2_chunk(c, qo, ro):
    # the single reflector reads raw input bits: deterministic vs stock
    # on every finite input, no band needed
    a11 = c[:, 0, 0]
    a21 = c[:, 1, 0]
    a12 = c[:, 0, 1]
    a22 = c[:, 1, 1]
    beta, tau, inv = _reflector(a11, np.abs(a21))
    v2 = a21 * inv
    s = a12 + v2 * a22
    ro[:, 0, 0] = beta
    ro[:, 0, 1] = a12 - tau * s
    ro[:, 1, 0] = 0.0
    ro[:, 1, 1] = a22 - tau * v2 * s
    if qo is not None:
        tv = tau * v2
        qo[:, 0, 0] = 1.0 - tau
        qo[:, 1, 0] = -tv
        qo[:, 0, 1] = -tv
        qo[:, 1, 1] = 1.0 - tv * v2
    return None


def _qr3_chunk(c, qo, ro):
    a11 = c[:, 0, 0]
    a21 = c[:, 1, 0]
    a31 = c[:, 2, 0]
    # reflector 1 on column (a11, a21, a31)
    beta1, tau1, inv1 = _reflector(a11, np.hypot(a21, a31))
    u2 = a21 * inv1
    u3 = a31 * inv1
    # apply H1 = I - tau1 u u^T (u = (1, u2, u3)) to columns 2 and 3
    a12 = c[:, 0, 1]
    a22 = c[:, 1, 1]
    a32 = c[:, 2, 1]
    s2 = a12 + u2 * a22 + u3 * a32
    b12 = a12 - tau1 * s2
    b22 = a22 - tau1 * u2 * s2
    b32 = a32 - tau1 * u3 * s2
    a13 = c[:, 0, 2]
    a23 = c[:, 1, 2]
    a33 = c[:, 2, 2]
    s3 = a13 + u2 * a23 + u3 * a33
    b13 = a13 - tau1 * s3
    b23 = a23 - tau1 * u2 * s3
    b33 = a33 - tau1 * u3 * s3
    # determinism band for reflector 2 (see QR_RTOL): tau1 == 0 means
    # (b22, b32) are raw input bits, deterministic either way
    band = QR_RTOL * np.abs(c).max(axis=(1, 2))
    safe = (tau1 == 0.0) | ((np.abs(b22) >= band) & (np.abs(b32) >= band))
    bad = None
    if not bool(safe.all()):
        # band-trippers get a clean identity second reflector here (their
        # values are wrong but finite, and the caller overwrites them
        # with stock's), so no warnings and no garbage propagate
        bad = ~safe
        b22 = np.where(bad, 1.0, b22)
        b32 = np.where(bad, 0.0, b32)
    # reflector 2 on the trailing column (b22, b32)
    beta2, tau2, inv2 = _reflector(b22, np.abs(b32))
    w3 = b32 * inv2
    t3 = b23 + w3 * b33
    ro[:, 0, 0] = beta1
    ro[:, 0, 1] = b12
    ro[:, 0, 2] = b13
    ro[:, 1, 0] = 0.0
    ro[:, 1, 1] = beta2
    ro[:, 1, 2] = b23 - tau2 * t3
    ro[:, 2, 0] = 0.0
    ro[:, 2, 1] = 0.0
    ro[:, 2, 2] = b33 - tau2 * w3 * t3
    if qo is not None:
        # Q = H1 @ diag(1, H2), columns assembled from the reflectors
        h12 = -tau1 * u2
        h13 = -tau1 * u3
        h22 = 1.0 - tau1 * u2 * u2
        h23 = -tau1 * u2 * u3
        h32 = -tau1 * u3 * u2
        h33 = 1.0 - tau1 * u3 * u3
        g22 = 1.0 - tau2
        g32 = -tau2 * w3
        g33 = 1.0 - tau2 * w3 * w3
        qo[:, 0, 0] = 1.0 - tau1
        qo[:, 1, 0] = h12
        qo[:, 2, 0] = h13
        qo[:, 0, 1] = h12 * g22 + h13 * g32
        qo[:, 1, 1] = h22 * g22 + h23 * g32
        qo[:, 2, 1] = h32 * g22 + h33 * g32
        qo[:, 0, 2] = h12 * g32 + h13 * g33
        qo[:, 1, 2] = h22 * g32 + h23 * g33
        qo[:, 2, 2] = h32 * g32 + h33 * g33
    return bad


def _run(a, *rest, **kwargs):
    mode = _mode_of((a, *rest), kwargs)
    d = a.shape[-1]
    factor = _qr2_chunk if d == 2 else _qr3_chunk
    want_q = mode != "r"
    r = np.empty(a.shape, dtype=a.dtype)
    q = np.empty(a.shape, dtype=a.dtype) if want_q else None
    src = a.reshape(-1, d, d)
    rdst = r.reshape(-1, d, d)
    qdst = q.reshape(-1, d, d) if want_q else None
    n = src.shape[0]
    bad_idx = []
    for s in range(0, n, CHUNK):
        bad = factor(
            src[s : s + CHUNK],
            None if qdst is None else qdst[s : s + CHUNK],
            rdst[s : s + CHUNK],
        )
        if bad is not None:
            bad_idx.append(s + np.flatnonzero(bad))
    if bad_idx:
        idx = np.concatenate(bad_idx)
        stock = GEARBOX.stock_fn("numpy.linalg.qr")
        if idx.size > QR_BAD_FRAC_MAX * n:
            # mostly inside the band: one batched stock call on the whole
            # stack beats per-matrix subset dispatch (see QR_BAD_FRAC_MAX)
            return stock(a, *rest, **kwargs)
        # split-and-recombine: only the band-trippers are served by stock
        # (qr never raises on the finite input the predicate admitted)
        sub = src[idx]
        if want_q:
            res = stock(sub)
            qdst[idx] = res.Q
            rdst[idx] = res.R
        else:
            rdst[idx] = stock(sub, mode="r")
    return _QRRESULT(q, r) if want_q else r


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="qr_small_batch",
            op="numpy.linalg.qr",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000053",
                "source": "https://github.com/numpy/numpy/issues/7179",
                "license": "unrolled Householder reflectors in LAPACK's dlarfg sign convention, textbook formulas; no third-party code",
                "comparison_mode": "numeric",
            },
        )
    )
