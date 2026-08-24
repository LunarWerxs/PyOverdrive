"""Fast paths: the singular-value family on 2x2/3x3 batches, from the
closed-form eigenvalues of A^T A.

Provenance (OPP-000056): the same per-matrix-LAPACK-dispatch vein as the
shipped eigvalsh/cholesky/qr small-batch families. numpy.linalg.svd,
norm(ord=2) and pinv all route each matrix in a stack through their own
LAPACK call (gesdd, plus gesdd again inside pinv), and at d in {2, 3}
the setup overhead dominates. The singular values of A are the square
roots of the eigenvalues of the 3x3-or-smaller symmetric gram matrix
A^T A, which has a closed form: the quadratic for d=2 and the same
trigonometric solution the shipped eigvalsh_3x3_trig path uses for d=3.
The gram is built entrywise here (six products for d=3), never with a
batched matmul - that alone was the difference between a 4x probe and a
20x one.

Three paths share that core, because they need three DIFFERENT accuracy
guarantees and therefore three different guards:

- norm2_small_batch (numpy.linalg.norm(a, ord=2, axis=(-2,-1))) needs
  only sigma_max, which the gram route delivers to eps at ANY condition
  number - measured relerr <= 7.8e-16 from cond 1e2 to 1e14, so it
  carries no CONDITIONING guard. It does carry the d=3 degeneracy guard
  below: a coalescing pair of singular values costs even the largest
  root its accuracy (4.5e-9 measured on a stack whose conditioning is a
  healthy 0.3), and no conditioning band sees that coming.
- svdvals_small_batch (numpy.linalg.svd(a, compute_uv=False)) returns
  every singular value. Judged the way LAPACK itself guarantees them -
  ABSOLUTE error against ||A||, the same standard the shipped
  eigvalsh paths use - the gram route holds tightly across the whole
  conditioning range. Relative accuracy on the SMALLEST value is NOT
  claimed and cannot be: squaring into the gram costs half the digits
  there, which is exactly why np.linalg.cond is NOT served by this
  module (it divides by sigma_min and would amplify that loss).
- pinv_small_batch (numpy.linalg.pinv) is the pseudo-inverse, which for
  a well-conditioned square matrix IS the inverse, so it is served by
  the adjugate formula (not by the gram at all). Its error grows only
  LINEARLY with the condition number - measured 2.2e-14 at cond 1e2 and
  2.2e-9 at cond 1e7 - so a band at cond <= 1e6 keeps it inside the
  1e-9 contract with a decade to spare. The gram is used only to
  MEASURE the conditioning cheaply.

CALIBRATION: PROVISIONAL pending the idle-box BATCH12-CAL run. Dev box
(fp 8f8198d9abab, numpy 2.4.5, guard-inclusive candidates): pinv 2x2
10.5x at batch 300, 24.7x at 1000, 24.4x at 10_000, 19.9x at 100_000;
pinv 3x3 5.9x/12.4x/9.5x; norm2 2x2 16.0-48.5x; svdvals 2x2 13.9-32.1x.

TWO INDEPENDENT d=3 HAZARDS, TWO GUARDS. Ill-conditioning (caught by
the sigma ratio) and near-degeneracy (a coalescing PAIR, caught by
DEGENERACY_MIN) are unrelated: a matrix can be perfectly conditioned
and still degenerate, which is exactly the case that slipped through
the first build of this module and was caught by its differential
tests. Both feed the same split-to-stock arm.

WHAT THE GUARDS COST, stated rather than buried: a stack that is
entirely ill-conditioned or degenerate pays for the closed form, learns
it cannot use it, and then pays stock as well - measured 0.88-0.96x, so
about a 10% regression on that input class. That is the irreducible
price of knowing: conditioning cannot be judged without measuring it.
It buys 8-24x on the ordinary case and a ~0.4% divert rate on random
input, which is a trade worth making, but it is a real cost on real
input and the calibration battery keeps cells for it.

BAND-TRIPPERS ARE SPLIT OUT, NOT BAILED ON. A whole-stack bail on a
conditioning band is a trap this project has already fallen into once
(qr_small_batch, batch 10): random matrices trip any such band with
some per-matrix probability, so a large enough stack almost surely
contains one and the entire large-batch regime collapses to stock. A
batch of 100_000 random 3x3s does contain one, measured. So the
trippers are gathered, served by stock, and scattered back; past
BAD_FRAC_MAX the whole call goes to stock in one batched pass, because
the gathered subset pays stock's per-matrix dispatch.

Correctness contract: plain float64 (..., d, d) ndarrays, d in {2, 3},
ndim >= 3, batch >= BATCH_MIN, all finite, default keyword arguments
only (pinv's rcond/rtol/hermitian, svd's full_matrices/hermitian and
any explicit compute_uv=True, and any norm spelling other than
ord=2 over the last two axes, all refuse to stock). Comparison mode:
numeric (spec section 9). Kill switches: pinv_small_batch,
norm2_small_batch, svdvals_small_batch.
"""

from __future__ import annotations

import math

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath, StockRaised

_F64 = np.dtype(np.float64)

# Measured floors (see CALIBRATION); the margin is already double digits
# well below these, so they are set for dispatch-tax safety.
BATCH_MIN = 100

# sigma_min / sigma_max floor for the adjugate pinv route: cond <= 1e6.
# Measured pinv relative error is linear in cond (2.2e-14 at 1e2, 2.4e-10
# at 1e6, 2.2e-9 at 1e7), so this keeps a decade of headroom under the
# 1e-9 contract.
PINV_SIGMA_RATIO_MIN = 1e-6

# Same idea, different exponent, for the singular VALUES. Forming the gram
# squares the condition number, so the error grows like eps*cond^2 rather
# than eps*cond - measured (worst of 4000 matrices, abs error / ||A||):
#
#   d=2   cond 1e4 -> 5.3e-13   1e6 -> 7.2e-11   1e7 -> 5.1e-10
#   d=3   cond 1e3 -> 2.9e-11   1e4 -> 2.7e-09   1e5 -> 3.0e-07
#
# d=3 carries a worse constant because its trigonometric solution
# amplifies error as the gram's eigenvalues cluster (the same mechanism
# documented in eigvalsh_3x3's DEGENERACY_MIN). Each floor is therefore
# the last measured decade that clears the 1e-9 contract with a decade of
# margin, per dimension - NOT one shared constant.
SVDVALS_SIGMA_RATIO_MIN = {2: 1e-6, 3: 1e-3}

# A SECOND, INDEPENDENT HAZARD, and the conditioning band does not catch
# it: a COALESCING PAIR of singular values, which can happen at any
# conditioning, including on a perfectly well-conditioned matrix. Both
# sizes suffer it, by different mechanisms, and both were measured
# shipping wrong answers before this guard existed - on stacks whose
# sigma_min/sigma_max is a healthy 0.3, so the band passed every one:
#
#   d=3, singular values (1, 0.6, 0.3)  ->  8.9e-16   (distinct: fine)
#   d=3, singular values (1, 1,   0.3)  ->  4.3e-09 svdvals, 4.5e-09 norm2
#   d=3, singular values (1, 0.3, 0.3)  ->  1.3e-08 svdvals
#   d=2, singular values (1, 1)         ->  7.5e-09
#
# i.e. four to thirteen times OUTSIDE the contract, silently. For d=3 the
# mechanism is the one eigvalsh_3x3's DEGENERACY_MIN documents: each root
# is reached through phi = arccos(r)/3, whose derivative
# 1/(3*sqrt(1 - r^2)) grows without bound as a pair coalesces. For d=2 it
# is catastrophic cancellation instead: the discriminant t^2 - 4*det goes
# to zero exactly when the pair coalesces, and the square root of a
# cancelling difference carries ~sqrt(eps) error.
#
# Different mechanisms, same shape of remedy, so one constant serves
# both: require the (squared, relative) separation margin to clear it -
# 1 - r^2 for d=3, the discriminant over t^2 for d=2. Measured at the
# threshold: 3.3e-11 for d=2 and ~1e-10 for d=3, i.e. a decade or more
# inside the contract. An exact multiple of the identity is NOT affected
# (d=3 takes the p == 0 branch and is exact, measured 6.7e-16), so it is
# not refused, and a random stack diverts only ~0.4% of its matrices.
DEGENERACY_MIN = 1e-12

# Past this fraction of band-trippers, one batched stock call beats
# gathering a subset and paying stock's per-matrix dispatch on it.
BAD_FRAC_MAX = 0.25


def _sv2_squared(a):
    """(sigma_max^2, sigma_min^2, degeneracy margin) for 2x2 stacks, from
    the gram's quadratic. The margin is the discriminant relative to t^2;
    see DEGENERACY_MIN for why callers must guard on it."""
    a00 = a[..., 0, 0]; a01 = a[..., 0, 1]
    a10 = a[..., 1, 0]; a11 = a[..., 1, 1]
    g00 = a00 * a00 + a10 * a10
    g11 = a01 * a01 + a11 * a11
    g01 = a00 * a01 + a10 * a11
    t = g00 + g11
    disc2 = np.maximum(t * t - 4.0 * (g00 * g11 - g01 * g01), 0.0)
    disc = np.sqrt(disc2)
    safe_t2 = np.where(t == 0.0, 1.0, t * t)
    return (t + disc) * 0.5, np.maximum((t - disc) * 0.5, 0.0), disc2 / safe_t2


def _sv3_squared(a):
    """(hi, mid, lo, 1 - r^2) squared singular values for 3x3 stacks: the
    shipped eigvalsh trigonometric form applied to an entrywise-built
    gram. The last item is the degeneracy margin (see DEGENERACY_MIN);
    callers that use more than the largest root must guard on it."""
    a00 = a[..., 0, 0]; a01 = a[..., 0, 1]; a02 = a[..., 0, 2]
    a10 = a[..., 1, 0]; a11 = a[..., 1, 1]; a12 = a[..., 1, 2]
    a20 = a[..., 2, 0]; a21 = a[..., 2, 1]; a22 = a[..., 2, 2]
    g00 = a00 * a00 + a10 * a10 + a20 * a20
    g11 = a01 * a01 + a11 * a11 + a21 * a21
    g22 = a02 * a02 + a12 * a12 + a22 * a22
    g01 = a00 * a01 + a10 * a11 + a20 * a21
    g02 = a00 * a02 + a10 * a12 + a20 * a22
    g12 = a01 * a02 + a11 * a12 + a21 * a22
    p1 = g01 * g01 + g02 * g02 + g12 * g12
    q = (g00 + g11 + g22) / 3.0
    d0 = g00 - q; d1 = g11 - q; d2 = g22 - q
    p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    safe = np.where(p == 0.0, 1.0, p)
    b0 = d0 / safe; b1 = d1 / safe; b2 = d2 / safe
    c01 = g01 / safe; c02 = g02 / safe; c12 = g12 / safe
    detb = (
        b0 * (b1 * b2 - c12 * c12)
        - c01 * (c01 * b2 - c12 * c02)
        + c02 * (c01 * c12 - b1 * c02)
    )
    r_raw = detb * 0.5
    r = np.clip(r_raw, -1.0, 1.0)
    phi = np.arccos(r) / 3.0
    hi = q + 2.0 * p * np.cos(phi)
    lo = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    mid = 3.0 * q - hi - lo
    # p == 0 is an exact multiple of the identity: all roots equal q, and
    # exactly so - not a degeneracy hazard, so it keeps a clean margin.
    margin = np.where(p == 0.0, 1.0, 1.0 - r_raw * r_raw)
    return np.maximum(hi, 0.0), np.maximum(mid, 0.0), np.maximum(lo, 0.0), margin


def _extremes_squared(a):
    """(sigma_max^2, sigma_min^2) for either supported size.

    The degeneracy margin is deliberately dropped here: the only caller
    is pinv, which reaches its answer through the adjugate and is immune
    (measured 1.5e-14 on exactly-degenerate input), and which uses these
    only to judge conditioning - where a degenerate matrix's ratio sits
    near 1, nowhere near the band.
    """
    if a.shape[-1] == 2:
        hi, lo, _margin = _sv2_squared(a)
        return hi, lo
    hi, _mid, lo, _margin = _sv3_squared(a)
    return hi, lo


def _shape_ok(a) -> bool:
    if type(a) is not np.ndarray or a.dtype != _F64 or a.ndim < 3:
        return False
    d = a.shape[-1]
    if a.shape[-2] != d or d not in (2, 3):
        return False
    if math.prod(a.shape[:-2]) < BATCH_MIN:
        return False
    return bool(np.isfinite(a).all())


# --- numpy.linalg.norm(a, ord=2, axis=(-2, -1)) -----------------------------


def _applicable_norm2(args: tuple, kwargs: dict) -> bool:
    if len(args) != 1 or set(kwargs) - {"ord", "axis", "keepdims"}:
        return False
    if kwargs.get("ord") != 2:
        return False
    if kwargs.get("keepdims", False):
        return False  # unmeasured shape class
    a = args[0]
    if not _shape_ok(a):
        return False
    # the matrix 2-norm over the trailing two axes, in either spelling
    return kwargs.get("axis") in ((-2, -1), (a.ndim - 2, a.ndim - 1))


def _run_norm2(x, **kwargs):
    if x.shape[-1] == 2:
        hi, _lo, margin = _sv2_squared(x)
    else:
        hi, _mid, _lo, margin = _sv3_squared(x)
    out = np.sqrt(hi)
    stock = GEARBOX.stock_fn("numpy.linalg.norm")
    served = _split_to_stock(
        out, x, margin < DEGENERACY_MIN,
        lambda sub: stock(sub, ord=2, axis=(-2, -1)),
    )
    if served is None:
        try:
            return stock(x, ord=2, axis=(-2, -1))
        except Exception as exc:  # noqa: BLE001 - stock's raise is the contract
            raise StockRaised(exc) from None
    return served


# --- numpy.linalg.svd(a, compute_uv=False) ----------------------------------


def _applicable_svdvals(args: tuple, kwargs: dict) -> bool:
    if len(args) != 1 or set(kwargs) - {"full_matrices", "compute_uv", "hermitian"}:
        return False
    if kwargs.get("compute_uv", True) is not False:
        return False  # the U/V factors have a sign convention we do not serve
    if kwargs.get("hermitian", False):
        return False
    return _shape_ok(args[0])


def _split_to_stock(out, a, bad, stock_call):
    """Serve the band-trippers from stock and scatter them back.

    Returns None when the whole call should go to stock instead (too many
    trippers to be worth gathering).
    """
    nbad = int(np.count_nonzero(bad))
    if not nbad:
        return out
    if nbad > BAD_FRAC_MAX * bad.size:
        return None
    n_batch = a.ndim - 2  # a is (*batch, d, d); out keeps those batch dims
    idx = np.flatnonzero(bad.reshape(-1))
    flat_in = a.reshape(-1, *a.shape[-2:])
    flat_out = out.reshape(-1, *out.shape[n_batch:])
    try:
        flat_out[idx] = stock_call(flat_in[idx])
    except Exception as exc:  # noqa: BLE001 - stock's raise is the contract
        raise StockRaised(exc) from None
    return out


def _run_svdvals(a, **kwargs):
    d = a.shape[-1]
    ratio2 = SVDVALS_SIGMA_RATIO_MIN[d] ** 2
    if d == 2:
        hi, lo, margin = _sv2_squared(a)
        sq = np.stack([hi, lo], axis=-1)
    else:
        hi, mid, lo, margin = _sv3_squared(a)
        sq = np.stack([hi, mid, lo], axis=-1)
    # ill-conditioned OR near-degenerate: two independent hazards
    bad = (sq[..., -1] <= ratio2 * sq[..., 0]) | (margin < DEGENERACY_MIN)
    out = np.sqrt(sq)
    stock = GEARBOX.stock_fn("numpy.linalg.svd")
    served = _split_to_stock(out, a, bad, lambda sub: stock(sub, compute_uv=False))
    if served is None:
        try:
            return stock(a, compute_uv=False)
        except Exception as exc:  # noqa: BLE001
            raise StockRaised(exc) from None
    return served


# --- numpy.linalg.pinv(a) ---------------------------------------------------


def _applicable_pinv(args: tuple, kwargs: dict) -> bool:
    if len(args) != 1 or kwargs:
        return False  # rcond/rtol/hermitian all change what stock computes
    return _shape_ok(args[0])


def _adjugate_inverse(a):
    d = a.shape[-1]
    out = np.empty_like(a)
    if d == 2:
        det = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
        inv = 1.0 / det
        out[..., 0, 0] = a[..., 1, 1] * inv
        out[..., 0, 1] = -a[..., 0, 1] * inv
        out[..., 1, 0] = -a[..., 1, 0] * inv
        out[..., 1, 1] = a[..., 0, 0] * inv
        return out
    c00 = a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1]
    c01 = a[..., 1, 2] * a[..., 2, 0] - a[..., 1, 0] * a[..., 2, 2]
    c02 = a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0]
    det = a[..., 0, 0] * c00 + a[..., 0, 1] * c01 + a[..., 0, 2] * c02
    inv = 1.0 / det
    out[..., 0, 0] = c00 * inv
    out[..., 1, 0] = c01 * inv
    out[..., 2, 0] = c02 * inv
    out[..., 0, 1] = (a[..., 0, 2] * a[..., 2, 1] - a[..., 0, 1] * a[..., 2, 2]) * inv
    out[..., 1, 1] = (a[..., 0, 0] * a[..., 2, 2] - a[..., 0, 2] * a[..., 2, 0]) * inv
    out[..., 2, 1] = (a[..., 0, 1] * a[..., 2, 0] - a[..., 0, 0] * a[..., 2, 1]) * inv
    out[..., 0, 2] = (a[..., 0, 1] * a[..., 1, 2] - a[..., 0, 2] * a[..., 1, 1]) * inv
    out[..., 1, 2] = (a[..., 0, 2] * a[..., 1, 0] - a[..., 0, 0] * a[..., 1, 2]) * inv
    out[..., 2, 2] = (a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]) * inv
    return out


def _run_pinv(a, **kwargs):
    hi2, lo2 = _extremes_squared(a)
    bad = lo2 <= (PINV_SIGMA_RATIO_MIN * PINV_SIGMA_RATIO_MIN) * hi2
    stock = GEARBOX.stock_fn("numpy.linalg.pinv")
    # For a square, well-conditioned matrix the pseudo-inverse IS the
    # inverse, which is what the adjugate formula gives.
    out = _adjugate_inverse(a)
    served = _split_to_stock(out, a, bad, stock)
    if served is None:
        try:
            return stock(a)
        except Exception as exc:  # noqa: BLE001 - stock's raise is the contract
            raise StockRaised(exc) from None
    return served


def register(gearbox) -> None:
    common = {
        "opportunity": "OPP-000056",
        "source": "closed-form eigenvalues of the 2x2/3x3 gram matrix; textbook formulas",
        "license": "textbook closed forms; no third-party code",
        "comparison_mode": "numeric",
    }
    gearbox.register(
        FastPath(
            name="norm2_small_batch",
            op="numpy.linalg.norm",
            applicable=_applicable_norm2,
            run=_run_norm2,
            provenance=common,
        )
    )
    gearbox.register(
        FastPath(
            name="svdvals_small_batch",
            op="numpy.linalg.svd",
            applicable=_applicable_svdvals,
            run=_run_svdvals,
            provenance=common,
        )
    )
    gearbox.register(
        FastPath(
            name="pinv_small_batch",
            op="numpy.linalg.pinv",
            applicable=_applicable_pinv,
            run=_run_pinv,
            provenance=common,
        )
    )
