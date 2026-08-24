"""OPP-000053: batched 2x2/3x3 numpy.linalg.qr via unrolled Householder.

numpy/numpy#7179 (jcrist 2016, retitled by seberg "linalg.qr should be a
gufunc") flagged qr as the odd one out of the linalg family. The gufunc
ask landed in numpy 1.22 (qr_reduced/qr_complete/qr_r_raw exist in
numpy 2.x), but the gufunc inner loop still makes per-matrix LAPACK
calls (dgeqrf, then dorgqr for Q), whose setup overhead dominates at
d in {2, 3}. Unrolling the Householder reflectors in LAPACK's own
dlarfg sign convention (beta = -sign(alpha)*hypot(alpha, xnorm) when
the below-diagonal part is nonzero; tau = 0, beta = alpha when it is
exactly zero) reproduces stock's Q and R to ~1e-15 of scale WITH
matching signs, and vectorizes across the stack.

House rules: never imports pyoverdrive. The candidate here is the
whole-stack unchunked closed form (the shipped path adds chunking; the
claim being reproduced is the per-matrix-dispatch margin itself).
Result JSON: benchmarks/results/OPP-000053/.
Run: .venv/Scripts/python benchmarks/historical/opp_000053_qr_small_batch.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000053", "batched small qr via unrolled Householder")
rng = np.random.default_rng(7179)


def qr_close(rtol):
    def _chk(c, b):
        cq, cr = c
        bq, br = b
        scale = max(1.0, float(np.abs(br).max()))
        return bool(np.allclose(cr, br, rtol=rtol, atol=rtol * scale)) and bool(
            np.allclose(cq, bq, rtol=rtol, atol=rtol)
        )

    return _chk


def _reflector(alpha, xnorm):
    zero = xnorm == 0.0
    nrm = np.hypot(alpha, xnorm)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(zero, alpha, -np.copysign(nrm, alpha))
        tau = np.where(zero, 0.0, (beta - alpha) / np.where(beta == 0.0, 1.0, beta))
        inv = np.where(zero, 0.0, 1.0 / np.where(alpha == beta, 1.0, alpha - beta))
    return beta, tau, inv


def qr2_householder(a):
    a11, a21 = a[:, 0, 0], a[:, 1, 0]
    a12, a22 = a[:, 0, 1], a[:, 1, 1]
    beta, tau, inv = _reflector(a11, np.abs(a21))
    v2 = a21 * inv
    s = a12 + v2 * a22
    r = np.empty_like(a)
    r[:, 0, 0] = beta
    r[:, 0, 1] = a12 - tau * s
    r[:, 1, 0] = 0.0
    r[:, 1, 1] = a22 - tau * v2 * s
    q = np.empty_like(a)
    tv = tau * v2
    q[:, 0, 0] = 1.0 - tau
    q[:, 1, 0] = -tv
    q[:, 0, 1] = -tv
    q[:, 1, 1] = 1.0 - tv * v2
    return q, r


def qr3_householder(a):
    a11, a21, a31 = a[:, 0, 0], a[:, 1, 0], a[:, 2, 0]
    beta1, tau1, inv1 = _reflector(a11, np.hypot(a21, a31))
    u2 = a21 * inv1
    u3 = a31 * inv1
    a12, a22, a32 = a[:, 0, 1], a[:, 1, 1], a[:, 2, 1]
    s2 = a12 + u2 * a22 + u3 * a32
    b12, b22, b32 = a12 - tau1 * s2, a22 - tau1 * u2 * s2, a32 - tau1 * u3 * s2
    a13, a23, a33 = a[:, 0, 2], a[:, 1, 2], a[:, 2, 2]
    s3 = a13 + u2 * a23 + u3 * a33
    b13, b23, b33 = a13 - tau1 * s3, a23 - tau1 * u2 * s3, a33 - tau1 * u3 * s3
    beta2, tau2, inv2 = _reflector(b22, np.abs(b32))
    w3 = b32 * inv2
    t3 = b23 + w3 * b33
    r = np.zeros_like(a)
    r[:, 0, 0] = beta1
    r[:, 0, 1] = b12
    r[:, 0, 2] = b13
    r[:, 1, 1] = beta2
    r[:, 1, 2] = b23 - tau2 * t3
    r[:, 2, 2] = b33 - tau2 * w3 * t3
    h12, h13 = -tau1 * u2, -tau1 * u3
    h22, h23 = 1.0 - tau1 * u2 * u2, -tau1 * u2 * u3
    h32, h33 = -tau1 * u3 * u2, 1.0 - tau1 * u3 * u3
    g22, g32, g33 = 1.0 - tau2, -tau2 * w3, 1.0 - tau2 * w3 * w3
    q = np.empty_like(a)
    q[:, 0, 0] = 1.0 - tau1
    q[:, 1, 0] = h12
    q[:, 2, 0] = h13
    q[:, 0, 1] = h12 * g22 + h13 * g32
    q[:, 1, 1] = h22 * g22 + h23 * g32
    q[:, 2, 1] = h32 * g22 + h33 * g32
    q[:, 0, 2] = h12 * g32 + h13 * g33
    q[:, 1, 2] = h22 * g32 + h23 * g33
    q[:, 2, 2] = h32 * g32 + h33 * g33
    return q, r


NS = [1_000, 10_000] if SMOKE else [300, 1_000, 10_000, 100_000, 1_000_000]

for d, fn in ((2, qr2_householder), (3, qr3_householder)):
    for n in NS:
        a = np.ascontiguousarray(rng.standard_normal((n, d, d)))
        suite.measure(
            case=f"qr_{d}x{d}_n{n}",
            params={"d": d, "n": n},
            baseline=(
                "numpy.linalg.qr",
                lambda a=a: (lambda res: (res.Q, res.R))(np.linalg.qr(a)),
            ),
            candidates={"householder": lambda a=a, fn=fn: fn(a)},
            check=qr_close(1e-9),
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
