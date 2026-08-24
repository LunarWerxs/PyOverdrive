"""OPP-000056: the singular-value family on 2x2/3x3 batches.

numpy.linalg.svd(compute_uv=False), norm(ord=2) and pinv each route every
matrix in a stack through their own LAPACK call, and at d in {2, 3} that
per-matrix dispatch dominates. The singular values of A are the square
roots of the eigenvalues of the small symmetric gram matrix A^T A, which
has a closed form (the quadratic for d=2, the trigonometric solution for
d=3), and for a well-conditioned square matrix pinv(A) is just inv(A),
which the adjugate gives directly.

The gram is built ENTRYWISE here, never with a batched matmul - the
difference between a 4x measurement and a 20x one, and the reason this
reproducer builds it the long way.

Accuracy is judged the way LAPACK guarantees singular values: absolute
error against ||A|| (the largest singular value). Relative accuracy on
the smallest value is not claimed - forming the gram squares the
condition number and costs half those digits.

House rules: never imports pyoverdrive. This reproduces the raw margin;
the shipped path adds conditioning bands and a split-to-stock arm.
Result JSON: benchmarks/results/OPP-000056/.
Run: .venv/Scripts/python benchmarks/historical/opp_000056_svd_small_batch.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000056", "svd/norm2/pinv on 2x2-3x3 batches via the gram closed form")
rng = np.random.default_rng(56)


def sv_close(rtol):
    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        return c.shape == b.shape and bool(
            np.all(np.abs(c - b) <= rtol * np.maximum(b[..., :1], 1e-300))
        )

    return _chk


def rel_close(rtol):
    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        return c.shape == b.shape and bool(
            np.all(np.abs(c - b) <= rtol * max(float(np.abs(b).max()), 1e-300))
        )

    return _chk


def _sv2(a):
    a00 = a[..., 0, 0]; a01 = a[..., 0, 1]
    a10 = a[..., 1, 0]; a11 = a[..., 1, 1]
    g00 = a00 * a00 + a10 * a10
    g11 = a01 * a01 + a11 * a11
    g01 = a00 * a01 + a10 * a11
    t = g00 + g11
    disc = np.sqrt(np.maximum(t * t - 4.0 * (g00 * g11 - g01 * g01), 0.0))
    return (t + disc) * 0.5, np.maximum((t - disc) * 0.5, 0.0)


def _sv3(a):
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
    detb = (b0 * (b1 * b2 - c12 * c12) - c01 * (c01 * b2 - c12 * c02)
            + c02 * (c01 * c12 - b1 * c02))
    r = np.clip(detb * 0.5, -1.0, 1.0)
    phi = np.arccos(r) / 3.0
    hi = q + 2.0 * p * np.cos(phi)
    lo = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    mid = 3.0 * q - hi - lo
    return np.maximum(hi, 0.0), np.maximum(mid, 0.0), np.maximum(lo, 0.0)


def svdvals(a):
    if a.shape[-1] == 2:
        hi, lo = _sv2(a)
        return np.sqrt(np.stack([hi, lo], axis=-1))
    hi, mid, lo = _sv3(a)
    return np.sqrt(np.stack([hi, mid, lo], axis=-1))


def norm2(a):
    return np.sqrt(_sv2(a)[0] if a.shape[-1] == 2 else _sv3(a)[0])


def adjugate_inverse(a):
    out = np.empty_like(a)
    if a.shape[-1] == 2:
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


NS = [1_000, 10_000] if SMOKE else [100, 300, 1_000, 10_000, 100_000]

for d in (2, 3):
    for n in NS:
        a = np.ascontiguousarray(rng.standard_normal((n, d, d)))
        suite.measure(
            case=f"pinv_{d}x{d}_n{n}",
            params={"op": "pinv", "d": d, "n": n},
            baseline=("numpy.linalg.pinv", lambda a=a: np.linalg.pinv(a)),
            candidates={"adjugate": lambda a=a: adjugate_inverse(a)},
            check=rel_close(1e-9),
            samples=SAMPLES,
        )
        suite.measure(
            case=f"norm2_{d}x{d}_n{n}",
            params={"op": "norm2", "d": d, "n": n},
            baseline=(
                "numpy.linalg.norm",
                lambda a=a: np.linalg.norm(a, ord=2, axis=(-2, -1)),
            ),
            candidates={"gram_sigma_max": lambda a=a: norm2(a)},
            check=rel_close(1e-12),
            samples=SAMPLES,
        )
        suite.measure(
            case=f"svdvals_{d}x{d}_n{n}",
            params={"op": "svdvals", "d": d, "n": n},
            baseline=("numpy.linalg.svd", lambda a=a: np.linalg.svd(a, compute_uv=False)),
            candidates={"gram_closed_form": lambda a=a: svdvals(a)},
            check=sv_close(1e-9),
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
