"""OPP-000047: batched 2x2/3x3 Cholesky decomposition, closed form vs LAPACK.

scipy/scipy#24474 documents live demand for a batched small-matrix
decomposition surface: stock np.linalg.cholesky routes each matrix in a
stack through one potrf call, so small-matrix batches pay per-matrix
LAPACK dispatch overhead thousands of times over, the same vein as the
shipped linalg_small_batch (OPP-000045) and inv_small_batch (OPP-000035).
The Cholesky-Crout closed form (explicit formulas for d=2/d=3) vectorizes
across the whole stack and never calls LAPACK. Numeric comparison
(rtol 1e-9, scaled by numpy's own allclose).

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000047/.
Run: .venv/Scripts/python benchmarks/historical/opp_000047_cholesky_small_batch.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000047", "small-batch Cholesky decomposition closed form")
rng = np.random.default_rng(24474)


def close(rtol):
    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        return c.shape == b.shape and c.dtype == b.dtype and bool(
            np.allclose(c, b, rtol=rtol, atol=0.0, equal_nan=True)
        )

    return _chk


def cholesky_closed(a):
    """Cholesky-Crout closed form for d=2/d=3 lower-triangular factors."""
    d = a.shape[-1]
    out = np.zeros(a.shape, dtype=a.dtype)
    a11 = a[..., 0, 0]
    l11 = np.sqrt(a11)
    if d == 2:
        l21 = a[..., 1, 0] / l11
        out[..., 0, 0] = l11
        out[..., 1, 0] = l21
        out[..., 1, 1] = np.sqrt(a[..., 1, 1] - l21 * l21)
        return out
    l21 = a[..., 1, 0] / l11
    l31 = a[..., 2, 0] / l11
    l22 = np.sqrt(a[..., 1, 1] - l21 * l21)
    l32 = (a[..., 2, 1] - l31 * l21) / l22
    out[..., 0, 0] = l11
    out[..., 1, 0] = l21
    out[..., 1, 1] = l22
    out[..., 2, 0] = l31
    out[..., 2, 1] = l32
    out[..., 2, 2] = np.sqrt(a[..., 2, 2] - l31 * l31 - l32 * l32)
    return out


def spd_batch(rng, n, d, dtype=np.float64):
    a = rng.standard_normal((n, d, d)).astype(dtype)
    m = a @ np.swapaxes(a, -1, -2) + d * np.eye(d, dtype=dtype)
    return np.ascontiguousarray(m)


BATCHES = [1_000] if SMOKE else [100, 1_000, 5_000, 20_000]

for d in (2, 3):
    for nb in BATCHES:
        m = spd_batch(rng, nb, d)
        suite.measure(
            case=f"cholesky_{d}x{d}_batch{nb}",
            params={"d": d, "batch": nb},
            baseline=("linalg.cholesky", lambda m=m: np.linalg.cholesky(m)),
            candidates={"closed_form": lambda m=m: cholesky_closed(m)},
            check=close(1e-9),
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
