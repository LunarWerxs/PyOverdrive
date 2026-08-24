"""OPP-000045: batched 2x2/3x3 det/slogdet/solve vs closed forms.

numpy/numpy#20052 documents slogdet's overhead; the mechanism is the
per-matrix LAPACK dispatch the SHIPPED inv_small_batch (OPP-000035,
numpy#17166) already exploits. Cofactor-expansion det, sign/log slogdet,
and Cramer solve vectorize across the stack. Numeric comparison
(rtol 1e-9 det / 1e-8 solve; slogdet signs exactly equal).

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000045/.
Run: .venv/Scripts/python benchmarks/historical/opp_000045_small_batch_linalg.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000045", "small-batch det/slogdet/solve closed forms")
rng = np.random.default_rng(20052)


def close(rtol):
    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        return c.shape == b.shape and c.dtype == b.dtype and bool(
            np.allclose(c, b, rtol=rtol, atol=0.0, equal_nan=True)
        )

    return _chk


def slogdet_check(c, b):
    return np.array_equal(c[0], b[0]) and bool(
        np.allclose(c[1], b[1], rtol=1e-9, atol=1e-12, equal_nan=True)
    )


def det_closed(m):
    d = m.shape[-1]
    if d == 2:
        return m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] * m[..., 1, 0]
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    dd, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    return a * (e * i - f * h) - b * (dd * i - f * g) + c * (dd * h - e * g)


def slogdet_closed(m):
    det = det_closed(m)
    return np.sign(det), np.log(np.abs(det))


def cramer(m, b):
    d = m.shape[-1]
    out = np.empty_like(b)
    b1, b2 = b[..., 0, 0], b[..., 1, 0]
    if d == 2:
        a11, a12 = m[..., 0, 0], m[..., 0, 1]
        a21, a22 = m[..., 1, 0], m[..., 1, 1]
        inv_det = 1.0 / (a11 * a22 - a12 * a21)
        out[..., 0, 0] = (b1 * a22 - b2 * a12) * inv_det
        out[..., 1, 0] = (a11 * b2 - a21 * b1) * inv_det
        return out
    a11, a12, a13 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    a21, a22, a23 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    a31, a32, a33 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    b3 = b[..., 2, 0]
    c11 = a22 * a33 - a23 * a32
    c12 = a23 * a31 - a21 * a33
    c13 = a21 * a32 - a22 * a31
    inv_det = 1.0 / (a11 * c11 + a12 * c12 + a13 * c13)
    out[..., 0, 0] = (b1 * c11 + b2 * (a13 * a32 - a12 * a33) + b3 * (a12 * a23 - a13 * a22)) * inv_det
    out[..., 1, 0] = (b1 * c12 + b2 * (a11 * a33 - a13 * a31) + b3 * (a13 * a21 - a11 * a23)) * inv_det
    out[..., 2, 0] = (b1 * c13 + b2 * (a12 * a31 - a11 * a32) + b3 * (a11 * a22 - a12 * a21)) * inv_det
    return out


BATCHES = [1_000] if SMOKE else [100, 300, 1_000, 5_000, 20_000]

for d in (2, 3):
    for nb in BATCHES:
        m = rng.standard_normal((nb, d, d))
        b = rng.standard_normal((nb, d, 1))
        suite.measure(
            case=f"det_{d}x{d}_batch{nb}",
            params={"d": d, "batch": nb},
            baseline=("linalg.det", lambda m=m: np.linalg.det(m)),
            candidates={"closed_form": lambda m=m: det_closed(m)},
            check=close(1e-9),
            samples=SAMPLES,
        )
        suite.measure(
            case=f"slogdet_{d}x{d}_batch{nb}",
            params={"d": d, "batch": nb},
            baseline=("linalg.slogdet", lambda m=m: np.linalg.slogdet(m)),
            candidates={"closed_form": lambda m=m: slogdet_closed(m)},
            check=slogdet_check,
            samples=SAMPLES,
        )
        suite.measure(
            case=f"solve_{d}x{d}_batch{nb}",
            params={"d": d, "batch": nb},
            baseline=("linalg.solve", lambda m=m, b=b: np.linalg.solve(m, b)),
            candidates={"cramer": lambda m=m, b=b: cramer(m, b)},
            check=close(1e-8),
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
