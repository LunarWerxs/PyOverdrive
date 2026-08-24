"""OPP-000048: batched 3x3 symmetric eigenvalue decomposition, closed form
vs LAPACK.

The d=3 sibling of the shipped eigvalsh_2x2_closed (OPP-000030,
numpy/numpy#22158): batched small-matrix eigh routes each matrix through a
per-matrix LAPACK call whose setup overhead dominates. For real symmetric
3x3 the characteristic cubic has the classical trigonometric solution
(Smith 1961 form): shift by q = tr(A)/3, scale by p = sqrt(tr((A-qI)^2)/6),
then the three roots are q + 2p*cos(phi + 2k*pi/3) with
phi = arccos(det(B)/2)/3 for the scaled deviator B. Vectorizes across the
whole stack; never calls LAPACK. eigvalsh output is mathematically UNIQUE
(no sign/basis freedom outside exact degeneracy), so the check is a
scaled-abs allclose, same standard as the OPP-000030 reproducer.

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000048/.
Run: .venv/Scripts/python benchmarks/historical/opp_000048_eigvalsh_3x3.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 22158
SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 9


def eigvalsh3x3(a):
    """Closed-form ascending eigenvalues, lower triangle (UPLO='L')."""
    a11 = a[..., 0, 0]
    a22 = a[..., 1, 1]
    a33 = a[..., 2, 2]
    a21 = a[..., 1, 0]
    a31 = a[..., 2, 0]
    a32 = a[..., 2, 1]
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
    r = np.clip(detb / 2.0, -1.0, 1.0)
    phi = np.arccos(r) / 3.0
    e_hi = q + 2.0 * p * np.cos(phi)
    e_lo = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    e_mid = 3.0 * q - e_hi - e_lo
    out = np.stack([e_lo, e_mid, e_hi], axis=-1)
    out.sort(axis=-1)
    return out


def sym_batch(rng, n, d=3, dtype=np.float64):
    a = rng.uniform(-1.0, 1.0, size=(n, d, d)).astype(dtype)
    return np.ascontiguousarray(a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(d, dtype=dtype))


def values_close(cand, base):
    if cand.dtype != base.dtype or cand.shape != base.shape:
        return False
    scale = np.maximum(1e-30, np.abs(base).max(axis=-1, keepdims=True))
    return bool((np.abs(cand - base) <= 1e-9 * scale).all())


suite = BenchSuite("OPP-000048", "batched 3x3 eigvalsh: trig closed form vs LAPACK loop")
rng = np.random.default_rng(SEED)

BATCHES = [1_000] if SMOKE else [100, 1_000, 10_000, 100_000]

for n in BATCHES:
    a = sym_batch(rng, n)
    suite.measure(
        case=f"eigvalsh_3x3_batch{n}",
        params={"batch": n, "d": 3, "op": "eigvalsh"},
        baseline=("numpy.linalg.eigvalsh", lambda a=a: np.linalg.eigvalsh(a)),
        candidates={"closed_form_values": lambda a=a: eigvalsh3x3(a)},
        check=values_close,
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
