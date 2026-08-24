"""OPP-000035: stacked small-matrix inverse, vectorized adjugate vs LAPACK loop.

numpy/numpy#17166 (bhaveshshrimali, 2020, numpy 1.19): np.linalg.inv on a
(1000, 4, 3, 3) stack took 79.2 ms where a hand-vectorized adjugate took
371 us (DERIVED 213.5x); thrasibule independently measured 13.4 ms vs
152 us (DERIVED 88.2x) and identified the mechanism: the adjugate route
reorders the loops so every arithmetic op vectorizes across the batch,
while stock runs per-matrix LAPACK calls whose overhead dominates at
d=3. The thread also shows the ANTI-REGIME: for a single 3x3, stock
npinv BEAT the reporter's hdinv on thrasibule's machine (10.7 us vs
20.1 us), so a batch floor is required. Same vein as OPP-000030
(eigvalsh 2x2 closed form, shipped at 31x).

What this reproducer measures:

  1. Batch sweep for (n, 3, 3) float64: n in {1, 100, 1_000, 10_000,
     100_000}, well-conditioned SPD-ish draws (A @ A.T + 0.1 I).
  2. The thread's own shape, flattened: (4000, 3, 3).
  3. A (n, 2, 2) sweep at {1_000, 100_000} (the 2x2 closed form is two
     mults and a swap; same mechanism).
  4. float32 at one batch size.
  5. An ill-conditioned witness (condition ~1e10) at n=10_000 with the
     same scaled check: if the adjugate loses accuracy there, the check
     FAILS and that is a regime finding (condition ceiling), not a
     script bug. LAPACK's LU pivots; the adjugate does not.

Correctness: different algorithm, different rounding: numeric check,
per-matrix scaled - abs(cand - stock) <= rtol * max(1, per-matrix
max |stock element|), rtol 1e-9 f64 / 1e-3 f32 on the well-conditioned
cases. Exactly-singular and non-finite inputs are OUT of regime (stock
raises LinAlgError; the adjugate would silently produce inf/nan): a
shipped predicate must scan finiteness and refuse |det| below a
scale-relative floor, exactly as OPP-000030's path refuses non-finite.

House rules: never imports pyoverdrive; the candidates use only
arithmetic and empty_like, so a patched dispatch could not recurse.

Result JSON: benchmarks/results/OPP-000035/.
Run: .venv/Scripts/python benchmarks/historical/opp_000035_inv_small_batch.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 17166
SMOKE = "--smoke" in sys.argv


def adj_inv_3x3(a):
    """Vectorized adjugate-over-determinant; a is (..., 3, 3)."""
    m00 = a[..., 0, 0]; m01 = a[..., 0, 1]; m02 = a[..., 0, 2]
    m10 = a[..., 1, 0]; m11 = a[..., 1, 1]; m12 = a[..., 1, 2]
    m20 = a[..., 2, 0]; m21 = a[..., 2, 1]; m22 = a[..., 2, 2]
    c00 = m11 * m22 - m12 * m21
    c10 = m12 * m20 - m10 * m22
    c20 = m10 * m21 - m11 * m20
    det = m00 * c00 + m01 * c10 + m02 * c20
    inv_det = 1.0 / det
    out = np.empty_like(a)
    out[..., 0, 0] = c00 * inv_det
    out[..., 0, 1] = (m02 * m21 - m01 * m22) * inv_det
    out[..., 0, 2] = (m01 * m12 - m02 * m11) * inv_det
    out[..., 1, 0] = c10 * inv_det
    out[..., 1, 1] = (m00 * m22 - m02 * m20) * inv_det
    out[..., 1, 2] = (m02 * m10 - m00 * m12) * inv_det
    out[..., 2, 0] = c20 * inv_det
    out[..., 2, 1] = (m01 * m20 - m00 * m21) * inv_det
    out[..., 2, 2] = (m00 * m11 - m01 * m10) * inv_det
    return out


def adj_inv_2x2(a):
    m00 = a[..., 0, 0]; m01 = a[..., 0, 1]
    m10 = a[..., 1, 0]; m11 = a[..., 1, 1]
    inv_det = 1.0 / (m00 * m11 - m01 * m10)
    out = np.empty_like(a)
    out[..., 0, 0] = m11 * inv_det
    out[..., 0, 1] = -m01 * inv_det
    out[..., 1, 0] = -m10 * inv_det
    out[..., 1, 1] = m00 * inv_det
    return out


def make_check(rtol):
    def check(cand, base):
        if cand.dtype != base.dtype or cand.shape != base.shape:
            return False
        scale = np.abs(base).max(axis=(-2, -1), keepdims=True)
        return bool((np.abs(cand - base) <= rtol * np.maximum(1.0, scale)).all())

    return check


CHECK = {"float64": make_check(1e-9), "float32": make_check(1e-3)}

rng = np.random.default_rng(SEED)


def spd_batch(n, d, dtype=np.float64):
    a = rng.uniform(-1.0, 1.0, size=(n, d, d)).astype(dtype)
    return np.ascontiguousarray(
        a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(d, dtype=dtype)
    )


suite = BenchSuite("OPP-000035", "stacked 2x2/3x3 inverse: vectorized adjugate vs LAPACK loop")

BATCHES = [1_000] if SMOKE else [1, 100, 1_000, 10_000, 100_000]
SAMPLES = 3 if SMOKE else 9

for n in BATCHES:
    a = spd_batch(n, 3)
    suite.measure(
        case=f"inv_3x3_batch{n}_float64",
        params={"batch": n, "d": 3, "dtype": "float64"},
        baseline=("numpy.linalg.inv", lambda a=a: np.linalg.inv(a)),
        candidates={"adjugate": lambda a=a: adj_inv_3x3(a)},
        check=CHECK["float64"],
        samples=SAMPLES if n <= 10_000 else max(5, SAMPLES - 2),
    )

if not SMOKE:
    # the thread's own shape, flattened to one batch axis
    a = spd_batch(4_000, 3)
    suite.measure(
        case="inv_3x3_batch4000_reported_shape",
        params={"batch": 4_000, "d": 3, "dtype": "float64", "provenance": "issue shape 1000x4"},
        baseline=("numpy.linalg.inv", lambda a=a: np.linalg.inv(a)),
        candidates={"adjugate": lambda a=a: adj_inv_3x3(a)},
        check=CHECK["float64"],
        samples=9,
    )
    for n in (1_000, 100_000):
        a2 = spd_batch(n, 2)
        suite.measure(
            case=f"inv_2x2_batch{n}_float64",
            params={"batch": n, "d": 2, "dtype": "float64"},
            baseline=("numpy.linalg.inv", lambda a=a2: np.linalg.inv(a2)),
            candidates={"adjugate": lambda a=a2: adj_inv_2x2(a2)},
            check=CHECK["float64"],
            samples=9,
        )
    a32 = spd_batch(10_000, 3, np.float32)
    suite.measure(
        case="inv_3x3_batch10000_float32",
        params={"batch": 10_000, "d": 3, "dtype": "float32"},
        baseline=("numpy.linalg.inv", lambda a=a32: np.linalg.inv(a32)),
        candidates={"adjugate": lambda a=a32: adj_inv_3x3(a32)},
        check=CHECK["float32"],
        samples=9,
    )
    # ill-conditioned witness: eigenvalue ratio ~1e10; a failed check here
    # is a CONDITION-CEILING finding for the predicate, recorded either way
    n = 10_000
    q, _ = np.linalg.qr(rng.uniform(-1.0, 1.0, size=(n, 3, 3)))
    w = np.stack([np.full(n, 1e-10), np.full(n, 1e-2), np.full(n, 1.0)], axis=-1)
    ill = q @ (w[..., None] * np.swapaxes(q, -1, -2))
    ill = np.ascontiguousarray(0.5 * (ill + np.swapaxes(ill, -1, -2)))
    suite.measure(
        case="inv_3x3_batch10000_illconditioned",
        params={"batch": n, "d": 3, "dtype": "float64", "condition": 1e10},
        baseline=("numpy.linalg.inv", lambda a=ill: np.linalg.inv(a)),
        candidates={"adjugate": lambda a=ill: adj_inv_3x3(a)},
        check=CHECK["float64"],
        samples=7,
    )
    suite.save()
