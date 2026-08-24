"""einsum_optimize calibration: contraction-pattern x size sweep for the floor.

The OPP-000018 reproducer measured two shapes of one contraction
('thd,Thd->thT') plus dgasmith's tiny-inner counter-case. The shipped path
routes DEFAULT (optimize=False) two-operand einsum calls through numpy's
own optimize=True machinery, so what the dispatch predicate needs is the
size floor per contraction FAMILY: optimize=True pays einsum_path planning
plus tensordot reshapes on every call, which is catastrophic on tiny
contractions (0.15x at len-3 inner, the reason maintainers keep
optimize=False as default) and huge on large batched ones (17.6-30.5x).
This battery sweeps representative two-operand patterns across operand
sizes and feeds SIZE_FLOOR in src/pyoverdrive/fastpaths/einsum_optimize.py.

Candidate = stock np.einsum(..., optimize=True): numpy's own machinery,
same semantics, different summation order (hence numeric check, not
bit-identical).

Result JSON: benchmarks/results/EINSUM-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_einsum_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 22604
SMOKE = "--smoke" in sys.argv


def close(c, b):
    # Tolerance is ABSOLUTE-scaled, not per-element relative: outputs of a
    # +/-1-uniform contraction include near-zero elements where float32
    # accumulation error dominates the value itself. Measured on this box:
    # at 'thd' 300x1x300 float32, DEFAULT einsum is 5.11x relative off the
    # float64 truth at such elements while optimize=True is 0.65x off, i.e.
    # both routes carry O(eps32 * L) absolute error and optimize is the
    # closer one. A per-element rtol check therefore fails correct results;
    # an absolute bound scaled to the output magnitude still fails any
    # genuinely wrong contraction (which is off by O(scale), not O(eps)).
    c = np.asarray(c)
    b = np.asarray(b)
    if c.shape != b.shape:
        return False
    scale = max(1.0, float(np.abs(b).max())) if b.size else 1.0
    if b.dtype == np.float32:
        return bool(np.allclose(c, b, rtol=1e-3, atol=1e-4 * scale))
    return bool(np.allclose(c, b, rtol=1e-6, atol=1e-9 * scale))


# (case label, subscripts, shape_a, shape_b): every pattern is a genuine
# two-operand contraction a user would hit; sizes bracket the crossover
if SMOKE:
    CASES = [("inner", "i,i->", (1000,), (1000,))]
    SAMPLES = 3
else:
    CASES = [
        # dgasmith's counter-case: tiny inner products MUST stay on stock
        ("inner", "i,i->", (3,), (3,)),
        ("inner", "i,i->", (100,), (100,)),
        ("inner", "i,i->", (10_000,), (10_000,)),
        ("inner", "i,i->", (1_000_000,), (1_000_000,)),
        # plain matmul-shaped
        ("matmul", "ij,jk->ik", (30, 30), (30, 30)),
        ("matmul", "ij,jk->ik", (100, 100), (100, 100)),
        ("matmul", "ij,jk->ik", (300, 300), (300, 300)),
        ("matmul", "ij,jk->ik", (1000, 1000), (1000, 1000)),
        # the reported batched contraction, original bug-report shape last
        ("thd", "thd,Thd->thT", (30, 1, 50), (30, 1, 50)),
        ("thd", "thd,Thd->thT", (100, 1, 100), (100, 1, 100)),
        ("thd", "thd,Thd->thT", (300, 1, 300), (300, 1, 300)),
        ("thd", "thd,Thd->thT", (1000, 1, 500), (1000, 1, 500)),
        # true batched matmul
        ("bmm", "ijk,ikl->ijl", (8, 20, 20), (8, 20, 20)),
        ("bmm", "ijk,ikl->ijl", (32, 64, 64), (32, 64, 64)),
        ("bmm", "ijk,ikl->ijl", (64, 128, 128), (64, 128, 128)),
        # Frobenius-style full contraction
        ("frob", "ij,ij->", (100, 100), (100, 100)),
        ("frob", "ij,ij->", (1000, 1000), (1000, 1000)),
    ]
    SAMPLES = 7

suite = BenchSuite("EINSUM-CAL", "np.einsum default vs optimize=True, pattern x size sweep")
rng = np.random.default_rng(SEED)

for dtype in ((np.float64,) if SMOKE else (np.float64, np.float32)):
    dt = np.dtype(dtype).name
    for label, subs, sa, sb in CASES:
        a = rng.uniform(-1.0, 1.0, size=sa).astype(dtype)
        b = rng.uniform(-1.0, 1.0, size=sb).astype(dtype)
        elements = min(a.size, b.size)
        samples = SAMPLES if max(a.size, b.size) <= 500_000 else max(3, SAMPLES - 3)
        suite.measure(
            case=f"{label}_{dt}_min{elements}",
            params={"pattern": subs, "dtype": dt, "shape_a": list(sa),
                    "shape_b": list(sb), "min_elements": elements},
            baseline=("numpy.einsum", lambda s=subs, a=a, b=b: np.einsum(s, a, b)),
            candidates={
                "optimize_true": lambda s=subs, a=a, b=b: np.einsum(s, a, b, optimize=True),
            },
            check=close,
            samples=samples,
        )

if not SMOKE:
    suite.save()
