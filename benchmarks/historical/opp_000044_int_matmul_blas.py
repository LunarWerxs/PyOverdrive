"""OPP-000044: integer matmul vs exact float64 BLAS with the 2^53 bound.

numpy/numpy#14556 (open): no BLAS path exists for integer matmul; the
naive loop is tens of times slower than float64 GEMM at the same size.
When k * max|A| * max|B| < 2^53 (int64; 2^31 for int32, which also
proves stock's int32 accumulator could not have wrapped) the cast ->
BLAS -> cast route is provably bit-exact. Both abs-max scans and both
casts are inside the candidate timing.

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000044/.
Run: .venv/Scripts/python benchmarks/historical/opp_000044_int_matmul_blas.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000044", "int matmul vs exact f64 BLAS round-trip")
rng = np.random.default_rng(14556)


def exact(c, b):
    return c.shape == b.shape and c.dtype == b.dtype and bool(np.array_equal(c, b))


def blas_route(x, y, bound):
    mx = max(-int(np.min(x)), int(np.max(x)))
    my = max(-int(np.min(y)), int(np.max(y)))
    if x.shape[1] * mx * my >= bound:
        return x @ y
    return (x.astype(np.float64) @ y.astype(np.float64)).astype(x.dtype)


SIZES = [50] if SMOKE else [50, 100, 200, 400, 800]

for dt, bound in ((np.int64, 2**53), (np.int32, 2**31)):
    for n in SIZES:
        x = rng.integers(-1000, 1000, (n, n)).astype(dt)
        y = rng.integers(-1000, 1000, (n, n)).astype(dt)
        suite.measure(
            case=f"matmul_{np.dtype(dt).name}_{n}",
            params={"dtype": np.dtype(dt).name, "n": n},
            baseline=("matmul", lambda x=x, y=y: x @ y),
            candidates={"f64_blas": lambda x=x, y=y, b=bound: blas_route(x, y, b)},
            check=exact,
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
