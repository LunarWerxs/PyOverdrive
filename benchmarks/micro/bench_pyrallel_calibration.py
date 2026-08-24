"""PyRallel calibration battery: which ops, dtypes, sizes and thread counts win.

Extends the OPP-000008 evidence (np.sin only, float64 only, 4/8/16 threads)
into the full table the parallel_ufunc fast path is calibrated from:

  ops      : sin cos tan exp log log10 tanh sqrt  (sqrt is the expected
             memory-bound loser, kept as the in-battery negative control)
  dtypes   : float64 float32
  sizes    : 1e4 .. 1e7 at roughly half-decade steps (crossover resolution)
  threads  : 2 4 8 16

The candidate is the SHIPPED mechanism (pyoverdrive.parallel.parallel_unary),
so every number includes PyRallel's own chunking and submit overhead. The
check is bit-identity (array_equal with equal_nan), not allclose: elementwise
kernels have no cross-element data flow and the fast path promises the stock
result exactly, so a platform where chunking changes a bit must FAIL here and
be excluded, not hidden behind a tolerance.

Result JSON: benchmarks/results/PYRALLEL-CAL/<fingerprint>.json
Run: .venv/Scripts/python benchmarks/micro/bench_pyrallel_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite
from pyoverdrive.parallel import parallel_unary

SMOKE = "--smoke" in sys.argv

if SMOKE:
    SIZES = [10_000]
    THREADS = [2]
    OPS = ["sin", "sqrt"]
    DTYPES = [np.float64]
    SAMPLES = 3
else:
    SIZES = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]
    THREADS = [2, 4, 8, 16]
    OPS = ["sin", "cos", "tan", "exp", "log", "log10", "tanh", "sqrt"]
    DTYPES = [np.float64, np.float32]
    SAMPLES = 7

# Input domains chosen so every op is finite everywhere (NaN/inf propagation
# is a differential-test concern, not a timing one).
DOMAINS = {
    "sin": (0.0, 2 * np.pi),
    "cos": (0.0, 2 * np.pi),
    "tan": (-1.5, 1.5),
    "exp": (-5.0, 5.0),
    "log": (0.1, 100.0),
    "log10": (0.1, 100.0),
    "tanh": (-4.0, 4.0),
    "sqrt": (0.0, 100.0),
}


def bit_identical(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


suite = BenchSuite(
    "PYRALLEL-CAL",
    "PyRallel chunked unary ufuncs vs stock: op x dtype x size x threads",
)

for op_name in OPS:
    ufunc = getattr(np, op_name)
    lo, hi = DOMAINS[op_name]
    for dtype in DTYPES:
        for n in SIZES:
            x = np.linspace(lo, hi, n, dtype=dtype)
            suite.measure(
                case=f"{op_name}_{np.dtype(dtype).name}_n{n}",
                params={"op": op_name, "dtype": np.dtype(dtype).name, "n": n},
                baseline=(f"numpy.{op_name}", lambda u=ufunc, x=x: u(x)),
                candidates={
                    f"pyrallel_{t}t": (lambda u=ufunc, x=x, t=t: parallel_unary(u, x, t))
                    for t in THREADS
                },
                check=bit_identical,
                samples=SAMPLES,
            )

if not SMOKE:
    suite.save()
