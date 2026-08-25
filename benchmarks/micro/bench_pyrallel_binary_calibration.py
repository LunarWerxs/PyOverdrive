"""PyRallel BINARY ufunc calibration: memory-bound arithmetic, does it scale?

SUPERSEDED for setting thresholds - use tools/calibrate_dispatch.py --family
binary, for the same reason as the unary battery beside it: on a hybrid CPU
the single-threaded BASELINE this measures against is a per-process coin
flip (P-core or E-core, 1.44x apart, measured over 25 fresh processes),
while the threaded candidate spans cores and averages over both. Every ratio
here is inflated by up to that factor, never deflated, and re-running on an
idle machine does not help: the split is reproducible WITHIN a process and
only visible across them. See docs/research/hybrid-cpu-baseline-coin-flip.md.

Worth stressing for this family in particular: its wins sit within about
0.1x of the 1.3x bar across the sizes where they cross it, and consecutive
sizes in one sweep can read 1.23x, 1.14x, 1.33x. That is non-monotone, so
the run-to-run spread is as large as the margin being measured, and a
threshold read off a single sweep is fitting noise. The shipped table is
derived from two independent sweeps with the WORSE reading kept per cell.

OPP-000008's control case measured np.add at 1e7 float64 gaining 2.4-2.75x
on 4-16 threads, against the issue reporter's prediction of no gain: one Zen
4 core cannot saturate dual-channel DDR5. That is a bandwidth effect, far
more machine dependent than the compute-bound transcendental wins, so it
gets its own table, derived the same way (lab/cli/calibrate_pyrallel.py
--suite PYRALLEL-BIN-CAL).

  ops      : add subtract multiply divide maximum minimum
  dtypes   : float64 float32 int64
  sizes    : 1e5 .. 1e7
  threads  : 2 4 8 16
  shape    : two same-shape C-contiguous operands, no broadcasting

The candidate is the SHIPPED mechanism (parallel_elementwise). Check is
bit-identity. Result JSON: benchmarks/results/PYRALLEL-BIN-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_pyrallel_binary_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite
from pyoverdrive.parallel import parallel_elementwise

SMOKE = "--smoke" in sys.argv

if SMOKE:
    SIZES = [10_000]
    THREADS = [2]
    OPS = ["add", "divide"]
    DTYPES = [np.float64]
    SAMPLES = 3
else:
    SIZES = [100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]
    THREADS = [2, 4, 8, 16]
    OPS = ["add", "subtract", "multiply", "divide", "maximum", "minimum"]
    DTYPES = [np.float64, np.float32, np.int64]
    SAMPLES = 7


def bit_identical(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


suite = BenchSuite(
    "PYRALLEL-BIN-CAL",
    "PyRallel chunked binary ufuncs vs stock: op x dtype x size x threads",
)
rng = np.random.default_rng(8208)

for op_name in OPS:
    ufunc = getattr(np, op_name)
    for dtype in DTYPES:
        if np.dtype(dtype).kind == "i" and op_name == "divide":
            continue  # int / int -> float64; a casting route, not this family
        for n in SIZES:
            if np.dtype(dtype).kind == "f":
                a = rng.random(n).astype(dtype) + 0.5
                b = rng.random(n).astype(dtype) + 0.5
            else:
                a = rng.integers(1, 1_000_000, size=n, dtype=dtype)
                b = rng.integers(1, 1_000_000, size=n, dtype=dtype)
            suite.measure(
                case=f"{op_name}_{np.dtype(dtype).name}_n{n}",
                params={"op": op_name, "dtype": np.dtype(dtype).name, "n": n},
                baseline=(f"numpy.{op_name}", lambda u=ufunc, a=a, b=b: u(a, b)),
                candidates={
                    f"pyrallel_{t}t": (
                        lambda u=ufunc, a=a, b=b, t=t: parallel_elementwise(u, (a, b), t)
                    )
                    for t in THREADS
                },
                check=bit_identical,
                samples=SAMPLES,
            )

if not SMOKE:
    suite.save()
