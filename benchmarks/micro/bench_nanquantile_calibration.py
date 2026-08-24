"""nanquantile_masked calibration: shape-regime sweep for the dispatch guard.

The OPP-000013 reproducer measured the WIN regime (many small slices:
53-102x) and one middling case ((500, 500): 5.4-9.9x). What the dispatch
predicate needs is the other side: where the win dies. Per cakedev0's
comment on the issue, stock's per-slice Python overhead only dominates
when the number of reductions is large relative to the reduced length, so
the suspect regime is FEW, LONG slices (e.g. (100000, 5) reduced along
axis 0 is five Python-loop iterations for stock, but a 5e5-element sort
for the candidate). This battery sweeps from tiny inputs through the
reproducer's shapes into that anti-regime, at 0% NaN (the
skip-to-quantile route) and 10% NaN (the masked route), and feeds
SIZE_FLOOR plus any slice-regime guard in
src/pyoverdrive/fastpaths/nanquantile_masked.py.

The candidate IS the shipped implementation (imported from the module), so
what this measures is what dispatch executes.

Check: exact equality including NaN pattern (the probe suite measured 0.0
worst-case difference across 140 cases; anything above ULP scale here is a
bug, not tolerance).

Result JSON: benchmarks/results/FASTNANQ-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_nanquantile_calibration.py [--smoke]
"""

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from lab.dyno import BenchSuite
from pyoverdrive.fastpaths.nanquantile_masked import _run

SEED = 16575
SMOKE = "--smoke" in sys.argv
Q = 0.8

# (shape, axis): reduced length is shape[axis], slice count is the rest
if SMOKE:
    CASES = [((20, 30), 0)]
    SAMPLES = 3
else:
    CASES = [
        ((10, 30), 0),        # 300 elements, tiny
        ((20, 50), 0),        # 1_000
        ((27, 100), 0),       # 2_700, the reporter's shape
        ((5, 500), 0),        # 2_500, many minimal slices
        ((100, 27), 0),       # 2_700 transposed regime: 27 slices of 100
        ((100, 1000), 0),     # 1e5, many small slices
        ((500, 500), 0),      # middling (reproducer: 5.4-9.9x)
        ((50, 100, 100), 0),  # reproducer's best case
        ((10000, 3), 0),      # ANTI-REGIME: 3 slices of 10_000
        ((100000, 5), 0),     # ANTI-REGIME: 5 slices of 100_000
        ((3, 10000), 1),      # anti-regime via a different axis
    ]
    SAMPLES = 7


def exact(c, b):
    return c.shape == b.shape and c.dtype == b.dtype and bool(
        np.array_equal(c, b, equal_nan=True)
    )


suite = BenchSuite("FASTNANQ-CAL", "vectorized masked nanquantile vs stock, shape-regime sweep")
rng = np.random.default_rng(SEED)

warnings.simplefilter("ignore", RuntimeWarning)

for shape, axis in CASES:
    n_red = shape[axis]
    n_slices = int(np.prod(shape)) // n_red
    for nan_frac in (0.0, 0.1):
        a = rng.uniform(-5.0, 5.0, size=shape)
        if nan_frac:
            a[rng.random(shape) < nan_frac] = np.nan
        label = "x".join(str(d) for d in shape)
        samples = SAMPLES if a.size <= 500_000 else max(3, SAMPLES - 3)
        suite.measure(
            case=f"nanq_{label}_ax{axis}_nan{int(nan_frac * 100)}",
            params={"shape": list(shape), "axis": axis, "q": Q,
                    "nan_frac": nan_frac, "reduced_len": n_red,
                    "n_slices": n_slices, "elements": a.size},
            baseline=("numpy.nanquantile", lambda a=a, ax=axis: np.nanquantile(a, Q, axis=ax)),
            candidates={"masked_vectorized": lambda a=a, ax=axis: _run(a, Q, ax)},
            check=exact,
            samples=samples,
        )

if not SMOKE:
    suite.save()
