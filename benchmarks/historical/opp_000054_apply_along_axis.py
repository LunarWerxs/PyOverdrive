"""OPP-000054: apply_along_axis with a known reducer vs the axis= form.

numpy.apply_along_axis is a documented convenience wrapper whose body is
a Python loop over an ndindex of the non-axis dimensions, calling func1d
once per 1-D slice. When func1d IS one of NumPy's own reductions, that
loop reproduces one slice at a time what the reduction's axis= argument
does in a single vectorized call.

The cells sweep SLICE COUNT at fixed slice length, because that - not
the element count - is what the Python loop charges for. Bit-identical
comparison (last axis; see the module docstring for why non-last axes
are served only for order-independent reducers).

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000054/.
Run: .venv/Scripts/python benchmarks/historical/opp_000054_apply_along_axis.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000054", "apply_along_axis known-reducer interception")
rng = np.random.default_rng(54)


def exact(c, b):
    c = np.asarray(c)
    b = np.asarray(b)
    return (
        c.shape == b.shape
        and c.dtype == b.dtype
        and bool(np.array_equal(c, b, equal_nan=True))
    )


SLICES = [64, 2_000] if SMOKE else [16, 64, 256, 1_000, 5_000, 20_000]
REDUCERS = ("mean", "sum", "max", "median") if not SMOKE else ("mean", "sum")

for nslices in SLICES:
    a = rng.standard_normal((nslices, 50))
    for name in REDUCERS:
        f = getattr(np, name)
        suite.measure(
            case=f"{name}_slices{nslices}",
            params={"reducer": name, "slices": nslices, "slice_len": 50},
            baseline=(
                "numpy.apply_along_axis",
                lambda a=a, f=f: np.apply_along_axis(f, -1, a),
            ),
            candidates={"axis_reduction": lambda a=a, f=f: f(a, axis=-1)},
            check=exact,
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
