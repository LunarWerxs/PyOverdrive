"""OPP-000051: numpy.take with out= via fancy-index assignment.

numpy/numpy#28636 - np.take runs measurably SLOWER when out= is provided
than without it (the C implementation takes a general clip/wrap-capable
path once an output buffer is involved), while the equivalent
out[...] = a[indices] runs the plain fancy-index gather plus one copy and
wins at every measured size. Bit-identical comparison.

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000051/.
Run: .venv/Scripts/python benchmarks/historical/opp_000051_take_out.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000051", "numpy.take out= via index assignment")
rng = np.random.default_rng(28636)


def exact():
    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        return c.shape == b.shape and c.dtype == b.dtype and bool(np.array_equal(c, b))

    return _chk


def take_index_assign(a, indices, out):
    out[...] = a[indices]
    return out


def build(rng, n):
    a = rng.standard_normal(n).astype(np.float64)
    idx = rng.integers(0, n, size=n).astype(np.intp)
    out_stock = np.empty(n, dtype=np.float64)
    out_candidate = np.empty(n, dtype=np.float64)
    return a, idx, out_stock, out_candidate


SIZES = [10_000] if SMOKE else [10_000, 1_000_000, 10_000_000]

for n in SIZES:
    a, idx, out_stock, out_candidate = build(rng, n)
    suite.measure(
        case=f"take_out_n{n}",
        params={"n": n},
        baseline=("take_out", lambda a=a, idx=idx, out=out_stock: np.take(a, idx, out=out)),
        candidates={
            "index_assign": lambda a=a, idx=idx, out=out_candidate: take_index_assign(
                a, idx, out
            )
        },
        check=exact(),
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
