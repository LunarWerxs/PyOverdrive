"""intersect_sorted calibration: the SHIPPED path vs stock np.intersect1d.

Sweeps combined size from 32 (where the two sortedness checks plus dispatch
must not lose to a microsecond stock call) to 1M x 100k, for int64 and int32,
in three input regimes:

  sorted   : both inputs strictly increasing (the issue's regime; no sort)
  random   : both inputs random with duplicates (dedup sort on both)
  mixed    : large reference sorted, small query random

Check is bit-identity. Result JSON: benchmarks/results/INTERSECT-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_intersect_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite
from pyoverdrive.fastpaths.intersect_sorted import _intersect_sorted

SMOKE = "--smoke" in sys.argv

if SMOKE:
    SIZE_PAIRS = [(100, 10), (1_000, 100)]
    DTYPES = [np.int64]
    SAMPLES = 3
else:
    SIZE_PAIRS = [
        (16, 16),
        (32, 32),
        (50, 14),
        (100, 100),
        (200, 200),
        (300, 300),
        (400, 100),
        (500, 500),
        (800, 200),
        (1_000, 100),
        (1_000, 1_000),
        (10_000, 1_000),
        (10_000, 10_000),
        (100_000, 10_000),
        (100_000, 100_000),
        (1_000_000, 10_000),
        (1_000_000, 100_000),
    ]
    DTYPES = [np.int64, np.int32]
    SAMPLES = 9

rng = np.random.default_rng(27042)


def sorted_unique(n, universe, dtype):
    n = min(n, universe)
    return np.sort(rng.choice(universe, size=n, replace=False)).astype(dtype)


def random_dups(n, universe, dtype):
    return rng.integers(0, universe, size=n).astype(dtype)


def bit_identical(c, b):
    return c.dtype == b.dtype and c.shape == b.shape and np.array_equal(c, b)


suite = BenchSuite("INTERSECT-CAL", "intersect_sorted fast path vs stock np.intersect1d")

for dtype in DTYPES:
    dt = np.dtype(dtype).name
    for na, nb in SIZE_PAIRS:
        universe = max(na, nb) * 3
        regimes = {
            "sorted": (sorted_unique(na, universe, dtype), sorted_unique(nb, universe, dtype)),
            "random": (random_dups(na, universe, dtype), random_dups(nb, universe, dtype)),
            "mixed": (sorted_unique(na, universe, dtype), random_dups(nb, universe, dtype)),
        }
        for regime, (a, b) in regimes.items():
            samples = SAMPLES if na < 1_000_000 else max(3, SAMPLES - 4)
            suite.measure(
                case=f"intersect1d_{dt}_{regime}_n{na}_m{nb}",
                params={"dtype": dt, "regime": regime, "len_a": na, "len_b": nb},
                baseline=("numpy.intersect1d", lambda a=a, b=b: np.intersect1d(a, b)),
                candidates={"intersect_sorted": lambda a=a, b=b: _intersect_sorted(a, b)},
                check=bit_identical,
                samples=samples,
            )

if not SMOKE:
    suite.save()
