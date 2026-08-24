"""OPP-000037: np.median vs the partition route, NaN check included.

numpy/numpy#18298 (anntzer, 2021): median() carries so much Python-level
overhead that partition(x, n//2)[n//2] beats it 11.2x at n=11, 11.1x at
n=101, 6.9x at n=1001, converging at large n. seberg's in-thread
correction: median ALSO partitions the -1 element as its NaN check, and
with partition(x, [n//2, -1]) the large-n gap nearly vanishes - so the
honest candidate replicates the NaN check and the win is
OVERHEAD-CLASS, small-to-mid 1-D arrays.

Candidates (both replicate stock's NaN semantics):

  - partition_median: odd n -> partition(x, (k, n-1)), NaN check via
    isnan(p[-1]), return p[k]; even n -> partition(x, (k-1, k, n-1)),
    return the mean of the two middles computed exactly as stock does
    (0.5 * (lo + hi)); NaN present -> return nan exactly like stock.
    Whether stock also emits a RuntimeWarning on NaN is probed by the
    differential battery at ship time, not here.

Cases: the thread's own size ladder 11..1e7 (odd sizes) plus even sizes
{10, 100, 1_000, 10_000} (the even path has different arithmetic), and a
NaN-salted case checked exact (both sides nan).

Correctness: bit-identical is EXPECTED (same partition kernel, same
element reads, same mean arithmetic) and the check demands exact
equality including the scalar type; if stock's mean arithmetic differs
the check fails and the comparison mode for any shipped path becomes
numeric - a finding either way.

House rules: never imports pyoverdrive. The candidate calls np.partition
(unpatched name) and basic arithmetic; a patched dispatch could not
recurse.

Result JSON: benchmarks/results/OPP-000037/.
Run: .venv/Scripts/python benchmarks/historical/opp_000037_median_partition.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 18298
SMOKE = "--smoke" in sys.argv


def partition_median(x):
    n = x.size
    k = n // 2
    if n % 2:
        p = np.partition(x, (k, n - 1))
        if np.isnan(p[-1]):
            return np.float64(np.nan)
        return p[k]
    p = np.partition(x, (k - 1, k, n - 1))
    if np.isnan(p[-1]):
        return np.float64(np.nan)
    return np.float64(0.5 * (p[k - 1] + p[k]))


def exact(cand, base):
    c = np.asarray(cand)
    b = np.asarray(base)
    return c.dtype == b.dtype and bool(np.array_equal(c, b, equal_nan=True))


suite = BenchSuite("OPP-000037", "median via partition (NaN check replicated) vs stock")
rng = np.random.default_rng(SEED)

if SMOKE:
    SIZES = [1_001]
    SAMPLES = 3
else:
    SIZES = [11, 101, 1_001, 10_001, 100_001, 1_000_001, 10_000_001]
    SIZES += [10, 100, 1_000, 10_000]
    SAMPLES = 11

for n in sorted(SIZES):
    x = rng.random(n)
    suite.measure(
        case=f"median_1d_n{n}_{'odd' if n % 2 else 'even'}",
        params={"n": n, "parity": "odd" if n % 2 else "even"},
        baseline=("numpy.median", lambda x=x: np.median(x)),
        candidates={"partition_median": lambda x=x: partition_median(x)},
        check=exact,
        samples=SAMPLES if n <= 1_000_000 else 5,
    )

if not SMOKE:
    x = rng.random(10_001)
    x[rng.integers(0, x.size)] = np.nan
    suite.measure(
        case="median_1d_n10001_nan_salted",
        params={"n": 10_001, "nan": True},
        baseline=("numpy.median", lambda x=x: np.median(x)),
        candidates={"partition_median": lambda x=x: partition_median(x)},
        check=exact,
        samples=9,
    )
    suite.save()
