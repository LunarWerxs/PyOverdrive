"""searchsorted_sortqueries calibration: the gaps the reproducer left.

The OPP-000015 reproducer (benchmarks/results/OPP-000015/) already swept
equal sizes and haystack/query ratios: float64 wins from len(y) = 1e4
(2.18-2.56x, 7.37x at the reporter's 5e6), int64 from 1e5 (1.51x, 14.5x at
1e6), losses only below those floors. What it did NOT measure is what the
dispatch predicate exposes:

- ALREADY-SORTED queries: the fast path pays an argsort + two permutations
  for nothing (stock is already fast on sorted queries - that is the whole
  finding). How bad is dispatching there?
- side='right': same mechanism by symmetry (each element's insertion index
  is independent of query order), but unmeasured.
- a TINY haystack under a huge query array: per-query binary search is
  already short (log2 of a small x), so the locality win may not repay the
  sort.
- nearly-sorted queries (sorted plus light shuffling), between the two
  extremes.

Feeds src/pyoverdrive/fastpaths/searchsorted_sortqueries.py.

Check: bit-identical (insertion indices are per-element deterministic, so
the unpermuted result must equal stock exactly).

Result JSON: benchmarks/results/SEARCHSORTED-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_searchsorted_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 10937
SMOKE = "--smoke" in sys.argv


def argsort_unpermute(x, y, side="left"):
    perm = np.argsort(y)
    idx = np.searchsorted(x, y[perm], side=side)
    out = np.empty_like(idx)
    out[perm] = idx
    return out


def exact(c, b):
    return c.dtype == b.dtype and c.shape == b.shape and bool(np.array_equal(c, b))


suite = BenchSuite("SEARCHSORTED-CAL", "argsort+unpermute vs stock: sortedness, side, tiny haystack")
rng = np.random.default_rng(SEED)

if SMOKE:
    N = 2_000
else:
    N = 1_000_000

x = np.sort(rng.standard_normal(N))
y_random = rng.standard_normal(N)
y_sorted = np.sort(rng.standard_normal(N))
y_nearly = np.sort(rng.standard_normal(N))
swap = rng.integers(0, N - 1, size=max(1, N // 100))
y_nearly[swap], y_nearly[swap + 1] = y_nearly[swap + 1], y_nearly[swap]
x_tiny = np.sort(rng.standard_normal(64))

def shuffled_fraction(base, frac, rng):
    """Sorted queries with `frac` of positions swapped with a random partner:
    tunes the locality stock's binary search enjoys, to place the disorder
    threshold the dispatch gate needs."""
    y = np.sort(base.copy())
    k = max(1, int(base.size * frac))
    i = rng.integers(0, base.size, size=k)
    j = rng.integers(0, base.size, size=k)
    y[i], y[j] = y[j], y[i]
    return y


x_1e3 = np.sort(rng.standard_normal(1_000))
x_1e4 = np.sort(rng.standard_normal(10_000))
CASES = [
    ("random_baseline_regime", x, y_random, "left"),
    ("already_sorted_queries", x, y_sorted, "left"),
    ("nearly_sorted_queries", x, y_nearly, "left"),
    ("shuffled05", x, shuffled_fraction(y_random, 0.05, rng), "left"),
    ("shuffled10", x, shuffled_fraction(y_random, 0.10, rng), "left"),
    ("shuffled25", x, shuffled_fraction(y_random, 0.25, rng), "left"),
    ("descending_queries", x, np.sort(y_random)[::-1].copy(), "left"),
    ("side_right_random", x, y_random, "right"),
    ("tiny_haystack_x64", x_tiny, y_random, "left"),
    ("haystack_x1000", x_1e3, y_random, "left"),
    ("haystack_x10000", x_1e4, y_random, "left"),
]

samples = 3 if SMOKE else 7
for label, xs, ys, side in CASES:
    suite.measure(
        case=f"{label}_n{ys.size}",
        params={"len_x": xs.size, "len_y": ys.size, "side": side, "dtype": "float64"},
        baseline=("numpy.searchsorted", lambda xs=xs, ys=ys, side=side: np.searchsorted(xs, ys, side=side)),
        candidates={"argsort_unpermute": lambda xs=xs, ys=ys, side=side: argsort_unpermute(xs, ys, side)},
        check=exact,
        samples=samples,
    )

if not SMOKE:
    suite.save()
