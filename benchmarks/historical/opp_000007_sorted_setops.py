"""OPP-000007: sorted-set operations (intersect1d, isin) on sorted, unique
int64 arrays.

numpy/numpy#27042 claims a 10x-1000x speedup is available for set operations
when both inputs are already sorted and unique, by using a merge / binary
search style algorithm (as sortednp does) instead of numpy's general
concatenate-and-sort path. This reproducer measures:

  - baseline: np.intersect1d(a, b) / np.isin(a, b) with no hints
  - candidate "assume_unique": same calls with assume_unique=True, numpy's
    existing (smaller) optimization that skips the internal dedup pass
  - candidate "searchsorted": a hand-rolled O(n+m)-ish fast path built on
    np.searchsorted, standing in for the sortednp-style algorithm the issue
    asks for

Both a and b are generated sorted and unique, drawn from a shared universe so
intersections are non-trivial (not always empty, not always a full subset).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 27042
SMOKE = "--smoke" in sys.argv


def make_sorted_unique(rng, n, universe_size):
    n = min(n, universe_size)
    vals = rng.choice(universe_size, size=n, replace=False)
    return np.sort(vals).astype(np.int64)


def searchsorted_intersect(sorted_ref, query):
    """Intersection of two sorted, unique 1-D arrays via searchsorted.

    Finds, for every element of `query`, where it would sit in `sorted_ref`;
    an element is in the intersection iff that slot actually holds it. Edges
    are guarded by clipping the insertion index into range before indexing,
    then relying on the equality check to reject an out-of-range match
    (an element past the end of sorted_ref clips to the last slot, which by
    construction cannot equal an element greater than sorted_ref's max).
    """
    if len(sorted_ref) == 0 or len(query) == 0:
        return np.empty(0, dtype=query.dtype)
    idx = np.searchsorted(sorted_ref, query)
    idx = np.clip(idx, 0, len(sorted_ref) - 1)
    mask = sorted_ref[idx] == query
    return query[mask]


def searchsorted_isin(elements, test_elements):
    """np.isin(elements, test_elements) equivalent via searchsorted, for
    sorted unique inputs. Same edge-guard reasoning as searchsorted_intersect."""
    if len(test_elements) == 0:
        return np.zeros(len(elements), dtype=bool)
    idx = np.searchsorted(test_elements, elements)
    idx = np.clip(idx, 0, len(test_elements) - 1)
    return test_elements[idx] == elements


if SMOKE:
    size_pairs = [(100, 10), (1_000, 100)]
else:
    size_pairs = [
        (1_000, 100),
        (1_000, 1_000),
        (10_000, 1_000),
        (10_000, 10_000),
        (100_000, 10_000),
        (100_000, 100_000),
        (1_000_000, 10_000),
        (1_000_000, 100_000),
    ]

suite = BenchSuite("OPP-000007", "sorted-set ops: intersect1d/isin vs searchsorted fast path")
samples = 3 if SMOKE else 11

rng = np.random.default_rng(SEED)

for a_size, b_size in size_pairs:
    universe = max(a_size, b_size) * 3
    a = make_sorted_unique(rng, a_size, universe)
    b = make_sorted_unique(rng, b_size, universe)
    case = f"n{len(a)}_m{len(b)}"
    params = {"dtype": "int64", "len_a": len(a), "len_b": len(b), "universe": universe}

    suite.measure(
        case=f"intersect1d_{case}",
        params=params,
        baseline=("numpy.intersect1d", lambda a=a, b=b: np.intersect1d(a, b)),
        candidates={
            "assume_unique": lambda a=a, b=b: np.intersect1d(a, b, assume_unique=True),
            "searchsorted": lambda a=a, b=b: searchsorted_intersect(a, b),
        },
        check=np.array_equal,
        samples=samples,
    )

    suite.measure(
        case=f"isin_{case}",
        params=params,
        baseline=("numpy.isin", lambda a=a, b=b: np.isin(a, b)),
        candidates={
            "assume_unique": lambda a=a, b=b: np.isin(a, b, assume_unique=True),
            "searchsorted": lambda a=a, b=b: searchsorted_isin(a, b),
        },
        check=np.array_equal,
        samples=samples,
    )

if not SMOKE:
    suite.save()
