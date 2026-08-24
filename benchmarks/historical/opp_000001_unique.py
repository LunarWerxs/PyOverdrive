"""OPP-000001: np.unique hash-path regression (numpy/numpy#31969).

Claim (2026-07-11, numpy 2.x): the hash-based `_unique_hash` path makes
`np.unique`/`np.unique_values` on large numeric arrays up to 46x slower than
a sort-based approach (712 ms vs 15.2 ms at 1M int64), because
std::unordered_set node allocation + cache misses lose badly to the
SIMD-accelerated sort (x86-simd-sort) on AVX-capable CPUs. Even at heavy
duplication (10 uniques x 100k repeats) sorting was reported 3x faster.

Candidates:
- sort_unique: np.sort + adjacent-difference mask (NaN-unsafe on purpose, to
  let Dyno's correctness check document the failure mode)
- sort_unique_nansafe: same, with the NaN run collapsed the way np.unique does

Both baselines are measured because np.unique and np.unique_values take
different paths: np.unique_values is documented to return unsorted output on
the hash path, so its check compares sorted sets.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SEED = 31969


def sort_unique(a):
    s = np.sort(a)
    if s.size == 0:
        return s
    mask = np.empty(s.size, dtype=bool)
    mask[0] = True
    np.not_equal(s[1:], s[:-1], out=mask[1:])
    return s[mask]


def sort_unique_nansafe(a):
    s = np.sort(a)
    if s.size == 0:
        return s
    mask = np.empty(s.size, dtype=bool)
    mask[0] = True
    np.not_equal(s[1:], s[:-1], out=mask[1:])
    if s.dtype.kind == "f" and np.isnan(s[-1]):
        first_nan = np.searchsorted(s, np.nan, side="left")
        mask[first_nan + 1 :] = False
        mask[first_nan] = True
    return s[mask]


def sorted_set_equal(cand, base):
    return np.array_equal(np.sort(cand), np.sort(base))


def make_data(rng, dtype, n, cardinality):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if cardinality == "high":
            return rng.integers(info.min, info.max, size=n, dtype=dtype)
        return rng.integers(0, int(cardinality), size=n, dtype=dtype)
    if cardinality == "high":
        return rng.random(n).astype(dtype)
    vals = rng.random(int(cardinality)).astype(dtype)
    return rng.choice(vals, size=n)


suite = BenchSuite("OPP-000001", "np.unique: hash path vs SIMD sort path (#31969)")
rng = np.random.default_rng(SEED)

if SMOKE:
    sweep = [("int64", 1_000, "high")]
else:
    sweep = [
        # main sweep: the reported dtype across sizes and both cardinality regimes
        ("int64", 1_000, "high"),
        ("int64", 10_000, "high"),
        ("int64", 100_000, "high"),
        ("int64", 1_000_000, "high"),   # the reported 46x case
        ("int64", 1_000, 100),
        ("int64", 10_000, 100),
        ("int64", 100_000, 100),
        ("int64", 1_000_000, 100),
        ("int64", 1_000_000, 10),       # the reported "still 3x" duplicate-heavy case
        # neighboring dtypes at the reported size
        ("int32", 1_000_000, "high"),
        ("int32", 1_000_000, 100),
        ("float64", 1_000_000, "high"),
        ("float64", 1_000_000, 100),
        ("float32", 1_000_000, "high"),
        ("float32", 1_000_000, 100),
        ("uint64", 1_000_000, "high"),
        ("uint32", 1_000_000, "high"),
        ("int16", 1_000_000, "high"),   # cardinality capped by dtype range
        ("int8", 1_000_000, "high"),
        # crossover probe: where does the sort path stop winning?
        ("int64", 64, "high"),
        ("int64", 128, "high"),
        ("int64", 256, "high"),
        ("int64", 512, "high"),
        ("int64", 256, 32),
        ("int64", 512, 32),
    ]

for dtype_name, n, card in sweep:
    dtype = np.dtype(dtype_name)
    data = make_data(rng, dtype, n, card)
    samples = 3 if SMOKE else (7 if n >= 1_000_000 else 11)
    label = f"{dtype_name}_n{n}_card{card}"
    suite.measure(
        case=f"unique_{label}",
        params={"dtype": dtype_name, "n": n, "cardinality": str(card),
                "baseline_op": "numpy.unique"},
        baseline=("numpy.unique", lambda d=data: np.unique(d)),
        candidates={"sort_unique": lambda d=data: sort_unique(d)},
        check=np.array_equal,
        samples=samples,
    )
    suite.measure(
        case=f"unique_values_{label}",
        params={"dtype": dtype_name, "n": n, "cardinality": str(card),
                "baseline_op": "numpy.unique_values"},
        baseline=("numpy.unique_values", lambda d=data: np.unique_values(d)),
        candidates={"sort_unique": lambda d=data: sort_unique(d)},
        check=sorted_set_equal,
        samples=samples,
    )

# correctness documentation case: NaN handling must match np.unique
nan_data = rng.random(10_000)
nan_data[rng.integers(0, 10_000, size=50)] = np.nan
suite.measure(
    case="unique_float64_n1e4_with_nans",
    params={"dtype": "float64", "n": 10_000, "cardinality": "high",
            "note": "50 NaN positions; naive candidate must FAIL this check"},
    baseline=("numpy.unique", lambda d=nan_data: np.unique(d)),
    candidates={
        "sort_unique": lambda d=nan_data: sort_unique(d),
        "sort_unique_nansafe": lambda d=nan_data: sort_unique_nansafe(d),
    },
    check=lambda c, b: np.array_equal(c, b, equal_nan=True),
    samples=3 if SMOKE else 11,
)

if not SMOKE:
    suite.save()
