"""Small-integer radix-sort calibration for the shared _sort_unique helper.

OPP-000010 (numpy/numpy#16923): numpy's stable sort kind for <= 16-bit
integers is a radix sort, but np.unique / np.intersect1d pick their sort by
return_indices, not dtype, so int8/uint8/int16/uint16 callers never get it.
The reproducer measured forcing kind='mergesort': unique up to 28x,
intersect1d up to 16.1x at high cardinality, ~2.1x at the reporter's own
low-cardinality regime, LOSSES only for 32-bit+ dtypes (excluded here).

The shipped change extends unique_sort._SUPPORTED_DTYPES with the four
small dtypes and switches its np.sort to kind='stable' when itemsize <= 2.
That ONE helper feeds THREE families, so this battery measures all three
surfaces against stock, at low (values 0-9) and high (full dtype range)
cardinality:

- np.unique 1-D (unique_sort)
- np.intersect1d (intersect_sorted; also needs the dtype in ITS table)
- np.unique(axis=0) single column (unique_axis0_column, threshold 1000
  rows) - it inherits the dtype set implicitly and must not ship
  unmeasured.

Candidates are the two helper variants (current quicksort-based vs
kind='stable'), run through the same dedupe logic, so the delta is the
sort kind alone; the baseline is stock numpy.

Check: exact equality (integer sort-unique is deterministic).

Result JSON: benchmarks/results/SMALLINT-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_smallint_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from lab.dyno import BenchSuite

SEED = 16923
SMOKE = "--smoke" in sys.argv


def sort_unique(a, kind=None):
    s = np.sort(a, axis=None, kind=kind)
    mask = np.empty(s.size, dtype=bool)
    mask[0] = True
    np.not_equal(s[1:], s[:-1], out=mask[1:])
    return s[mask]


def intersect_via(a, b, kind=None):
    ua = sort_unique(a, kind)
    ub = sort_unique(b, kind)
    ref, query = (ua, ub) if ua.size >= ub.size else (ub, ua)
    idx = np.searchsorted(ref, query)
    idx[idx == ref.size] = ref.size - 1
    return query[ref[idx] == query]


def exact(c, b):
    return c.dtype == b.dtype and c.shape == b.shape and bool(np.array_equal(c, b))


DTYPES = [np.int8, np.uint8, np.int16, np.uint16]
SIZES = [200] if SMOKE else [100, 1_000, 10_000, 100_000]
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("SMALLINT-CAL", "kind='stable' radix vs quicksort for <=16-bit unique/intersect")
rng = np.random.default_rng(SEED)


def draw(dtype, n, cardinality):
    info = np.iinfo(dtype)
    if cardinality == "low":
        lo, hi = 0, 10
    else:
        lo, hi = info.min, info.max + 1
    return rng.integers(lo, hi, size=n, dtype=dtype)


for dtype in DTYPES:
    dt = np.dtype(dtype).name
    for n in SIZES:
        for card in ("low", "high"):
            a = draw(dtype, n, card)
            suite.measure(
                case=f"unique_{dt}_n{n}_{card}",
                params={"op": "unique", "dtype": dt, "n": n, "cardinality": card},
                baseline=("numpy.unique", lambda a=a: np.unique(a)),
                candidates={
                    "sortunique_quicksort": lambda a=a: sort_unique(a),
                    "sortunique_stable": lambda a=a: sort_unique(a, "stable"),
                },
                check=exact,
                samples=SAMPLES,
            )

for dtype in (DTYPES if not SMOKE else [np.int16]):
    dt = np.dtype(dtype).name
    for n in ([200] if SMOKE else [400, 1_000, 3_000, 10_000, 100_000]):
        for card in ("low", "high"):
            a = draw(dtype, n, card)
            b = draw(dtype, max(n // 5, 10), card)
            suite.measure(
                case=f"intersect_{dt}_n{n}_{card}",
                params={"op": "intersect1d", "dtype": dt, "n": n, "cardinality": card},
                baseline=("numpy.intersect1d", lambda a=a, b=b: np.intersect1d(a, b)),
                candidates={
                    "intersect_quicksort": lambda a=a, b=b: intersect_via(a, b),
                    "intersect_stable": lambda a=a, b=b: intersect_via(a, b, "stable"),
                },
                check=exact,
                samples=SAMPLES,
            )

# unique(axis=0) single column inherits the dtype set through the shared
# helper; every dtype it gains must be measured here before it ships
for dtype in (DTYPES if not SMOKE else [np.int16]):
    dt = np.dtype(dtype).name
    for n in ([500] if SMOKE else [1_000, 3_000, 10_000, 100_000]):
        a = draw(dtype, n, "high").reshape(-1, 1)
        suite.measure(
            case=f"unique_axis0_{dt}_n{n}",
            params={"op": "unique_axis0", "dtype": dt, "n": n},
            baseline=("numpy.unique_axis0", lambda a=a: np.unique(a, axis=0)),
            candidates={
                "column_stable": lambda a=a: sort_unique(a.ravel(), "stable").reshape(-1, 1),
            },
            check=exact,
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
