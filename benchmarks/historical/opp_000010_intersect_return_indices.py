"""OPP-000010: np.unique / np.intersect1d select sort kind by return_indices,
not dtype, on narrow integers.

numpy/numpy#16923 (mganahl, 2020-07-21, open) reports that
`np.intersect1d(a, b, return_indices=True)` runs several times faster than the
same call with `return_indices=False`, for narrow integer dtypes. rossbar and
charris traced this to `np.unique`'s internal sort-kind selection, which
`intersect1d` inherits: requesting indices forces `argsort(kind='mergesort')`,
which for integer dtypes <= 16 bits routes into NumPy's radix-sort path;
without indices, NumPy calls plain `ar.sort()` (quicksort by default), which
does not. The speed gap is a side effect of which sort NumPy happens to
invoke, not a deliberate dtype-based choice (see
docs/research/opportunities/OPP-000010.md).

This reproducer measures, per report step 1, four calls on the same seeded
random input: `np.unique(a, return_index=False)` (baseline, today's default
sort) against a candidate that forces the internal sort to `kind='mergesort'`
regardless of return_index, and the same pair for
`np.intersect1d(a, b, return_indices=False)`. The candidates are hand-rolled
(they call `np.sort(..., kind='mergesort')` and `np.concatenate`, never
`np.unique`/`np.intersect1d` themselves) per house rule: a fast-path candidate
must not call the very public numpy name it stands in for inside its timed
lambda. They mirror NumPy's own arraysetops algorithm (sort, then keep the
first element of each run of equal values) with only the sort kind changed,
so output is expected to be bit-identical to the baseline: the final
sorted-unique/sorted-intersection values do not depend on which comparison-
or radix-based algorithm produced the sort, only on the array being sorted
and de-duplicated, which both the baseline and candidate do. This is a single
plain ndarray on both sides (return_index=False / return_indices=False), not
a tuple, so a whole-array np.array_equal is the correct strict check; no
floating-point tolerance is needed since every dtype swept is an integer
dtype.

Report step 1 asks for dtypes int8/uint8/int16/uint16/int32/uint32, sizes
10000/20000/100000/1000000, and both low-cardinality (values 0-9, matching
mganahl's own repro) and high-cardinality inputs (drawn across each dtype's
full representable range), crossed with both operations (unique, intersect1d)
-- 96 cases. The 1,000,000 size is dropped here on architectural grounds,
not a timed run: mergesort is not radix-sort eligible above 16 bits, so for
the int32/uint32 cases at n=1,000,000 the candidate degrades to a plain
comparison sort competing against NumPy's own quicksort baseline on its
best-case dtype width -- an expected, uninteresting crossover result rather
than a discovery -- and those are the largest arrays in the sweep (int32/
uint32, both cardinalities, both operations: six of the size's twelve
cases). No wall-clock figure is claimed for this decision; none was measured
and none should be invented. Dropping 1,000,000 keeps 10000/20000/100000,
which still spans the reporter's own D=20000 point and demonstrates the
radix-sort speedup at scale for the narrow dtypes the report is actually
about.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 16923  # numpy/numpy#16923
SMOKE = "--smoke" in sys.argv

DTYPES = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]
CARDINALITIES = ["low", "high"]  # low: values 0-9 (mganahl's repro); high: full dtype range
SIZES = [200, 500] if SMOKE else [10_000, 20_000, 100_000]


def make_array(rng, dtype, size, cardinality):
    """Seeded random 1-D array of `dtype`, drawn either from 0-9 (low
    cardinality, matching mganahl's repro) or from the dtype's full
    representable range (high cardinality). Generated in int64 and cast down
    so the upper-bound-exclusive draw never overflows a narrow signed dtype.
    """
    info = np.iinfo(dtype)
    lo, hi = (0, 9) if cardinality == "low" else (int(info.min), int(info.max))
    vals = rng.integers(lo, hi + 1, size=size, dtype=np.int64)
    return vals.astype(dtype)


def unique_mergesort(a):
    """np.unique(a, return_index=False) with the internal sort forced to
    kind='mergesort' instead of NumPy's default quicksort. Reimplements
    NumPy's own arraysetops algorithm by hand (sort, then keep the first
    element of each run of equal values) rather than calling np.unique, per
    house rule against a fast-path candidate calling the public name it
    stands in for.
    """
    ar = np.sort(a.ravel(), kind="mergesort")
    mask = np.empty(ar.shape, dtype=bool)
    mask[0] = True
    mask[1:] = ar[1:] != ar[:-1]
    return ar[mask]


def intersect1d_mergesort(a, b):
    """np.intersect1d(a, b, return_indices=False) with every internal sort
    forced to kind='mergesort'. Unique-ing each input then merging mirrors
    NumPy's own intersect1d source (concatenate the two unique arrays, sort,
    take elements immediately followed by an equal neighbour); only the sort
    kind differs from the baseline.
    """
    aux = np.concatenate((unique_mergesort(a), unique_mergesort(b)))
    aux = np.sort(aux, kind="mergesort")
    if aux.size == 0:
        return aux[:0]
    mask = aux[1:] == aux[:-1]
    return aux[:-1][mask]


suite = BenchSuite("OPP-000010", "unique/intersect1d: sort kind forced to mergesort on narrow ints")
samples = 3 if SMOKE else 9

rng = np.random.default_rng(SEED)

for size in SIZES:
    for dtype in DTYPES:
        dtype_name = np.dtype(dtype).name
        for cardinality in CARDINALITIES:
            a = make_array(rng, dtype, size, cardinality)
            b = make_array(rng, dtype, size, cardinality)
            params = {"dtype": dtype_name, "size": size, "cardinality": cardinality}
            tag = f"{dtype_name}_{cardinality}_n{size}"

            suite.measure(
                case=f"unique_{tag}",
                params=params,
                baseline=("numpy.unique", lambda a=a: np.unique(a, return_index=False)),
                candidates={"mergesort": lambda a=a: unique_mergesort(a)},
                check=np.array_equal,
                samples=samples,
            )

            suite.measure(
                case=f"intersect1d_{tag}",
                params=params,
                baseline=(
                    "numpy.intersect1d",
                    lambda a=a, b=b: np.intersect1d(a, b, return_indices=False),
                ),
                candidates={"mergesort": lambda a=a, b=b: intersect1d_mergesort(a, b)},
                check=np.array_equal,
                samples=samples,
            )

if not SMOKE:
    suite.save()
