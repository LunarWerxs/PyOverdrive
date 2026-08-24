"""OPP-000015: pre-sorted query-array fast path for np.searchsorted.

numpy/numpy#10937 claims a ~10.71x wall-clock speedup (reporter's own %time:
10.4s vs 971ms at N=5,000,000 float64) when np.searchsorted(x, y) is called
with x pre-sorted and y left in its original unsorted order, versus sorting y
first (timed together with the sort) and calling the identical searchsorted.
This reproducer measures:

  - baseline: "numpy.searchsorted" -- np.searchsorted(x, y) with x sorted,
    y unsorted (what users write today; report step 1)
  - candidate "argsort_unpermute": perm = np.argsort(y); y_sorted = y[perm];
    idx = np.searchsorted(x, y_sorted); result = np.empty_like(idx);
    result[perm] = idx -- argsort + search + unpermute timed as one unit,
    matching how the reporter timed sort-then-search together (report step 1)

Deviation from the report's size sweep, noted per instructions: the report
asks for sizes 1e3, 1e4, 1e5, 1e6, 5e6, 1e7 (both float64 and int64) plus a
size-ratio sweep with len(x) fixed at 1e6 while len(y) ranges 1e3-1e7. That
full matrix at 1e7 elements, both dtypes, 5-11 samples each, does not fit
the ~90s non-smoke time budget on ordinary hardware. So: 1e7 is dropped
entirely; the reporter's own 5e6 point is kept (float64 only, reduced
sample count) since it is the one directly-reported shape; the size-ratio
sweep keeps len(x) fixed at 1e6 as asked but is float64-only, also with a
reduced sample count at its largest points. The 1e3-1e6 equal-size sweep
still runs both float64 and int64 at full sample count.

Correctness check is bit-identical (np.array_equal on the integer index
arrays): each element's insertion index depends only on x and that single
element of y, not on y's other elements or their order (see OPP-000015.md,
"What semantics differ"), so reordering y and unpermuting the result must
reproduce stock output exactly -- no tolerance is justified here.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 10937
SMOKE = "--smoke" in sys.argv
DTYPES = ["float64", "int64"]


def make_haystack(rng, n, dtype, value_scale):
    """Sorted haystack `x` of length n. value_scale sets the integer value
    range for int64 so density is comparable to the float64 (randn) case;
    for float64 it is unused (randn's own spread is used, as reported)."""
    if dtype == "float64":
        return np.sort(rng.standard_normal(n))
    hi = max(value_scale, 1) * 4 + 1
    return np.sort(rng.integers(0, hi, size=n, dtype=np.int64))


def make_queries(rng, n, dtype, value_scale):
    """Unsorted query array `y` of length n, drawn from the same value range
    as a haystack built with the same value_scale (so queries actually land
    inside the haystack's range, including for size-ratio cases where n
    differs from the haystack's own length)."""
    if dtype == "float64":
        return rng.standard_normal(n)
    hi = max(value_scale, 1) * 4 + 1
    return rng.integers(0, hi, size=n, dtype=np.int64)


def argsort_unpermute(x, y):
    """Candidate from the report: sort a copy of y, searchsorted against the
    sorted copy, then scatter the result indices back into y's original
    positions using the same permutation. No stability requirement, since
    each element's insertion index is independent of the other query
    elements (see OPP-000015.md, "What semantics differ")."""
    perm = np.argsort(y)
    y_sorted = y[perm]
    idx = np.searchsorted(x, y_sorted)
    result = np.empty_like(idx)
    result[perm] = idx
    return result


if SMOKE:
    equal_sizes = [50, 500]
    reporter_scale_sizes: list[int] = []
    ratio_x_size = 500
    ratio_y_sizes = [50, 2_000]
else:
    equal_sizes = [1_000, 10_000, 100_000, 1_000_000]
    reporter_scale_sizes = [5_000_000]
    ratio_x_size = 1_000_000
    ratio_y_sizes = [1_000, 10_000, 100_000, 1_000_000, 5_000_000]

suite = BenchSuite(
    "OPP-000015", "presorted-query fast path: np.searchsorted vs argsort+unpermute"
)
samples = 3 if SMOKE else 7
samples_big = 3 if SMOKE else 5

rng = np.random.default_rng(SEED)


def run_case(case_name, x, y, n_samples):
    suite.measure(
        case=case_name,
        params={"dtype": str(x.dtype), "len_x": len(x), "len_y": len(y)},
        baseline=("numpy.searchsorted", lambda x=x, y=y: np.searchsorted(x, y)),
        candidates={
            "argsort_unpermute": lambda x=x, y=y: argsort_unpermute(x, y),
        },
        check=np.array_equal,
        samples=n_samples,
    )


# equal-size sweep, both dtypes, full sample count
for dtype in DTYPES:
    for n in equal_sizes:
        x = make_haystack(rng, n, dtype, n)
        y = make_queries(rng, n, dtype, n)
        run_case(f"equal_n{n}_{dtype}", x, y, samples)

# reporter's own shape (N=5,000,000 float64), reduced samples for time budget
for n in reporter_scale_sizes:
    x = make_haystack(rng, n, "float64", n)
    y = make_queries(rng, n, "float64", n)
    run_case(f"equal_n{n}_float64_reporterscale", x, y, samples_big)

# size-ratio sweep: haystack fixed at ratio_x_size, query length varies
# (tests eric-wieser's "for large Nb, search_sorted is faster" caveat)
x_ratio = make_haystack(rng, ratio_x_size, "float64", ratio_x_size)
for m in ratio_y_sizes:
    y = make_queries(rng, m, "float64", ratio_x_size)
    n_samples = samples_big if max(ratio_x_size, m) >= 1_000_000 else samples
    run_case(f"ratio_x{ratio_x_size}_y{m}_float64", x_ratio, y, n_samples)

if not SMOKE:
    suite.save()
