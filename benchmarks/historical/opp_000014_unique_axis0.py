"""OPP-000014: np.unique(arr, axis=0) on a trivial single-column 2-D array
pays the full structured-dtype comparison/sort overhead that axis=0 unique
needs for genuine multi-column data, even though there is nothing to make
"contiguous" that is not already contiguous (numpy/numpy#15713).

This reproducer measures two separate things, per OPP-000014.md's "What
PyOverdrive should do next" steps 1 and 2:

  1. Single-column trivial-expansion case (the issue's actual claim):
     baseline `np.unique(mat, axis=0)` where `mat = np.expand_dims(arr, 1)`
     (shape (N, 1)), against a ravel-and-reshape fast-path candidate that
     ravels `mat` back to 1-D and reshapes the unique result to (-1, 1) -
     semantically identical, since row-uniqueness on a single-column array
     is definitionally the same as element-uniqueness of that column (see
     OPP-000014.md "What semantics differ"). Swept across dtypes [int32,
     int64, uint32, uint64, float32, float64] as the report specifies. Sizes
     are [1000, 10000, 100000, 400000] - the report names 1_000_000 as the
     top size, but that is shrunk here to 400_000 to keep the full
     (non-smoke) battery under ~90s: 6 dtypes x 4 sizes x 2 kernels, plus
     the multi-column part below, is already 30 timed cases, and the
     relative (candidate vs baseline) cost ratio this claim is about is
     visible well before N=1e6.

     House rule: a fast-path candidate must not call a public numpy name it
     might itself patch inside a timed lambda. `ravel_reshape_unique` below
     therefore calls the private `numpy.lib._arraysetops_impl._unique1d`
     (the exact function `np.unique(arr)` with no axis/return_* flags
     dispatches to) instead of the public `np.unique`, so this reproducer
     stays honest even if PyOverdrive later patches `np.unique` itself.

  2. Genuine multi-column case (report step 2, not covered by the issue's
     own benchmark numbers): random (N, 3) and (N, 8) int64 arrays with
     duplicate rows, baseline `np.unique(arr, axis=0)` against
     DominicAntonacci's lexsort-then-diff `unique_axis0` (posted in-thread,
     2021-03-11), implemented here from its description ("based on
     numpy.lib.arraysetops._unique1d": lexsort the rows, then diff adjacent
     sorted rows to find the first occurrence of each distinct row - the
     same sort-then-diff shape as _unique1d itself, applied to whole rows
     via lexsort). Its own author calls this "asymptotically worse than
     np.unique" for many columns, so it is measured only at M=3 and M=8
     columns as the report names, not swept to a wide column count. Sizes
     [1000, 10000, 100000] keep this part inside the same time budget.

Per OPP-000014.md step 5, PR numpy/numpy#10588 (the PR that removed the
*original* axis=0 optimization eric-wieser refers to) was not separately
fetched here - this reproducer measures current numpy's behavior directly,
which is the thing that matters for a PyOverdrive fast-path decision
regardless of what #10588 contained.

Neither candidate performs arithmetic - only ravel/reshape, comparison,
sorting, and lexsort - and numpy's own axis=0 unique sorts a structured view
with one typed field per column (field-wise numeric comparison, not raw
byte/void memcmp - the byte-order hazard seberg raises is specific to the
*removed* PR #10588 optimization eric-wieser proposes reviving, not to
current numpy's behavior). So both the row set AND the row order should
match exactly; results are checked for exact (bit-identical) equality via
np.array_equal, not approximate closeness.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from numpy.lib._arraysetops_impl import _unique1d

from lab.dyno import BenchSuite

SEED = 15713
SMOKE = "--smoke" in sys.argv


def ravel_reshape_unique(mat):
    """Fast-path candidate for the single-column trivial-expansion case:
    ravel `mat` (shape (N, 1)) to 1-D, run the private unique1d kernel
    (what `np.unique(arr)` with no axis dispatches to), and reshape the
    result back to (-1, 1). Semantically identical to
    `np.unique(mat, axis=0)` for a single-column array - see module
    docstring.
    """
    return _unique1d(mat.ravel())[0].reshape(-1, 1)


def unique_axis0_lexsort(arr):
    """DominicAntonacci's lexsort-then-diff unique_axis0, implemented from
    its in-thread description: sort rows lexicographically (primary key =
    column 0), then keep only rows that differ from the previous sorted
    row. Mirrors _unique1d's own sort-then-diff structure but applied to
    whole rows via lexsort instead of a structured-dtype view.
    """
    if arr.shape[0] == 0:
        return arr.copy()
    order = np.lexsort(arr.T[::-1])
    sorted_arr = arr[order]
    is_new = np.concatenate(
        ([True], np.any(sorted_arr[1:] != sorted_arr[:-1], axis=1))
    )
    return sorted_arr[is_new]


DTYPES = [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64]

if SMOKE:
    single_col_sizes = [200, 2_000]
    multi_col_sizes = [200]
else:
    single_col_sizes = [1_000, 10_000, 100_000, 400_000]
    multi_col_sizes = [1_000, 10_000, 100_000]

suite = BenchSuite(
    "OPP-000014",
    "np.unique(axis=0): single-column ravel fast path and multi-column lexsort",
)
samples = 3 if SMOKE else 7

rng = np.random.default_rng(SEED)

# --- Part 1: single-column trivial-expansion case ---
for dtype in DTYPES:
    for n in single_col_sizes:
        hi = n // 2 + 10  # force duplicates regardless of dtype range
        if np.issubdtype(dtype, np.integer):
            arr = rng.integers(0, hi, size=n, dtype=dtype)
        else:
            arr = rng.integers(0, hi, size=n).astype(dtype)
        mat = np.expand_dims(arr, 1)
        case = f"singlecol_{dtype.__name__}_n{n}"
        params = {"dtype": dtype.__name__, "n": n, "shape": "(N, 1)"}

        suite.measure(
            case=case,
            params=params,
            baseline=("numpy.unique(mat, axis=0)", lambda mat=mat: np.unique(mat, axis=0)),
            candidates={
                "ravel_reshape": lambda mat=mat: ravel_reshape_unique(mat),
            },
            check=np.array_equal,
            samples=samples,
        )

# --- Part 2: genuine multi-column case ---
for ncols in (3, 8):
    for n in multi_col_sizes:
        hi = max(n // 4, 10)  # force duplicate rows
        arr = rng.integers(0, hi, size=(n, ncols)).astype(np.int64)
        case = f"multicol_c{ncols}_n{n}"
        params = {"dtype": "int64", "n": n, "shape": f"(N, {ncols})"}

        suite.measure(
            case=case,
            params=params,
            baseline=("numpy.unique(arr, axis=0)", lambda arr=arr: np.unique(arr, axis=0)),
            candidates={
                "lexsort": lambda arr=arr: unique_axis0_lexsort(arr),
            },
            check=np.array_equal,
            samples=samples,
        )

if not SMOKE:
    suite.save()
