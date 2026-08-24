"""OPP-000028: argsort on already-sorted int data -- numpy 2.x AVX2
regression re-measurement, plus the sorted-scan + arange fast path.

numpy/numpy#28714 reports that from numpy 1.26.4 to 2.0.0, arr.argsort() on
already-sorted data got roughly 3.6x-4.1x slower (reporter: 444us -> 1.59ms
for np.arange(10**5); r-devulap's ASV row time_argsort('quick', 'int64',
('ordered',)) at ratio 4.07), while NON-sorted data got faster. Cause per
r-devulap (numpy member): the AVX2 x86-simd-sort argsort from PR 25610,
which "has regressions for ordered arrays specially on 64-bit dtypes". The
load-bearing caveat: his fix (intel/x86-simd-sort PR 197, folded into numpy
PR 28619) merged 2025-05-14 for the 2.3.0 milestone, so on the numpy this
venv carries the HONEST outcome may be not_reproduced -- that is the point
of re-measuring, not a failure of the reproducer.

What this reproducer measures:

  1. Regression probe (stock only). Baseline arr.argsort() -- the stock
     call exactly as the reporter writes it -- on ordered vs random input
     at the same sizes. The random-pattern cases carry an EMPTY candidates
     dict on purpose: they exist so the evidence JSON records stock timing
     for both patterns on the same machine and numpy. The decisive read is
     ordered-baseline vs random-baseline at equal n: ordered materially
     slower than random is the issue's smoking gun still alive; ordered at
     or below random means PR 28619 did its job and the 4.07x claim is
     historical (scoped to the 2.0-2.2.x window).
  2. Fast-path headline. Candidate "sorted_scan_arange", the route named
     in the record's fast-path sketch: an O(n) strictly-increasing scan,
     then np.arange(n, dtype=np.intp) with no sort at all. The scan is
     INSIDE the timed candidate call, so the measured speedup is net of
     the guard's own cost. Strictly increasing input makes the argsort
     permutation unique, so the output is bit-identical to stock for every
     sort kind; the check is exact (np.array_equal plus dtype and shape),
     as befits an index-producing op -- no tolerance.
  3. Dtype/kind sweep at n=100_000 (the issue's own size): int32 alongside
     int64 (r-devulap: damage is "specially on 64-bit dtypes", so int32
     calibrates that), and kind='stable' alongside the default 'quick'.
  4. Adversarial guard floor: baseline-only case "almost_sorted" -- arange
     with its last two elements swapped, i.e. the single inversion at index
     n-2 that docs/research/opportunities/OPP-000028.md names as the
     worst case for a guarded router (full scan paid, then the full sort).

What is deliberately NOT measured, and why:

  - The candidate has NO fallback branch and is only ever exercised on
    strictly-increasing input (it asserts otherwise). A guarded all-cases
    router would have to call argsort itself for the non-sorted regime,
    which is exactly the self-patch recursion the house rule forbids; this
    follows the same pattern as opp_000013's skip_to_quantile. The
    router's worst case is still priced from what IS measured: adversarial
    router cost ~= candidate time on the ordered case of equal n (the scan
    runs to completion there, so that time IS the full-scan-plus-arange
    cost) + the almost_sorted baseline time.
  - Ties (merely non-decreasing input) and float dtypes stay outside the
    candidate: under ties stock quicksort's permutation is unspecified-but-
    concrete and arange may legitimately differ bitwise, and a naive
    comparison scan mishandles NaN. Strict increase on integer dtypes is
    the regime where bit-identity holds by construction, so that is the
    only regime measured.
  - The scan here is the plain-numpy np.all(a[:-1] < a[1:]), a full O(n)
    pass with no early exit. On the sorted inputs measured that is the
    same work an early-exit scan would do (there is no inversion to exit
    on); a real dispatch would use a chunked early-exit scan so random
    input pays a few comparisons, but that constant is not measurable from
    plain numpy and is not claimed here.

Sizes are the full sweep the research doc asks for -- n in {10**4, 10**5,
10**6}, int64, plus the n=10**5 dtype/kind sweep -- with no shrink: even
the slowest single call (stock argsort of 10**6 random int64) is tens of
milliseconds, so the whole non-smoke battery lands well under the ~90s
budget. Seed is the issue number. --smoke runs one small ordered/random
pair at n=1000 with 3 samples and does not save evidence.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 28714
SMOKE = "--smoke" in sys.argv


def make_ordered(n, dtype):
    return np.arange(n, dtype=dtype)


def make_random(rng, n, dtype):
    # A permutation of arange: distinct values (no ties), same value range
    # as the ordered case, matching the issue's ordered-vs-random contrast.
    return rng.permutation(n).astype(dtype)


def make_almost_sorted(n, dtype):
    a = np.arange(n, dtype=dtype)
    a[[-2, -1]] = a[[-1, -2]]  # single inversion at index n-2
    return a


def sorted_scan_arange(a):
    """The record's route: strictly-increasing scan, then arange.

    Only ever exercised on strictly-increasing input in this script (see
    module docstring); no fallback branch exists, so this candidate cannot
    recurse into argsort under patching. The scan is part of the timed
    call, so its cost is inside every measured speedup."""
    if not bool(np.all(a[:-1] < a[1:])):
        raise AssertionError(
            "sorted_scan_arange exercised on non-strictly-increasing input"
        )
    return np.arange(a.shape[0], dtype=np.intp)


def exact_indices(cand, base):
    """Exact check for an index-producing op: same shape, same dtype, same
    values. Bit-identity is the claim (unique permutation under strict
    increase), so no tolerance is appropriate."""
    return (
        cand.shape == base.shape
        and cand.dtype == base.dtype
        and np.array_equal(cand, base)
    )


if SMOKE:
    SIZES = [1_000]
else:
    SIZES = [10_000, 100_000, 1_000_000]

suite = BenchSuite(
    "OPP-000028",
    "argsort ordered-input regression re-measure + sorted-scan arange fast path",
)

rng = np.random.default_rng(SEED)

# Part 1: ordered vs random int64 across sizes, default kind ('quick').
for n in SIZES:
    samples = 3 if SMOKE else (5 if n >= 1_000_000 else 7)

    ordered = make_ordered(n, np.int64)
    suite.measure(
        case=f"ordered_int64_quick_n{n}",
        params={"dtype": "int64", "n": n, "kind": "quick", "pattern": "ordered"},
        baseline=("ndarray.argsort", lambda a=ordered: a.argsort()),
        candidates={"sorted_scan_arange": lambda a=ordered: sorted_scan_arange(a)},
        check=exact_indices,
        samples=samples,
    )

    random_arr = make_random(rng, n, np.int64)
    suite.measure(
        case=f"random_int64_quick_n{n}",
        params={"dtype": "int64", "n": n, "kind": "quick", "pattern": "random"},
        baseline=("ndarray.argsort", lambda a=random_arr: a.argsort()),
        candidates={},  # regression probe: stock timing on record, no route
        samples=samples,
    )

# Part 2: dtype/kind sweep and the adversarial guard floor, at the issue's
# own size. Skipped in smoke mode.
if not SMOKE:
    N = 100_000
    samples = 7

    # int32: r-devulap scopes the regression "specially" to 64-bit dtypes.
    ordered32 = make_ordered(N, np.int32)
    suite.measure(
        case=f"ordered_int32_quick_n{N}",
        params={"dtype": "int32", "n": N, "kind": "quick", "pattern": "ordered"},
        baseline=("ndarray.argsort", lambda a=ordered32: a.argsort()),
        candidates={"sorted_scan_arange": lambda a=ordered32: sorted_scan_arange(a)},
        check=exact_indices,
        samples=samples,
    )
    random32 = make_random(rng, N, np.int32)
    suite.measure(
        case=f"random_int32_quick_n{N}",
        params={"dtype": "int32", "n": N, "kind": "quick", "pattern": "random"},
        baseline=("ndarray.argsort", lambda a=random32: a.argsort()),
        candidates={},
        samples=samples,
    )

    # kind='stable': arange is bit-identical here too (strict increase),
    # and stable is the kind under which a future ties extension would be
    # exact, so its stock cost on ordered input is worth having on record.
    ordered_st = make_ordered(N, np.int64)
    suite.measure(
        case=f"ordered_int64_stable_n{N}",
        params={"dtype": "int64", "n": N, "kind": "stable", "pattern": "ordered"},
        baseline=(
            "ndarray.argsort",
            lambda a=ordered_st: a.argsort(kind="stable"),
        ),
        candidates={
            "sorted_scan_arange": lambda a=ordered_st: sorted_scan_arange(a)
        },
        check=exact_indices,
        samples=samples,
    )
    random_st = make_random(rng, N, np.int64)
    suite.measure(
        case=f"random_int64_stable_n{N}",
        params={"dtype": "int64", "n": N, "kind": "stable", "pattern": "random"},
        baseline=(
            "ndarray.argsort",
            lambda a=random_st: a.argsort(kind="stable"),
        ),
        candidates={},
        samples=samples,
    )

    # Adversarial guard floor: a guarded router's worst case is the full
    # scan (priced by the ordered candidate above) plus this baseline.
    adversarial = make_almost_sorted(N, np.int64)
    suite.measure(
        case=f"almost_sorted_int64_quick_n{N}",
        params={
            "dtype": "int64",
            "n": N,
            "kind": "quick",
            "pattern": "almost_sorted_inversion_at_n_minus_2",
        },
        baseline=("ndarray.argsort", lambda a=adversarial: a.argsort()),
        candidates={},
        samples=samples,
    )

if not SMOKE:
    suite.save()
