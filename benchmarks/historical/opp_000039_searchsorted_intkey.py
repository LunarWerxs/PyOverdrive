"""OPP-000039: searchsorted with a Python int key vs a dtype-cast key.

numpy/numpy#29719: np.searchsorted(a_int64_sorted, python_int) is
~32432x slower than passing a same-dtype key (12 s vs 0.37 ms for 100
calls), because the mismatched scalar promotes the COMPARISON to
float64-ish semantics element by element instead of bisecting natively.
Closed by-design upstream (charris: promotion rules), no fix landed -
a live gap on current numpy, which this battery re-measures.

What this reproducer measures:

  1. Key-cost grid: sorted int64 arrays at n in {1e3, 1e5, 1e7}, key a
     Python int in range, side='left' and side='right' - stock (Python
     int passed through) vs the cast route np.searchsorted(a,
     np.int64(key)).
  2. uint64 array with a Python int key (the promotion case the
     spun-off correctness issue #29727 concerns) - measured for time,
     checked for exact agreement; if stock itself answers WRONG here
     (that issue's claim), the check comparing routes will fail and
     that is a finding to record, not a script bug.
  3. Out-of-range keys (beyond int64 max / below min): the cast route
     cannot represent them, so the shipped predicate would refuse and
     these cases document stock's exact behavior as evidence rows with
     identical baseline/candidate (candidate = stock, ratio ~1.0).

Correctness: bit-identical indices (intp scalars).

House rules: never imports pyoverdrive. The candidate calls
np.searchsorted with a SAME-DTYPE key, which the shipped predicate
(python-int key only) refuses, so a patched dispatch could not recurse;
the existing searchsorted_sortqueries path needs ARRAY queries and also
refuses scalars.

Result JSON: benchmarks/results/OPP-000039/.
Run: .venv/Scripts/python benchmarks/historical/opp_000039_searchsorted_intkey.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 29719
SMOKE = "--smoke" in sys.argv


def cast_key(a, v, side="left"):
    return np.searchsorted(a, a.dtype.type(v), side=side)


def exact(cand, base):
    return type(cand) is type(base) and bool(np.array_equal(cand, base))


suite = BenchSuite("OPP-000039", "searchsorted python-int key: cast route vs stock promotion")
rng = np.random.default_rng(SEED)

SIZES = [10_000] if SMOKE else [1_000, 100_000, 10_000_000]
SAMPLES = 3 if SMOKE else 11

for n in SIZES:
    a = np.sort(rng.integers(-(2**60), 2**60, size=n, dtype=np.int64))
    key = int(a[n // 3]) + 1  # a plain Python int, in range
    for side in ("left", "right"):
        suite.measure(
            case=f"searchsorted_i64_n{n}_pyint_{side}",
            params={"n": n, "dtype": "int64", "side": side, "key": "python-int"},
            baseline=("numpy.searchsorted", lambda a=a, k=key, s=side: np.searchsorted(a, k, side=s)),
            candidates={"cast_key": lambda a=a, k=key, s=side: cast_key(a, k, s)},
            check=exact,
            samples=SAMPLES if n <= 100_000 else max(5, SAMPLES - 4),
        )

if not SMOKE:
    # uint64 with a python int key: the promotion-correctness minefield of
    # the spun-off numpy/numpy#29727; a check failure here is a FINDING
    au = np.sort(rng.integers(0, 2**63, size=100_000, dtype=np.uint64))
    ku = int(au[50_000]) + 1
    suite.measure(
        case="searchsorted_u64_n100000_pyint",
        params={"n": 100_000, "dtype": "uint64", "key": "python-int"},
        baseline=("numpy.searchsorted", lambda a=au, k=ku: np.searchsorted(a, k)),
        candidates={"cast_key": lambda a=au, k=ku: cast_key(a, k)},
        check=exact,
        samples=9,
    )
    # out-of-range key: stock handles arbitrary-precision ints correctly
    # (slowly); the cast route CANNOT and a shipped predicate refuses, so
    # this row documents stock behavior with a stock-equal candidate
    big_key = 2**70
    suite.measure(
        case="searchsorted_i64_n100000_key_beyond_int64",
        params={"n": 100_000, "dtype": "int64", "key": "2**70 (predicate would refuse)"},
        baseline=(
            "numpy.searchsorted",
            lambda a=a, k=big_key: np.searchsorted(a, k),
        ),
        candidates={"stock_echo": lambda a=a, k=big_key: np.searchsorted(a, k)},
        check=exact,
        samples=7,
    )
    suite.save()
