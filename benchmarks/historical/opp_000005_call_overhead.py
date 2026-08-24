"""OPP-000005: per-call dispatch/allocation overhead floor for NumPy ufuncs.

NumPy issue numpy/numpy#27456 proposes CPython call-site specialization
(quickening) so a ufunc call site can cache its casting/broadcasting/dispatch
decisions across repeat calls, claiming >2x speedups for small to medium
arrays. That mechanism itself is not reproducible in stock NumPy (it needs a
non-standard CPython build), so this is a baseline_only reproducer: it
quantifies the addressable overhead floor that such a mechanism would need to
attack, by comparing three ways of calling the same op:

  - baseline: np.add(a, b)                    (full generic ufunc dispatch)
  - operator: a + b                            (same dispatch via __add__)
  - preallocated out=: np.add(a, b, out=o)     (skips result allocation)

across array sizes from tiny (dispatch-dominated) to large (compute-bound),
so the size at which per-call overhead stops mattering is visible directly.

Run:
    .venv/Scripts/python benchmarks/historical/opp_000005_call_overhead.py --smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 27456
SMOKE = "--smoke" in sys.argv

suite = BenchSuite("OPP-000005", "ufunc call-site dispatch/allocation overhead floor")

sizes = [4, 64] if SMOKE else [4, 64, 1024, 65536]
samples = 3 if SMOKE else 11

for n in sizes:
    rng = np.random.default_rng(SEED + n)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    out = np.empty(n, dtype=a.dtype)

    def _baseline(a=a, b=b):
        return np.add(a, b)

    def _operator(a=a, b=b):
        return a + b

    def _preallocated_out(a=a, b=b, out=out):
        return np.add(a, b, out=out)

    suite.measure(
        case=f"float64_n{n}",
        params={"dtype": "float64", "n": n},
        baseline=("numpy.add", _baseline),
        candidates={
            "operator_form": _operator,
            "preallocated_out": _preallocated_out,
        },
        check=np.allclose,
        samples=samples,
    )

if not SMOKE:
    suite.save()
