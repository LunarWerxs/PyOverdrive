"""OPP-000003: np.add.at (ufunc.at) vs alternatives for grouped/indexed accumulation.

Source claim (numpy/numpy#5922, numpy/numpy#11156): np.add.at is 10-25x slower
than hand-vectorized alternatives for accumulating values into an output array
at repeated integer indices, up to ~60x slower than np.bincount in one
maintainer quick-test. Both issues were closed in 2023 by numpy/numpy#23136
("create and use indexed inner loops"), which is expected to have closed most
of the gap. This reproducer measures the current gap, sweeping n (data size)
and k (number of distinct groups) to expose any remaining threshold effects.

Usage:
    .venv/Scripts/python benchmarks/historical/opp_000003_ufunc_at.py --smoke
    .venv/Scripts/python benchmarks/historical/opp_000003_ufunc_at.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 20260823
SMOKE = "--smoke" in sys.argv

N_VALUES = [1_000] if SMOKE else [10_000, 100_000, 1_000_000]
K_VALUES = [10, 1_000, 100_000]


def make_data(n, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=n, dtype=np.int64)
    vals = rng.uniform(-50.0, 50.0, size=n)
    return idx, vals


def run_add_at(idx, vals, k):
    def run():
        out = np.zeros(k, dtype=np.float64)
        np.add.at(out, idx, vals)
        return out

    return run


def run_bincount(idx, vals, k):
    def run():
        return np.bincount(idx, weights=vals, minlength=k)

    return run


def run_reduceat(idx, vals, k):
    # Sort-then-reduceat trick, timed end to end (sort included) so it is
    # compared fairly against add.at/bincount, which both also accept
    # unsorted idx. Groups with zero members stay zero in the output.
    def run():
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        sorted_vals = vals[order]
        unique_idx, start = np.unique(sorted_idx, return_index=True)
        sums = np.add.reduceat(sorted_vals, start)
        out = np.zeros(k, dtype=np.float64)
        out[unique_idx] = sums
        return out

    return run


def main():
    suite = BenchSuite("OPP-000003", "ufunc.at (np.add.at) vs bincount/reduceat for grouped accumulation")

    for n in N_VALUES:
        for k in K_VALUES:
            idx, vals = make_data(n, k, seed=SEED + n + k)
            case = f"n{n}_k{k}"
            samples = 3 if SMOKE else (5 if n >= 1_000_000 else 11)
            suite.measure(
                case=case,
                params={"dtype": "float64", "index_dtype": "int64", "n": n, "k": k},
                baseline=("numpy.add.at", run_add_at(idx, vals, k)),
                candidates={
                    "bincount": run_bincount(idx, vals, k),
                    "sort_reduceat": run_reduceat(idx, vals, k),
                },
                check=lambda a, b: np.allclose(a, b),
                samples=samples,
            )

    if not SMOKE:
        suite.save()


if __name__ == "__main__":
    main()
