"""reduce_tiny_trailing calibration: the regime edges OPP-000026 left open.

The OPP-000026 reproducer fixed total size at ~3e6 elements and swept the
trailing axis k: per-slice wins 2.60-6.56x at k in {2,3,4} (f32 8.41x),
k=1 is a wash, k=8 flips to 0.74x. A dispatch predicate needs two edges
the reproducer never measured:

- the ROWS floor: every winning cell had >= 750_000 rows. Where does the
  win die as the array shrinks? (Per-slice pays k stock-call overheads;
  small arrays cannot amortize them.)
- the k in {5,6,7} gap between the measured 4 (win) and 8 (loss).

Candidate = the per-slice route exactly as the fast path will run it:
reshape(-1, k) (a view for C-order input) and one full stock reduction
per column. This script never imports pyoverdrive, and the column calls
are full reductions (axis=None) that a small-trailing-axis predicate
refuses by construction, so the measured candidate is the shipped one.

Correctness: summation order differs from stock's pairwise traversal, so
the check is allclose with the reproducer's dtype-scaled rtol (1e-9 for
float64, 1e-3 for float32), not exactness.

Result JSON: benchmarks/results/MEANSUM-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_meansum_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv


def per_slice(op, a2):
    out = np.empty(a2.shape[1], dtype=a2.dtype)
    for c in range(a2.shape[1]):
        out[c] = op(a2[:, c])
    return out


def make_check(rtol):
    def check(cand, base):
        return (
            cand.shape == base.shape
            and cand.dtype == base.dtype
            and bool(np.allclose(cand, base, rtol=rtol, atol=0.0))
        )

    return check


CHECK = {"float64": make_check(1e-9), "float32": make_check(1e-3)}

suite = BenchSuite("MEANSUM-CAL", "per-slice tiny-trailing-axis reduction: rows floor + k edge")
rng = np.random.default_rng(8480)

# (op_name, stock_op, dtype, rows, k)
if SMOKE:
    CASES = [("mean", np.mean, "float64", 10_000, 3)]
    SAMPLES = 3
else:
    CASES = []
    # rows floor sweep, f64 mean, the three measured-winning k values
    for k in (2, 3, 4):
        for rows in (1_000, 10_000, 100_000, 750_000):
            CASES.append(("mean", np.mean, "float64", rows, k))
    # sum shares the ufunc.reduce mechanism; confirm its floor tracks mean's
    # at every k the predicate would accept, not just k=3
    for rows in (1_000, 10_000, 100_000):
        CASES.append(("sum", np.sum, "float64", rows, 3))
    for k in (2, 4):
        for rows in (10_000, 100_000):
            CASES.append(("sum", np.sum, "float64", rows, k))
    # f32 grid (the reproducer measured 8.41x only at 1e6 rows, k=3, mean):
    # every (op, k, rows) cell the predicate would admit at its floor
    for op_name, op in (("mean", np.mean), ("sum", np.sum)):
        for k in (2, 3, 4, 5):
            for rows in (10_000, 100_000):
                CASES.append((op_name, op, "float32", rows, k))
    for rows in (10_000, 100_000):
        CASES.append(("sum", np.sum, "float64", rows, 5))
    # the k edge between measured 4 (2.60x) and 8 (0.74x), plus k=5 at the
    # smaller rows the floor would admit
    for k in (5, 6, 7):
        CASES.append(("mean", np.mean, "float64", 500_000, k))
    for rows in (10_000, 100_000):
        CASES.append(("mean", np.mean, "float64", rows, 5))
    SAMPLES = 9

for op_name, op, dtype, rows, k in CASES:
    a = rng.random(size=(rows, k), dtype=dtype)
    samples = SAMPLES if rows <= 100_000 else max(5, SAMPLES - 2)
    suite.measure(
        case=f"{op_name}_axis0_rows{rows}_k{k}_{dtype}",
        params={"op": op_name, "dtype": dtype, "rows": rows, "k": k},
        baseline=(f"numpy.{op_name}", lambda a=a, op=op: op(a, axis=0)),
        candidates={"per_slice": lambda a=a, op=op: per_slice(op, a)},
        check=CHECK[dtype],
        samples=samples,
    )

if not SMOKE:
    suite.save()
