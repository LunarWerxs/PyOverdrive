"""OPP-000020: contiguous-copy fast path for np.dot / np.matmul on
non-contiguous sliced arrays.

numpy/numpy#19650 reports that a[:, 0, :] @ a[:, :, 20] on a 3-D array
a = np.random.randn(1000, 100, 1000) (float64) is ~197x slower than copying
both operand views to a contiguous layout first with np.ascontiguousarray
and then running the same matmul. This reproducer measures:

  - baseline: a[:, mid, :] @ a[:, :, last]  -- the non-contiguous slice-and-
    multiply users write today, exactly as in the issue's own microbenchmark
  - candidate "ascontiguousarray": np.ascontiguousarray(a[:, mid, :]) @
    np.ascontiguousarray(a[:, :, last])  -- the reporter's own workaround,
    with the copy cost included inside the timed call

Because matmul (n, t) @ (x, m) requires t == x, and a[:, 0, :] has shape
(n, t) while a[:, :, 20] has shape (n, m), the leading and trailing array
dimensions must be equal (call it n); only the middle dimension m is free.
a's element count is therefore n*n*m, which grows very fast in n. The report
suggests sweeping "100, 500, 1000, 2000 in place of 1000 and 100" across
both n and m; taken as a literal full cross product that reaches shapes
like n=2000, m=2000 (16e9 elements, tens of GB) or n=1000, m=2000 (2e9
elements, ~16 GB float64), which cannot finish in ~90 seconds or fit in
memory on a normal dev machine. Per the report's own instruction to follow
the spirit when the literal sweep is impossible: this reproducer sweeps n
across {100, 300, 1000} at the reporter's original m=100 (so the reporter's
exact (1000, 100, 1000) shape is included unmodified), then separately
sweeps m across {100, 500, 1000} at smaller/moderate n to see the effect of
growing the free dimension without blowing up total elements. The largest
single case is the reporter's own (1000, 100, 1000) at ~800 MB (float64);
nothing larger is attempted. samples=7 (non-smoke) / 3 (smoke), chosen at
the low end of the house 5-11 range specifically to keep the full battery
under the ~90s budget given how slow the unfixed baseline path can be at
n=1000.

What semantics differ: none, mathematically -- both paths compute the same
matrix product. But they are NOT expected to be bit-identical: the
non-contiguous baseline runs numpy's generic strided ufunc inner loop while
the contiguous candidate runs a BLAS gemm call, and the two accumulate the
same sum in a different order. Different summation order is a different
rounding sequence in floating point, so correctness here is checked with
np.allclose at a tolerance keyed to the dtype's own precision (tighter for
float64 than float32), not with bit-identity.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 19650
SMOKE = "--smoke" in sys.argv
MID_IDX = 0
LAST_IDX = 20


def make_check(dtype):
    """allclose, not array_equal: the two paths use different accumulation
    order (generic strided loop vs BLAS gemm), see module docstring."""
    if dtype == np.float32:
        rtol, atol = 1e-4, 1e-5
    else:
        rtol, atol = 1e-9, 1e-12

    def check(cand, base):
        return np.allclose(cand, base, rtol=rtol, atol=atol)

    return check


if SMOKE:
    nm_pairs = [(50, 50), (100, 100)]
    dtypes = [np.float64]
else:
    nm_pairs = [
        (100, 100),
        (300, 100),
        (1000, 100),  # the reporter's exact (1000, 100, 1000) shape
        (300, 500),
        (100, 1000),
    ]
    dtypes = [np.float32, np.float64]

suite = BenchSuite("OPP-000020", "non-contiguous dot/matmul vs ascontiguousarray copy-first")
samples = 3 if SMOKE else 7

rng = np.random.default_rng(SEED)

for n, m in nm_pairs:
    last = min(LAST_IDX, m - 1)
    for dtype in dtypes:
        a = rng.standard_normal((n, m, n), dtype=dtype)
        dtype_name = np.dtype(dtype).name
        case = f"n{n}_m{m}_{dtype_name}"
        params = {
            "dtype": dtype_name,
            "n": n,
            "m": m,
            "a_elements": n * n * m,
            "view1_shape": [n, n],
            "view2_shape": [n, m],
        }

        def _noncontig(a=a, last=last):
            return a[:, MID_IDX, :] @ a[:, :, last]

        def _contig(a=a, last=last):
            x = np.ascontiguousarray(a[:, MID_IDX, :])
            y = np.ascontiguousarray(a[:, :, last])
            return x @ y

        suite.measure(
            case=case,
            params=params,
            baseline=("a[:,0,:] @ a[:,:,last]", _noncontig),
            candidates={"ascontiguousarray": _contig},
            check=make_check(dtype),
            samples=samples,
        )

if not SMOKE:
    suite.save()
