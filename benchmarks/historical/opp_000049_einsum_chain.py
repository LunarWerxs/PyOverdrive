"""OPP-000049: chain (>=3-operand) numpy.einsum, default loop vs optimize=True.

numpy/numpy#11714 class: with optimize=False numpy runs a single fused C
loop over the union of every index in the contraction, so a chain like
ij,jk,kl->il costs O(i*j*k*l) where the optimized route pays two BLAS
matmuls instead. This is the chain-regime sibling of the shipped
two-operand einsum_optimize (OPP-000018): same routing (stock
np.einsum(..., optimize=True), no reimplemented contraction), different
gate (naive loop volume: product of the distinct input labels' extents).
Numeric comparison (rtol 1e-9): optimize=True sums in a different order.

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000049/.
Run: .venv/Scripts/python benchmarks/historical/opp_000049_einsum_chain.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000049", "chain einsum (>=3 operands): default loop vs optimize=True")
rng = np.random.default_rng(11714)


def close(rtol):
    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        return c.shape == b.shape and c.dtype == b.dtype and bool(
            np.allclose(c, b, rtol=rtol, atol=0.0, equal_nan=True)
        )

    return _chk


SIZES = [16] if SMOKE else [16, 32, 64]
SUBS3 = "ij,jk,kl->il"

for n in SIZES:
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    c = rng.standard_normal((n, n))
    suite.measure(
        case=f"einsum_chain3_ijkl_n{n}",
        params={"n": n, "operands": 3, "subs": SUBS3},
        baseline=("einsum_default", lambda a=a, b=b, c=c: np.einsum(SUBS3, a, b, c)),
        candidates={"einsum_optimize": lambda a=a, b=b, c=c: np.einsum(SUBS3, a, b, c, optimize=True)},
        check=close(1e-9),
        samples=SAMPLES,
    )

N4 = 16 if SMOKE else 32
SUBS4 = "ij,jk,kl,lm->im"
a4 = rng.standard_normal((N4, N4))
b4 = rng.standard_normal((N4, N4))
c4 = rng.standard_normal((N4, N4))
d4 = rng.standard_normal((N4, N4))
suite.measure(
    case=f"einsum_chain4_ijklm_n{N4}",
    params={"n": N4, "operands": 4, "subs": SUBS4},
    baseline=("einsum_default", lambda a=a4, b=b4, c=c4, d=d4: np.einsum(SUBS4, a, b, c, d)),
    candidates={"einsum_optimize": lambda a=a4, b=b4, c=c4, d=d4: np.einsum(SUBS4, a, b, c, d, optimize=True)},
    check=close(1e-9),
    samples=SAMPLES,
)

if not SMOKE:
    suite.save()
