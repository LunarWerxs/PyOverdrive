"""OPP-000002: np.inner vs np.tensordot for ndim > 2 (numpy/numpy#12778).

Claim (2019, still open; duplicate of #619 from 2012): np.inner(a, b) is ~10x
slower than the mathematically equivalent np.tensordot(a, b, axes=(-1, -1))
for multidimensional arrays. Maintainer-confirmed mechanism: inner routes
through PyArray_MatrixProduct2, which only makes a single BLAS call for
ndim <= 2 and otherwise loops 2-D BLAS calls; tensordot reshapes to one GEMM.

Semantics caveat carried into the record: tensordot's single-GEMM summation
order differs from inner's looped order, so results are numerically
equivalent, not bit-identical. check uses a tight relative tolerance.

The 2-D and 1-D cases are regression guards: inner already uses BLAS there,
so the candidate must NOT be dispatched in those regimes (tensordot's Python
overhead makes it slower); the crossover is the Gearbox predicate.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SEED = 12778


def tight_close(cand, base):
    # summation-order differences scale with precision: tight but achievable
    rtol = 1e-9 if base.dtype == np.float64 else 1e-4
    atol = 1e-12 if base.dtype == np.float64 else 1e-6
    return (
        cand.shape == base.shape
        and cand.dtype == base.dtype
        and np.allclose(cand, base, rtol=rtol, atol=atol)
    )


suite = BenchSuite("OPP-000002", "np.inner vs tensordot single-GEMM (#12778)")
rng = np.random.default_rng(SEED)

if SMOKE:
    shapes = [((5, 5, 32), (100, 32), 3, "float64")]
else:
    shapes = [
        # the reported case, verbatim shapes
        ((25, 25, 500), (10_000, 500), 5, "float64"),
        # neighbors: smaller and differently proportioned ndim>2 cases
        ((5, 5, 100), (1_000, 100), 11, "float64"),
        ((10, 10, 2_000), (5_000, 2_000), 5, "float64"),
        ((50, 4, 64), (20_000, 64), 7, "float64"),
        # dtype neighbor: float32 on the reported shape
        ((25, 25, 500), (10_000, 500), 5, "float32"),
        # regression guards: regimes where inner is already a single BLAS call
        ((500, 500), (500, 500), 11, "float64"),
        ((1_000,), (1_000,), 11, "float64"),
    ]

for shape_a, shape_b, samples, dtype_name in shapes:
    dt = np.dtype(dtype_name)
    a = rng.random(shape_a).astype(dt)
    b = rng.random(shape_b).astype(dt)
    ndim_tag = f"{len(shape_a)}d_x_{len(shape_b)}d"
    size_tag = "x".join(map(str, shape_a)) + "__" + "x".join(map(str, shape_b))
    suite.measure(
        case=f"inner_{ndim_tag}_{size_tag}_{dtype_name}",
        params={
            "shape_a": list(shape_a),
            "shape_b": list(shape_b),
            "dtype": dtype_name,
            "contraction_len": shape_a[-1],
        },
        baseline=("numpy.inner", lambda a=a, b=b: np.inner(a, b)),
        candidates={
            "tensordot": lambda a=a, b=b: np.tensordot(a, b, axes=(-1, -1)),
        },
        check=tight_close,
        samples=3 if SMOKE else samples,
    )

if not SMOKE:
    suite.save()
