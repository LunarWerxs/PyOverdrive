"""OPP-000027: real-by-complex np.dot is the one slow pairing of the four.

numpy/numpy#10468 (dbstein, 2018-01-25) reports that A.dot(B) with A real
float64 and B complex128 is "very slow ... (10-50x compared to A real/B
real, A complex/B complex, and A complex/B real)", persisting across
BLAS/LAPACK setups. The repro is a matrix-vector (GEMV-shaped) product:
Ar = A.real.copy() a (10000, 1000) float64 matrix, B a (1000,) complex128
vector. On the reporter's MKL box Ar.dot(B) took 110 ms vs 2.54 ms for the
all-complex product, the 43.31x carried by the corpus record as the claim
figure (derived: 110 / 2.54, upcast cost excluded). All of that was numpy
1.13.3/1.14.0 under Python 2.7; this reproducer re-measures the claim
regime on current numpy, so the honest outcome may be not_reproduced.

What IS measured, at the record's own sizes and dtypes (no shrink needed:
one (10000, 1000) x (1000,) matvec is milliseconds-scale even on the slow
path, so the full battery is nowhere near the ~90s budget):

  - baseline: Ar.dot(B), the stock call exactly as the reporter wrote it
    (real float64 left, complex128 right), the one slow direction all three
    measurement sets in the thread agree on.
  - candidate "view_as_real_gemv": the record's preferred transparent
    route. A C-contiguous complex128 vector's buffer is interleaved
    (re, im) float64 pairs, so B.view(np.float64).reshape(m, 2) is a
    zero-copy (m, 2) real matrix; one real matmul (n, m) x (m, 2) then a
    view of the C-contiguous (n, 2) result as complex128 gives Ar.dot(B)
    with no complex arithmetic and no 80 MB upcast of the matrix.
  - candidate "upcast_then_complex_gemv": the record's fallback route,
    upcast-then-complex-dot: Ar.astype(np.complex128) @ B, with the upcast
    INSIDE the timed call, so the measured speedup prices the cast the
    thread's derived 43.31 figure excluded (seberg's numbers warn this
    route can be cast-cost-dominated on some machines and even lose).
  - the reverse direction A.dot(Br) (complex left, real right), which the
    thread measured as roughly as fast as all-complex, as its own case
    with the same upcast route applied to the cheap operand (the vector).
    The record's route sketch only covers the real-left slow direction,
    and OPP-000027.md asks that both directions be measured before the
    predicate is designed; if this direction shows ~1x, the predicate
    needs only real-left.

Candidate routes never import pyoverdrive and never call np.dot or
ndarray.dot at all: the products inside candidates go through np.matmul
(the @ operator), which for these 2-D x 1-D shapes computes the identical
BLAS-backed product through a different numpy entry point, so a patched
np.dot cannot recurse into itself via a candidate.

What is NOT measured, and why:

  - The thread's full four-pairing table (Ar.dot(Br) and A.dot(B)) as
    separate suite cases: a Dyno case races a baseline against candidate
    routes, and the record names no alternative route for the two pure
    pairings (they are the fast cases). Their timings are bounded by the
    candidates anyway: view_as_real_gemv is ~2x the real-real GEMV work,
    and upcast_then_complex_gemv is cast + the complex-complex GEMV.
  - The size ladder, GEMM (matrix-matrix) shapes, float32/complex64
    pairings, and the BLAS-vs-fallback profiling that OPP-000027.md
    steps 2-3 call for: the corpus record's input_regime names exactly
    float64/complex128 at (10000, 1000) x (1000,), so the claim is
    re-established there; the wider sweep belongs to fast-path design,
    after this reproducer settles whether the gap still exists.

Correctness: results are complex128 vectors of length n. The candidates
reach the same componentwise arithmetic as stock (real times complex is
componentwise scaling) but through BLAS calls whose summation order can
differ from whatever loop stock numpy uses for the mixed case, the same
class of difference numpy already exhibits between its own BLAS and
non-BLAS paths. So the check is tolerance-equal, not bit-equal, at
ULP-scale headroom (rtol=1e-9, atol=1e-12) for ~1000-term float64 dot
products of standard normals: honest rounding differences sit around
1e-13 relative, while a genuinely wrong route (dropped imaginary part,
wrong interleave) is off by O(1) and fails hard.

Smoke mode shrinks the shape to (200, 100) x (100,) and drops samples to
3, purely to prove the harness end to end.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 10468
SMOKE = "--smoke" in sys.argv


def view_as_real_gemv(Ar, B):
    """Preferred route: reinterpret the contiguous complex128 vector as an
    (m, 2) float64 matrix, run one real matmul, view the (n, 2) result
    back as complex128. Zero copies of the large matrix, no complex
    arithmetic. Uses np.matmul, never np.dot (see module docstring)."""
    n = Ar.shape[0]
    Bv = B.view(np.float64).reshape(-1, 2)
    return (Ar @ Bv).view(np.complex128).reshape(n)


def upcast_then_complex_gemv(Ar, B):
    """Fallback route from the record: upcast the real matrix to
    complex128, then the complex product. The astype is inside the timed
    call on purpose: the thread's 43.31x excluded it, and seberg's
    timings show it can dominate."""
    return Ar.astype(np.complex128) @ B


def upcast_vector_gemv(A, Br):
    """Reverse-direction route: same upcast idea applied to the operand
    that is real there, the small vector, so the cast is cheap."""
    return A @ Br.astype(np.complex128)


def close_check(cand, base):
    """Tolerance-equal, not bit-equal: BLAS summation order in the
    candidate routes can differ from stock numpy's mixed-dtype path. The
    bound is ULP-scale for these ~1000-term float64 accumulations, not a
    semantic relaxation (see module docstring)."""
    return np.allclose(cand, base, rtol=1e-9, atol=1e-12)


if SMOKE:
    N, M = 200, 100
    samples = 3
else:
    N, M = 10_000, 1_000
    samples = 9

suite = BenchSuite(
    "OPP-000027",
    "real.dot(complex) GEMV vs view-as-real / upcast routes",
)

rng = np.random.default_rng(SEED)

# Built exactly as in the thread: complex A and B, real copies taken from
# their real parts.
A = rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))
B = rng.standard_normal(M) + 1j * rng.standard_normal(M)
Ar = A.real.copy()
Br = B.real.copy()

# The claim regime: real matrix .dot complex vector, the slow direction.
suite.measure(
    case=f"real_dot_complex_{N}x{M}",
    params={
        "left_dtype": "float64",
        "right_dtype": "complex128",
        "shape_left": [N, M],
        "shape_right": [M],
        "direction": "real_left_complex_right",
    },
    baseline=("numpy.dot", lambda: Ar.dot(B)),
    candidates={
        "view_as_real_gemv": lambda: view_as_real_gemv(Ar, B),
        "upcast_then_complex_gemv": lambda: upcast_then_complex_gemv(Ar, B),
    },
    check=close_check,
    samples=samples,
)

# The reverse direction, which the thread measured as fast (~ all-complex
# speed): measured so current numpy confirms the asymmetry, per
# OPP-000027.md's instruction to sweep both directions before designing
# the predicate.
suite.measure(
    case=f"complex_dot_real_{N}x{M}",
    params={
        "left_dtype": "complex128",
        "right_dtype": "float64",
        "shape_left": [N, M],
        "shape_right": [M],
        "direction": "complex_left_real_right",
    },
    baseline=("numpy.dot", lambda: A.dot(Br)),
    candidates={
        "upcast_vector_gemv": lambda: upcast_vector_gemv(A, Br),
    },
    check=close_check,
    samples=samples,
)

if not SMOKE:
    suite.save()
