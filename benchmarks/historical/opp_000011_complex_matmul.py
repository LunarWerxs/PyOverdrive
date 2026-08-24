"""OPP-000011: 3-multiply (Gauss) trick for complex128/complex64 matmul.

numpy/numpy#13229 ("Complex matmul is very slow (and here's a 2x speedup)")
claims that decomposing complex A @ B into three real matmuls (the
classical Gauss/Karatsuba trick) instead of the naive four-real-matmul
complex expansion saves 25% of the floating-point multiplications and, on
the reporter's numpy 1.13.1 build, ran (2048,4096,2048) complex128 matmul
2.21x faster in wall time (7.45s -> 3.37s). The reporter's own follow-up
measurement on numpy 1.16.2, identical shape, narrowed the gap to 1.09x
(2.28s -> 2.09s), and mattip's opening question ("Does this replicate on a
more recent numpy? matmul was rewritten as a gufunc.") was never answered
in the thread as retrieved. This reproducer measures both questions on
current numpy:

  - baseline: A @ B (numpy.matmul), freshly generated complex128 and
    complex64 matrices, no hints.
  - candidate "gauss3": C1 = A.real @ B.real; C2 = A.imag @ B.imag;
    C3 = (A.real + A.imag) @ (B.real + B.imag);
    C = (C1 - C2) + 1j*(C3 - C1 - C2), hand-implemented per the issue's
    own description. No numpy-internal or scipy fast path implements this,
    so there is nothing to probe-import here.

Sizes: OPP-000011.md step 1 names a square sweep of 256/512/1024/2048/4096
plus the reporter's own rectangular (m,n,k) = (2048,4096,2048) shape. All
of it is kept: a direct timing probe on this dev machine (scipy-openblas
0.3.31.188.0, SkylakeX, before this file existed, not part of any timed
suite run) measured stock complex128 matmul at n=2048 in ~0.10s and the
gauss3 candidate at ~0.13s, so the full sweep at 7 samples/case comfortably
clears the ~90s house budget on this machine's BLAS; a reference-BLAS
build would need the largest sizes trimmed first.

Correctness: gauss3 is NOT bit-identical to stock complex matmul (the
corpus record's "What semantics differ": the extra C3 - C1 - C2
subtraction is additional rounding-error-introducing arithmetic direct
complex multiply-accumulate does not have), so results are compared with
the reporter's own methodology from the issue -
norm(candidate - baseline) / norm(baseline) against a tolerance derived
from a Wilkinson-style sqrt(k)*eps forward-error bound for a k-term matmul
sum, padded 100x for gauss3's one extra subtraction (see make_checker).
A matrix-level relative-norm check is used rather than elementwise
allclose deliberately: elementwise comparison spuriously "fails" on
individual output entries that are near-zero because of catastrophic
cancellation (exactly the magnitude-skewed regime below) even when the
matrix as a whole is accurate to near-eps; the reporter's own single
correctness sample used this same norm-ratio form.

Two extra small, cheap correctness-only cases probe the record's step-2
concern directly: a magnitude-skewed case (real parts ~1e4x the imaginary
parts, where the C3-C1-C2 cancellation is expected to be least precise -
see the long comment at edge_magnitude_skew below for why the matrix-norm
check can still pass there) and a near-zero-components case. NaN/Inf
inputs are deliberately NOT exercised in this suite's pass/fail gate: the
corpus record flags their handling as an open question the trick may
genuinely answer differently from stock (a different cancellation path,
not a bug), so gating this reproducer's exit code on them would conflate
"known, documented semantic difference" with "broken candidate"; that
audit is left as explicit follow-up per the record's step 5, not swept
here.

mattip's and seberg's open questions are tracked via case naming rather
than a separate report: the square cases at n=2048/4096 and the
rect_2048x4096x2048_reporter_shape case are the direct "does this still
replicate post-gufunc-rewrite, and at the reporter's own out-of-cache
size" answer - read their printed speedups against the 2.21x (numpy
1.13.1) and 1.09x (numpy 1.16.2) historical figures in OPP-000011.md.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 13229
SMOKE = "--smoke" in sys.argv


def gauss3_matmul(A, B):
    """3-multiply (Gauss/Karatsuba) complex matmul: trade one of the four
    real matmuls the naive complex expansion needs for O(mn+nk+mk)
    elementwise adds/subtracts. dtype-preserving (complex64 in -> complex64
    out) so the result compares like-for-like against stock.
    """
    Ar, Ai = A.real, A.imag
    Br, Bi = B.real, B.imag
    C1 = Ar @ Br
    C2 = Ai @ Bi
    C3 = (Ar + Ai) @ (Br + Bi)
    out = np.empty(C1.shape, dtype=A.dtype)
    out.real = C1 - C2
    out.imag = C3 - C1 - C2
    return out


def make_complex(rng, shape, dtype, scale=1.0):
    real_dtype = np.float64 if dtype == np.complex128 else np.float32
    real = (rng.standard_normal(shape) * scale).astype(real_dtype)
    imag = (rng.standard_normal(shape) * scale).astype(real_dtype)
    out = np.empty(shape, dtype=dtype)
    out.real = real
    out.imag = imag
    return out


def make_checker(dtype, k):
    """Relative-Frobenius-norm checker: norm(got - expected)/norm(expected)
    against a Wilkinson-style sqrt(k)*eps bound, padded 100x for gauss3's
    extra subtraction (see module docstring "Correctness" for the full
    justification). Floors: 1e-10 for complex128, 1e-6 for complex64, so
    tiny k does not produce an unrealistically tight bound.
    """
    eps = np.finfo(dtype).eps
    floor = 1e-10 if dtype == np.complex128 else 1e-6
    rel_tol = max(100 * np.sqrt(max(k, 1)) * eps, floor)

    def check(candidate, baseline):
        denom = np.linalg.norm(baseline)
        if denom == 0:
            return np.linalg.norm(candidate) <= rel_tol
        return np.linalg.norm(candidate - baseline) / denom <= rel_tol

    return check


rng = np.random.default_rng(SEED)

if SMOKE:
    square_sizes = [32, 64]
    samples = 3
    edge_n = 16
else:
    square_sizes = [256, 512, 1024, 2048, 4096]
    samples = 7
    edge_n = 64

suite = BenchSuite("OPP-000011", "3-multiply (Gauss) trick vs stock complex matmul")

for dtype in (np.complex128, np.complex64):
    dtype_name = np.dtype(dtype).name
    for n in square_sizes:
        A = make_complex(rng, (n, n), dtype)
        B = make_complex(rng, (n, n), dtype)
        suite.measure(
            case=f"square_{n}_{dtype_name}",
            params={"dtype": dtype_name, "m": n, "n": n, "k": n},
            baseline=("numpy.matmul", lambda A=A, B=B: A @ B),
            candidates={"gauss3": lambda A=A, B=B: gauss3_matmul(A, B)},
            check=make_checker(dtype, n),
            samples=samples,
        )

# Rectangular bracket matching the reporter's own (m,n,k) = (2048,4096,2048)
# shape (complex128 only; shrunk in --smoke to keep the quick path tiny).
m, n, k = (24, 40, 16) if SMOKE else (2048, 4096, 2048)
label = "smoke" if SMOKE else "reporter_shape"
A = make_complex(rng, (m, n), np.complex128)
B = make_complex(rng, (n, k), np.complex128)
suite.measure(
    case=f"rect_{m}x{n}x{k}_{label}_complex128",
    params={"dtype": "complex128", "m": m, "n": n, "k": k},
    baseline=("numpy.matmul", lambda A=A, B=B: A @ B),
    candidates={"gauss3": lambda A=A, B=B: gauss3_matmul(A, B)},
    check=make_checker(np.complex128, n),
    samples=samples,
)

# edge_magnitude_skew: real parts ~1e4x the imaginary parts on both
# operands, the regime OPP-000011.md's "What semantics differ" flags as
# where gauss3's extra C3-C1-C2 subtraction is expected to lose the most
# relative precision (it subtracts two O(real*real)-magnitude quantities
# to recover an O(real*imag)-magnitude result). The matrix-norm check
# above can still pass here even though individual small-magnitude output
# entries carry larger relative error, because the huge real-part entries
# dominate the norm; that is a known limitation of a norm-level check
# (the reporter's own single-sample check shares it) and is why this case
# exists as a documented, not silently assumed, correctness probe rather
# than being folded into the main sweep's uniform-random inputs.
A = make_complex(rng, (edge_n, edge_n), np.complex128, scale=1.0)
A.real *= 1e4
B = make_complex(rng, (edge_n, edge_n), np.complex128, scale=1.0)
B.real *= 1e4
suite.measure(
    case=f"edge_magnitude_skew_{edge_n}_complex128",
    params={"dtype": "complex128", "m": edge_n, "n": edge_n, "k": edge_n, "regime": "real 1e4x imag"},
    baseline=("numpy.matmul", lambda A=A, B=B: A @ B),
    candidates={"gauss3": lambda A=A, B=B: gauss3_matmul(A, B)},
    check=make_checker(np.complex128, edge_n),
    samples=samples,
)

# edge_near_zero: both components pinned near the small-magnitude end of
# the normal float range (no underflow), checking gauss3's extra
# subtraction under otherwise-ordinary but tiny values.
A = make_complex(rng, (edge_n, edge_n), np.complex128, scale=1e-8)
B = make_complex(rng, (edge_n, edge_n), np.complex128, scale=1e-8)
suite.measure(
    case=f"edge_near_zero_{edge_n}_complex128",
    params={"dtype": "complex128", "m": edge_n, "n": edge_n, "k": edge_n, "regime": "near-zero ~1e-8"},
    baseline=("numpy.matmul", lambda A=A, B=B: A @ B),
    candidates={"gauss3": lambda A=A, B=B: gauss3_matmul(A, B)},
    check=make_checker(np.complex128, edge_n),
    samples=samples,
)

if not SMOKE:
    suite.save()
