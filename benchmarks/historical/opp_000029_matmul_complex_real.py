"""OPP-000029: complex-matrix times real-matrix matmul via two real GEMMs.

numpy/numpy#24565 (LemonBoy, 2023) proposes computing C @ R (C complex
MxN, R real NxQ) without promoting R to complex, "using one strided GEMM
call, exploiting the contiguous nature of the real and imaginary parts".
The thread contains NO numpy measurement at all - only a hedged aside
that MATLAB shows "a 2x difference" between C * R and the transpose
trick. So this reproducer ESTABLISHES the numpy numbers rather than
reproducing claimed ones; the mechanism to beat is stock's upcast (numpy
promotes R to complex128, a full copy, then runs a complex GEMM whose
multiply-accumulate costs ~4 real GEMM-equivalents where the split route
pays 2).

Candidates (both from first principles; the thread posts no code):

  - split_gemm: out.real = C.real @ R; out.imag = C.imag @ R, written
    into a preallocated complex output. C.real / C.imag are strided
    VIEWS (stride 2 floats) - BLAS GEMM consumes strided operands
    natively, so no copy of C is made and R is used as-is (the upcast
    copy stock pays disappears entirely).
  - split_gemm_reverse: the same identity for R @ C (real times complex,
    matrix-matrix), covering the direction OPP-000027 measured only for
    vectors.

Correctness: a complex GEMM and two real GEMMs accumulate in different
orders, so agreement is numeric (BLAS-rounding scale), not bit-exact:
check is allclose with rtol 1e-12 and atol scaled to the result
magnitude for complex128 (the dot_mixed_view precedent measured 2.7e-16
agreement for the same decomposition class), rtol 1e-5 for complex64.
Non-finite inputs are OUT of regime (the split changes inf/nan
propagation - measured and refused in the shipped dot_mixed_view path);
all inputs here are finite by construction, as the predicate would
require.

House rules: this script never imports pyoverdrive. The candidates call
np.matmul on real operands only (a mixed-dtype predicate refuses those),
so a patched dispatch could not recurse.

Result JSON: benchmarks/results/OPP-000029/.
Run: .venv/Scripts/python benchmarks/historical/opp_000029_matmul_complex_real.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 24565
SMOKE = "--smoke" in sys.argv


def split_gemm(c, r):
    out = np.empty((c.shape[0], r.shape[1]), dtype=c.dtype)
    np.matmul(c.real, r, out=out.real)
    np.matmul(c.imag, r, out=out.imag)
    return out


def split_gemm_reverse(r, c):
    out = np.empty((r.shape[0], c.shape[1]), dtype=c.dtype)
    np.matmul(r, c.real, out=out.real)
    np.matmul(r, c.imag, out=out.imag)
    return out


def make_check(rtol):
    def check(cand, base):
        if cand.dtype != base.dtype or cand.shape != base.shape:
            return False
        scale = max(1.0, float(np.abs(base).max()))
        return bool(np.allclose(cand, base, rtol=rtol, atol=rtol * scale))

    return check


CHECK = {"complex128": make_check(1e-12), "complex64": make_check(1e-5)}

suite = BenchSuite(
    "OPP-000029", "complex-by-real matmul: two real GEMMs vs stock upcast"
)
rng = np.random.default_rng(SEED)

# (m, n, q) shapes: square, tall-thin, and wide
if SMOKE:
    SHAPES = [(100, 100, 100)]
    DTYPES = [("complex128", np.complex128, np.float64)]
    SAMPLES = 3
else:
    SHAPES = [(200, 200, 200), (1000, 500, 300), (2000, 2000, 64), (64, 2000, 2000)]
    DTYPES = [
        ("complex128", np.complex128, np.float64),
        ("complex64", np.complex64, np.float32),
    ]
    SAMPLES = 9

for m, n, q in SHAPES:
    for label, cdt, rdt in DTYPES:
        c = (
            rng.uniform(0.5, 1.5, size=(m, n)) + 1j * rng.uniform(0.5, 1.5, size=(m, n))
        ).astype(cdt)
        r = rng.uniform(0.5, 1.5, size=(n, q)).astype(rdt)
        suite.measure(
            case=f"matmul_C{m}x{n}_R{n}x{q}_{label}",
            params={"m": m, "n": n, "q": q, "dtype": label, "direction": "C@R"},
            baseline=("numpy.matmul", lambda c=c, r=r: np.matmul(c, r)),
            candidates={"split_gemm": lambda c=c, r=r: split_gemm(c, r)},
            check=CHECK[label],
            samples=SAMPLES,
        )

# reverse direction R @ C at one representative shape per dtype
if not SMOKE:
    for label, cdt, rdt in DTYPES:
        c = (
            rng.uniform(0.5, 1.5, size=(500, 300))
            + 1j * rng.uniform(0.5, 1.5, size=(500, 300))
        ).astype(cdt)
        r = rng.uniform(0.5, 1.5, size=(1000, 500)).astype(rdt)
        suite.measure(
            case=f"matmul_R1000x500_C500x300_{label}_reverse",
            params={"m": 1000, "n": 500, "q": 300, "dtype": label, "direction": "R@C"},
            baseline=("numpy.matmul", lambda r=r, c=c: np.matmul(r, c)),
            candidates={"split_gemm_reverse": lambda r=r, c=c: split_gemm_reverse(r, c)},
            check=CHECK[label],
            samples=SAMPLES,
        )

    # np.dot surface, same regime (dot and matmul share the 2-D semantics)
    c = (
        rng.uniform(0.5, 1.5, size=(1000, 500))
        + 1j * rng.uniform(0.5, 1.5, size=(1000, 500))
    ).astype(np.complex128)
    r = rng.uniform(0.5, 1.5, size=(500, 300))
    suite.measure(
        case="dot_C1000x500_R500x300_complex128",
        params={"m": 1000, "n": 500, "q": 300, "dtype": "complex128", "direction": "C@R", "op": "dot"},
        baseline=("numpy.dot", lambda c=c, r=r: np.dot(c, r)),
        candidates={"split_gemm": lambda c=c, r=r: split_gemm(c, r)},
        check=CHECK["complex128"],
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
