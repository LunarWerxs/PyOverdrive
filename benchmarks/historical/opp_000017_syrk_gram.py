"""OPP-000017: BLAS herk fast path for a @ a.H (Hermitian rank-k) operations.

Source: docs/research/opportunities/OPP-000017.md, numpy/numpy#6951
("Optimization of `a @ a.H` style operations using BLAS herk routines").
lab/corpus/OPP-000017.yaml carries the same claim: baseline is the stock
gemm call `a @ a.conj().T` (equivalently `np.dot(a, a.conj().T)`), and the
candidate is a direct BLAS herk (Hermitian rank-k update) call, which only
has to compute one triangle of the provably-Hermitian result. The thread
supplies no benchmark numbers of its own (claim.speedup is null in the
corpus file) -- any speedup below comes from this run, not the issue.

Two deviations from the report's step-1 instructions, both forced by
feasibility and both documented here rather than silently applied:

1. Dropped square size 4096. The report names square sizes 64/256/1024/4096.
   Calibrating gemm on this machine (`a @ a.conj().T`, complex128) gave
   measured medians of n=256 ~1.05ms/call, n=1024 ~31.4ms/call -- this is a
   real BenchSuite run, not a number typed into prose: see
   benchmarks/micro/bench_syrk_gram_calibration.py and its saved evidence at
   benchmarks/results/OPP-000017-CAL/8f8198d9abab.json (machine was under
   64-66% foreign load at calibration time per that run's own contention
   warning, so these are conservative/noisy, not lab-quiet, numbers).
   n=4096 is 64x the flops of n=1024, so roughly 31.4ms * 64 =~ 2.0s/call;
   at 3 samples warmup + 7 measured samples, for 2 variants (baseline + herk)
   x 2 dtypes, that size alone would cost roughly 4 * 10 * 2.0s = 80s,
   consuming nearly the entire ~90s whole-suite budget by itself and leaving
   nothing for every other case in this file. Square sizes here are
   therefore {64, 256, 1024}.

2. Tall-and-skinny shape family uses `a.conj().T @ a` (the k x k Gram
   matrix) as its baseline, not the report's literal `a @ a.conj().T`. For
   tall a (e.g. 100000 x 256), `a @ a.conj().T` is the m x m product --
   100000 x 100000 complex128 is 1.6e11 bytes (~160 GB), not something any
   machine allocates for a benchmark. The report's own framing of *why* the
   tall-skinny family is in scope names the fix directly: "rileyjmurray
   (2021) wanted `a.H @ b` to avoid the extra allocation of
   `a.conjugate()`, i.e. a Gram/covariance-matrix computation on tall
   data" (report, "Is it common enough to matter"). `a.conj().T @ a` is
   exactly that Gram matrix, has the feasible k x k shape, and is the
   BLAS-herk trans='C' case, so it is what this file measures for the
   tall-skinny family instead of the literal (and here impossible) m x m
   call.

The herk candidate needs scipy (numpy does not expose herk/syrk directly).
scipy is not installed in this repo's .venv, so `HAVE_SCIPY_BLAS` is False
here and the herk candidate is skipped for every case with a printed note,
never attempted -- per the reproducer contract this is a probe-and-skip,
not a crash. The herk call sites below (trans=0 i.e. 'N' for a @ a.H,
trans=2 i.e. 'C' for a.H @ a, per scipy.linalg.blas cherk/zherk's
documented convention) have NOT been exercised against a real BLAS call in
this environment; treat the printed CORRECT / CORRECTNESS-FAIL for the herk
variant as the real verification wherever this is run with scipy present,
not this docstring.

samples: 7 for the real run (within the house 5-11 band), 3 in --smoke.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 6951  # the numpy issue number this opportunity comes from
SMOKE = "--smoke" in sys.argv

try:
    from scipy.linalg.blas import cherk, zherk

    HAVE_SCIPY_BLAS = True
except ImportError:
    HAVE_SCIPY_BLAS = False
    cherk = zherk = None
    print("[opp-000017] scipy not importable: herk candidate will be skipped for every case")


def make_complex(rng, shape, dtype):
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    re = rng.standard_normal(shape).astype(real_dtype)
    im = rng.standard_normal(shape).astype(real_dtype)
    return (re + 1j * im).astype(dtype)


def baseline_square(a):
    return a @ a.conj().T


def baseline_gram(a):
    return a.conj().T @ a


def herk_fn_for(dtype):
    return cherk if dtype == np.complex64 else zherk


def mirror_upper(c, n, dtype):
    """Build the full dense Hermitian matrix from herk's upper triangle.

    herk (lower=0) only guarantees the upper triangle (diagonal included) of
    `c` is meaningful; the strict lower triangle is not read here at all, it
    is reconstructed as the conjugate of the strict upper triangle, which is
    what Hermitian symmetry requires -- avoids depending on whatever the
    BLAS call happened to leave in the unused half of its output buffer.
    """
    full = np.zeros((n, n), dtype=dtype)
    iu_all = np.triu_indices(n)
    full[iu_all] = c[iu_all]
    iu_strict = np.triu_indices(n, k=1)
    il_strict = (iu_strict[1], iu_strict[0])
    full[il_strict] = np.conj(c[iu_strict])
    return full


def herk_square_candidate(a, herk_fn):
    n = a.shape[0]
    c = herk_fn(alpha=1.0, a=a, trans=0, lower=0)  # C := a @ a.conj().T
    return mirror_upper(c, n, a.dtype)


def herk_gram_candidate(a, herk_fn):
    k = a.shape[1]
    c = herk_fn(alpha=1.0, a=a, trans=2, lower=0)  # C := a.conj().T @ a
    return mirror_upper(c, k, a.dtype)


def gram_check(dtype):
    """allclose, not bit-identity: herk sums the same k products as gemm but
    in a different accumulation order (report, "What semantics differ"), so
    results differ only in rounding proportional to machine epsilon and to
    how many terms were summed. rtol/atol both set to 50x the dtype's eps --
    honest for complex64 (eps ~1.2e-7) and complex128 (eps ~2.2e-16) alike,
    not a number pulled from nowhere.
    """
    tol = np.finfo(dtype).eps * 50

    def _check(cand, base):
        if cand.shape != base.shape:
            return False
        return np.allclose(cand, base, rtol=tol, atol=tol)

    return _check


if SMOKE:
    SQUARE_SIZES = [32]
    TALL_SHAPES = [(2_000, 16)]
    DTYPES = [np.complex64, np.complex128]
    SAMPLES = 3
else:
    SQUARE_SIZES = [64, 256, 1024]  # 4096 dropped, see module docstring
    TALL_SHAPES = [(100_000, 64), (100_000, 256)]
    DTYPES = [np.complex64, np.complex128]
    SAMPLES = 7

suite = BenchSuite("OPP-000017", "a @ a.H / a.H @ a: stock gemm vs BLAS herk fast path")
rng = np.random.default_rng(SEED)

for dtype in DTYPES:
    dtype_name = np.dtype(dtype).name
    herk_fn = herk_fn_for(dtype)

    for n in SQUARE_SIZES:
        a = make_complex(rng, (n, n), dtype)
        candidates = {}
        if HAVE_SCIPY_BLAS:
            candidates["herk"] = lambda a=a, herk_fn=herk_fn: herk_square_candidate(a, herk_fn)
        else:
            print(f"[opp-000017] square_n{n}_{dtype_name}: herk candidate skipped (no scipy)")

        suite.measure(
            case=f"square_n{n}_{dtype_name}",
            params={"dtype": dtype_name, "n": n, "shape": f"{n}x{n}"},
            baseline=("a @ a.conj().T", lambda a=a: baseline_square(a)),
            candidates=candidates,
            check=gram_check(dtype),
            samples=SAMPLES,
        )

    for m, k in TALL_SHAPES:
        a = make_complex(rng, (m, k), dtype)
        candidates = {}
        if HAVE_SCIPY_BLAS:
            candidates["herk"] = lambda a=a, herk_fn=herk_fn: herk_gram_candidate(a, herk_fn)
        else:
            print(f"[opp-000017] tall_m{m}_k{k}_{dtype_name}: herk candidate skipped (no scipy)")

        suite.measure(
            case=f"tall_m{m}_k{k}_{dtype_name}",
            params={"dtype": dtype_name, "m": m, "k": k, "shape": f"{m}x{k}"},
            baseline=("a.conj().T @ a", lambda a=a: baseline_gram(a)),
            candidates=candidates,
            check=gram_check(dtype),
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
