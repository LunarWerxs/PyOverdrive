"""OPP-000021: opt-in post-hoc covariance to avoid the m - mean(m) copy in
np.cov for n_samples >> n_features.

numpy/numpy#31292 proposes replacing np.cov's centered two-pass computation
(which allocates a full m-sized `m - mean(m)` copy before the matmul) with
the algebraically equivalent post-hoc formula already used by scikit-learn's
PCA(svd_solver="covariance_eigh"):

    cov = (m @ m.conj().T) / fact - (N / fact) * np.outer(avg, avg.conj())

The reporter measured 136 ms / 640 MB peak (stock np.cov) versus 69 ms /
~0 MB peak (post-hoc) on a (16, 5_000_000) float64 array: a 1.97x wall-clock
improvement and near-total elimination of the centering copy. This
reproducer measures:

  - baseline: numpy.cov(m) (stock, no keywords)
  - candidate "post_hoc": the formula above, implemented from the report's
    description, no numpy internals, no pyoverdrive import

Per docs/research/opportunities/OPP-000021.md step 1, this sweeps
n_features in {2, 16, 64}, n_samples across a size ladder, dtypes float32
and float64, and both well-conditioned (mean 0, std 1) and deliberately
ill-conditioned (mean/std, i.e. mu/sigma, swept through the reporter's
claimed float32 (~10) and float64 (~1e4) safety thresholds) data, to check
the speed numbers and the precision cliff independently.

Deviations from step 1, and why (house rule: follow the spirit, say so
here):

  - n_samples is capped at 1_000_000 for the timed BenchSuite sweep rather
    than swept up to 5_000_000, to keep the whole non-smoke battery well
    under the ~90s budget (each timed case runs warmup + calibration +
    several samples for BOTH baseline and candidate, and stock np.cov's
    centering copy is itself allocation-bound at these sizes). The report's
    exact (16, 5_000_000) float64 shape is instead reproduced once,
    untimed-by-BenchSuite, in the REPORTED_SHAPE_CHECK block below (a
    single call per variant, timed with a plain wall clock and profiled
    with tracemalloc), so the shape that anchors the reporter's headline
    numbers is still exercised on this machine.
  - Dyno's BenchSuite only times wall clock; it has no built-in peak-memory
    instrumentation, and the report's headline claim is a MEMORY claim
    (640 MB -> ~0 MB), not primarily a speed one. Peak memory is measured
    separately with tracemalloc, outside BenchSuite, at one representative
    size per dtype (informational prints, not part of suite.cases/save()).
  - The precision-cliff sweep (mu/sigma through the safety thresholds) is
    also run outside BenchSuite's check= gate: on ill-conditioned data the
    post-hoc candidate is EXPECTED to diverge from stock np.cov, and Dyno's
    check= exists to gate "is this candidate a correct drop-in replacement"
    for the timed benchmark, not to log an expected, size-independent
    precision cliff. Gating that as a normal correctness check would make a
    known, reported degradation look like a benchmark defect. The timed
    BenchSuite cases below therefore use ONLY well-conditioned data (mean
    0, std 1), where the reporter's own bound is ~2e-14 max absolute
    deviation for float64.

The post-hoc candidate calls only ndarray.mean, np.outer, and @ (matmul) on
the raw input array; it never calls numpy.cov itself, so nothing here is
circular against a future numpy.cov patch.
"""

import sys
import time
import tracemalloc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 31292
SMOKE = "--smoke" in sys.argv


def post_hoc_cov(m):
    """Post-hoc covariance, per numpy/numpy#31292 and the OPP-000021 corpus
    entry: cov = (m @ m.conj().T) / fact - (N / fact) * outer(avg, avg.conj()).

    fact = N - 1 matches np.cov's default (bias=False, ddof=None -> ddof=1).
    Avoids ever materializing an m-sized centered copy.
    """
    n_features, n = m.shape
    fact = n - 1
    avg = m.mean(axis=1)
    return (m @ m.conj().T) / fact - (n / fact) * np.outer(avg, avg.conj())


def make_check(dtype):
    """float64: tight tolerance, matmul/mean rounding only, well-conditioned
    data (mean 0, std 1) puts the reporter's own bound (~2e-14 max abs
    deviation) far inside this.
    float32: looser tolerance. float32 carries ~7 decimal digits and the
    post-hoc path subtracts two similarly-scaled quantities (a classic
    catastrophic-cancellation shape), so it accumulates more relative error
    than float64 even on well-conditioned data; the bound below is still
    tight enough that a genuine algorithm bug would fail it.
    """
    if dtype == np.float64:
        return lambda cand, base: np.allclose(cand, base, rtol=1e-9, atol=1e-9)
    return lambda cand, base: np.allclose(cand, base, rtol=1e-3, atol=1e-5)


def make_data(rng, n_features, n_samples, dtype, mean=0.0, std=1.0):
    return (rng.standard_normal((n_features, n_samples)) * std + mean).astype(dtype)


if SMOKE:
    n_features_list = [2, 16, 64]
    n_samples_list = [100, 1_000]
    dtypes = [np.float32, np.float64]
    samples = 3
else:
    n_features_list = [2, 16, 64]
    n_samples_list = [10_000, 100_000, 1_000_000]
    dtypes = [np.float32, np.float64]
    samples = 7

suite = BenchSuite("OPP-000021", "np.cov: stock centered path vs post-hoc BLAS-gemm path")
rng = np.random.default_rng(SEED)

# --- Timed sweep: well-conditioned data only (see docstring for why). ---
for n_features in n_features_list:
    for n_samples in n_samples_list:
        for dtype in dtypes:
            dtype_name = np.dtype(dtype).name
            m = make_data(rng, n_features, n_samples, dtype)
            case = f"n{n_features}_m{n_samples}_{dtype_name}"
            params = {
                "dtype": dtype_name,
                "n_features": n_features,
                "n_samples": n_samples,
                "conditioning": "well-conditioned (mean=0, std=1)",
            }
            suite.measure(
                case=case,
                params=params,
                baseline=("numpy.cov", lambda m=m: np.cov(m)),
                candidates={"post_hoc": lambda m=m: post_hoc_cov(m)},
                check=make_check(dtype),
                samples=samples,
            )

# --- Precision-cliff diagnostic: informational, not gated by BenchSuite. ---
# Fixed moderate shape (precision is size-independent per the report); sweep
# mu/sigma across the reporter's claimed float32 (~10) and float64 (~1e4)
# safety thresholds, on both dtypes, to see where the divergence appears.
print("\n[precision-cliff] mu/sigma sweep, fixed shape n_features=16, "
      f"n_samples={100 if SMOKE else 50_000} (informational, not a pass/fail check)")
cliff_n_samples = 100 if SMOKE else 50_000
ratios = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000] if not SMOKE else [1, 100, 10_000]
for dtype in dtypes:
    dtype_name = np.dtype(dtype).name
    for ratio in ratios:
        m = make_data(rng, 16, cliff_n_samples, dtype, mean=float(ratio), std=1.0)
        base = np.cov(m)
        cand = post_hoc_cov(m)
        base_scale = np.max(np.abs(base))
        max_abs_dev = float(np.max(np.abs(cand - base)))
        rel_dev = max_abs_dev / base_scale if base_scale > 0 else float("nan")
        print(f"  {dtype_name} mu/sigma={ratio:<10g} max_abs_dev={max_abs_dev:.3e} "
              f"rel_to_baseline_scale={rel_dev:.3e}")

# --- Reported-shape check: reproduce the report's exact (16, 5_000_000)
# float64 shape once, untimed by BenchSuite, for wall time + peak memory. ---
if SMOKE:
    print("\n[reported-shape] skipped under --smoke "
          "(16 x 5_000_000 float64 is not a smoke-sized allocation)")
else:
    print("\n[reported-shape] n_features=16, n_samples=5_000_000, float64, "
          "well-conditioned (single call each, wall clock + tracemalloc)")
    m_big = make_data(rng, 16, 5_000_000, np.float64)

    tracemalloc.start()
    t0 = time.perf_counter()
    base_big = np.cov(m_big)
    t_base = time.perf_counter() - t0
    _, peak_base = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    cand_big = post_hoc_cov(m_big)
    t_cand = time.perf_counter() - t0
    _, peak_cand = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ok_big = np.allclose(cand_big, base_big, rtol=1e-9, atol=1e-9)
    print(f"  numpy.cov:  {t_base * 1e3:.1f} ms, tracemalloc peak {peak_base / 1e6:.1f} MB")
    print(f"  post_hoc:   {t_cand * 1e3:.1f} ms, tracemalloc peak {peak_cand / 1e6:.1f} MB "
          f"({'CORRECT' if ok_big else 'CORRECTNESS-FAIL'})")

if not SMOKE:
    suite.save()
