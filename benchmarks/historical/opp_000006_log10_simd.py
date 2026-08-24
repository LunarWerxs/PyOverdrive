"""OPP-000006: SLEEF SIMD transcendental prototype (numpy/numpy#23068).

Claim (2023-01, NumPy 1.23.3-dev on AArch64/A64FX SVE): combining a SLEEF-based
vectorized math library with SVE intrinsics (per companion PR #22265) made
np.log10 about 4x faster than the scalar libm fallback used on non-AVX512
platforms. The issue was never merged and mattip (maintainer) recommended
closing it, partly over vendoring cost and partly because NumPy was already
moving these functions to its own "universal intrinsics" instead.

This machine is x86-64, not AArch64/SVE, so it cannot reproduce the reported
SVE number at all. This reproducer is BASELINE_ONLY: it establishes where
NumPy's transcendental functions already have a SIMD path (AVX512 -> SVML on
x86-64) versus where they still fall back to scalar libm, by timing
np.log10, np.log, np.log2 and np.exp side by side. That split is the durable,
architecture-portable version of the underlying opportunity (unvectorized
transcendentals), independent of the disputed AArch64 prototype.

No candidates are measured (no PyOverdrive implementation exists yet); each
function is its own "case" and medians are meant to be compared to each other
in the report, not checked for numerical equality (they are different math
functions).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SEED = 23068

FUNCS = {
    "log10": np.log10,
    "log": np.log,
    "log2": np.log2,
    "exp": np.exp,
}

suite = BenchSuite("OPP-000006", "transcendental SIMD coverage: log10/log/log2/exp (#23068)")
rng = np.random.default_rng(SEED)

sizes = [1_000] if SMOKE else [10_000, 1_000_000]
dtypes = ["float64", "float32"]

for dtype_name in dtypes:
    dtype = np.dtype(dtype_name)
    for n in sizes:
        # positive, finite, well away from 0 so every function (incl. log*) is
        # defined everywhere in the array without special-casing domain edges.
        # Capped at 10 (not 100+) so exp(x) stays well inside float32 range
        # too (exp(10) ~ 2.2e4 vs a float32 max of ~3.4e38).
        data = (rng.random(n).astype(dtype) * 10.0 + 1e-3).astype(dtype)
        samples = 3 if SMOKE else (5 if n >= 1_000_000 else 11)
        for fname, fn in FUNCS.items():
            suite.measure(
                case=f"{fname}_{dtype_name}_n{n}",
                params={"dtype": dtype_name, "n": n, "function": fname},
                baseline=(f"numpy.{fname}", lambda a=data, f=fn: f(a)),
                candidates={},
                check=None,
                samples=samples,
            )

if not SMOKE:
    suite.save()
