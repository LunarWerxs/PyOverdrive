"""OPP-000017 sizing calibration: gemm cost for `a @ a.conj().T`, complex128.

Exists solely to substantiate the size-4096-dropped justification in
benchmarks/historical/opp_000017_syrk_gram.py's module docstring with a real
measured artifact instead of a number typed into prose. Baseline-only (no
candidate): this is sizing input, not an opportunity result. Uses the same
BenchSuite machinery as every other reproducer, so setup (array construction)
is untimed and the timed region is exactly `a @ a.conj().T`.

Result JSON: benchmarks/results/OPP-000017-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_syrk_gram_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 6951  # same seed as opp_000017_syrk_gram.py, for consistency
SMOKE = "--smoke" in sys.argv
SIZES = [256] if SMOKE else [256, 1024]
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000017-CAL", "gemm sizing calibration for a @ a.conj().T, complex128")
rng = np.random.default_rng(SEED)

for n in SIZES:
    re = rng.standard_normal((n, n)).astype(np.float64)
    im = rng.standard_normal((n, n)).astype(np.float64)
    a = (re + 1j * im).astype(np.complex128)

    suite.measure(
        case=f"square_n{n}_complex128",
        params={"dtype": "complex128", "n": n, "shape": f"{n}x{n}"},
        baseline=("a @ a.conj().T", lambda a=a: a @ a.conj().T),
        candidates={},
        check=lambda cand, base: True,  # baseline-only sizing probe, nothing to compare
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
