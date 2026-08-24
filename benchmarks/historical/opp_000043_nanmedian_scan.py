"""OPP-000043: nanmedian 2-D vs one isnan scan + vectorized np.median.

numpy/numpy#4683 documents the per-slice-Python-loop mechanism class
(for np.ma.median; nanmedian shares it). For NaN-free 2-D input the
scan + plain median(axis=...) is bit-identical and skips the loop. The
anti-regime cell (few long slices) and the wasted-scan cell (NaNs
present) measure both losing directions honestly.

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000043/.
Run: .venv/Scripts/python benchmarks/historical/opp_000043_nanmedian_scan.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000043", "nanmedian 2-D vs isnan-scan + vectorized median")
rng = np.random.default_rng(4683)


def exact(c, b):
    c = np.asarray(c)
    b = np.asarray(b)
    return c.shape == b.shape and c.dtype == b.dtype and bool(
        np.array_equal(c, b, equal_nan=True)
    )


def scan_or(a, axis):
    if np.isnan(np.min(a)):
        return np.nanmedian(a, axis=axis)
    return np.median(a, axis=axis)


CASES = [
    ((500, 500), 1, 0.0),
    ((1000, 1000), 1, 0.0),
    ((2000, 200), 1, 0.0),
    ((200, 10_000), 1, 0.0),  # anti-regime: few long slices
    ((1000, 1000), 1, 0.01),  # wasted scan
]
if SMOKE:
    CASES = CASES[:1]

for shape, axis, nan_frac in CASES:
    a = rng.standard_normal(shape)
    if nan_frac:
        a[rng.random(shape) < nan_frac] = np.nan
    tag = "wasted" if nan_frac else "clean"
    suite.measure(
        case=f"nanmedian_{tag}_{shape[0]}x{shape[1]}",
        params={"shape": list(shape), "axis": axis, "nan_frac": nan_frac},
        baseline=("nanmedian", lambda a=a, axis=axis: np.nanmedian(a, axis=axis)),
        candidates={"scan_route": lambda a=a, axis=axis: scan_or(a, axis)},
        check=exact,
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
