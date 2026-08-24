"""OPP-000041: nan-aggregation wrappers vs one isnan scan + plain reduction.

numpy/numpy#5691 (2015) documented the class (nanmax ~3x slower than
max); modern numpy fixed nanmax/nanmin but nanmean/nansum/nanstd/nanvar
retain the Python-wrapper mask-and-copy overhead on NaN-free input. The
pydata/bottleneck README table (numpy 2.4.2) shows the same class at
(100,) axis=0: nansum 12.2x, nanmean 29.8x, nanstd 34.2x vs their C.

Candidate: np.isnan(np.min(a)) probe (min propagates NaN), then the
plain reduction; NaN present falls back to the nan-wrapper, so the
wasted-scan cell measures pure guard cost. Check demands bit-identical
results (equal_nan for the NaN cells).

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000041/.
Run: .venv/Scripts/python benchmarks/historical/opp_000041_nan_reductions.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000041", "nan-aggregations vs isnan-scan + plain reduction")
rng = np.random.default_rng(5691)


def exact(c, b):
    c = np.asarray(c)
    b = np.asarray(b)
    return c.shape == b.shape and c.dtype == b.dtype and bool(
        np.array_equal(c, b, equal_nan=True)
    )


def scan_or(fast, fallback, a, **kw):
    if np.isnan(np.min(a)):
        return fallback(a, **kw)
    return fast(a, **kw)


CASES = [
    ("nanmean", np.nanmean, np.mean, 100, None, 0.0),
    ("nanmean", np.nanmean, np.mean, 1_000, None, 0.0),
    ("nanmean", np.nanmean, np.mean, 100_000, None, 0.0),
    ("nanmean", np.nanmean, np.mean, (1000, 100), 1, 0.0),
    ("nanmean_wasted", np.nanmean, np.mean, 100_000, None, 0.01),
    ("nansum", np.nansum, np.sum, 10_000, None, 0.0),
    ("nansum", np.nansum, np.sum, 100_000, None, 0.0),
    ("nanstd", np.nanstd, np.std, 3_000, None, 0.0),
    ("nanstd", np.nanstd, np.std, (1000, 100), 1, 0.0),
    ("nanvar", np.nanvar, np.var, (1000, 100), 1, 0.0),
]
if SMOKE:
    CASES = CASES[:2]

for label, nanfn, fastfn, shape, axis, nan_frac in CASES:
    a = rng.standard_normal(shape)
    if nan_frac:
        a[rng.random(np.shape(a)) < nan_frac] = np.nan
    kw = {} if axis is None else {"axis": axis}
    suite.measure(
        case=f"{label}_n{a.size}",
        params={"shape": list(np.shape(a)), "axis": axis, "nan_frac": nan_frac},
        baseline=(nanfn.__name__, lambda a=a, kw=kw, f=nanfn: f(a, **kw)),
        candidates={
            "scan_route": lambda a=a, kw=kw, f=fastfn, g=nanfn: scan_or(f, g, a, **kw)
        },
        check=exact,
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
