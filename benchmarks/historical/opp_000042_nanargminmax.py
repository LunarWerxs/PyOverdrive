"""OPP-000042: nanargmax/nanargmin vs one isnan scan + plain argmax/argmin.

The wrappers build a where(isnan, -inf) masked COPY before reducing even
when no NaN exists. pydata/bottleneck's README table (numpy 2.4.2) shows
nanargmax 26.0x at (100,) axis=0 vs their C; the pure-numpy scan route
recovers a large share of that for NaN-free input, which plain argmax
answers bit-identically (same reduction, same first-occurrence ties).

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000042/.
Run: .venv/Scripts/python benchmarks/historical/opp_000042_nanargminmax.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000042", "nanargmax/nanargmin vs isnan-scan + plain arg-reduction")
rng = np.random.default_rng(20260824)


def exact(c, b):
    c = np.asarray(c)
    b = np.asarray(b)
    return c.shape == b.shape and c.dtype == b.dtype and bool(np.array_equal(c, b))


def scan_or(fast, fallback, a, **kw):
    if np.isnan(np.min(a)):
        return fallback(a, **kw)
    return fast(a, **kw)


CASES = [
    ("nanargmax", np.nanargmax, np.argmax, 300, None, 0.0),
    ("nanargmax", np.nanargmax, np.argmax, 10_000, None, 0.0),
    ("nanargmax", np.nanargmax, np.argmax, 100_000, None, 0.0),
    ("nanargmax", np.nanargmax, np.argmax, (1000, 1000), 1, 0.0),
    ("nanargmin", np.nanargmin, np.argmin, 100_000, None, 0.0),
    ("nanargmax_wasted", np.nanargmax, np.argmax, 100_000, None, 0.01),
]
if SMOKE:
    CASES = CASES[:1]

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
