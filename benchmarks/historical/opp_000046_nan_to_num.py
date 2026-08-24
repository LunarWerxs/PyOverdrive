"""OPP-000046: nan_to_num default-args vs copy + in-place copyto masks.

numpy/numpy#23140: stock nan_to_num chains np.where allocations. The
candidate replicates the full default semantics (NaN -> 0.0, +inf ->
f64 max, -inf -> f64 min, fresh copy) with in-place masking, skipping
the isinf work when no infinity exists.

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000046/.
Run: .venv/Scripts/python benchmarks/historical/opp_000046_nan_to_num.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000046", "nan_to_num vs copy + copyto masks")
rng = np.random.default_rng(23140)


def exact(c, b):
    return c.shape == b.shape and c.dtype == b.dtype and bool(np.array_equal(c, b))


def where_route(a):
    out = a.copy()
    np.copyto(out, 0.0, where=np.isnan(out))
    isinf = np.isinf(out)
    if isinf.any():
        info = np.finfo(out.dtype)
        np.copyto(out, info.max, where=isinf & (out > 0))
        np.copyto(out, info.min, where=isinf & (out < 0))
    return out


CASES = [
    ("nan_only", 10_000, 0.01, 0.0),
    ("nan_only", 1_000_000, 0.01, 0.0),
    ("nan_inf", 1_000_000, 0.01, 0.005),
    ("clean", 1_000_000, 0.0, 0.0),
]
if SMOKE:
    CASES = CASES[:1]

for tag, n, nan_frac, inf_frac in CASES:
    z = rng.standard_normal(n)
    if nan_frac:
        z[rng.random(n) < nan_frac] = np.nan
    if inf_frac:
        z[rng.random(n) < inf_frac] = np.inf
        z[rng.random(n) < inf_frac] = -np.inf
    suite.measure(
        case=f"{tag}_n{n}",
        params={"n": n, "nan_frac": nan_frac, "inf_frac": inf_frac},
        baseline=("nan_to_num", lambda z=z: np.nan_to_num(z)),
        candidates={"where_route": lambda z=z: where_route(z)},
        check=exact,
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
