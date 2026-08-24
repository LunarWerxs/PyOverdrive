"""OPP-000050: numpy.interp on a uniformly spaced grid, direct index vs
binary search.

numpy/numpy#2513 (Trac #1920) asked for a better-than-binary-search interp
back in 2012; upstream kept the general-grid searchsorted route. When xp
is uniform by construction (linspace/arange grids - the resampling
regime), the bin of every query is one subtract-multiply away, and the
whole lookup vectorizes with no per-element bisection at all. The
candidate route inlines the same uniform-grid guard the shipped
interp_uniform_grid fast path runs (every np.diff(xp) within 1e-9
relative of the first, plus isfinite checks on x and fp) so the guard
cost is part of what gets measured, not assumed away. Numeric comparison
(rtol 1e-9, scaled by numpy's own allclose).

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000050/.
Run: .venv/Scripts/python benchmarks/historical/opp_000050_interp_uniform.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000050", "uniform-grid numpy.interp direct index")
rng = np.random.default_rng(2513)

UNIFORM_RTOL = 1e-9  # mirrors interp_uniform_grid.UNIFORM_RTOL (copied, no import)


def close_scaled(rtol):
    # scaled ABSOLUTE tolerance: lerp outputs cross zero, where a
    # relative-only check explodes on rounding noise (a large full run
    # measured exactly that as a spurious CORRECTNESS-FAIL)
    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        if c.shape != b.shape or c.dtype != b.dtype:
            return False
        scale = float(np.abs(b).max()) if b.size else 1.0
        return bool((np.abs(c - b) <= rtol * max(1.0, scale)).all())

    return _chk


def interp_direct_index(x, xp, fp):
    """Uniform-grid guard plus direct-index lerp, mirroring the fast path."""
    d = np.diff(xp)
    dx0 = d[0]
    uniform = dx0 > 0 and bool((np.abs(d - dx0) <= UNIFORM_RTOL * dx0).all())
    finite = bool(np.isfinite(fp).all()) and bool(np.isfinite(x).all())
    if not (uniform and finite):
        return np.interp(x, xp, fp)
    dx = (xp[-1] - xp[0]) / (xp.size - 1)
    pos = (x - xp[0]) / dx
    idx = pos.astype(np.intp)
    np.clip(idx, 0, fp.size - 2, out=idx)
    frac = pos - idx
    lo = fp[idx]
    out = lo + frac * (fp[idx + 1] - lo)
    out[x <= xp[0]] = fp[0]
    out[x >= xp[-1]] = fp[-1]
    return out


def build(rng, nq, grid):
    xp = np.linspace(0.0, 1.0, grid).astype(np.float64)
    fp = (np.sin(xp * 7.0) + 0.1 * xp).astype(np.float64)
    x = rng.uniform(0.0, 1.0, nq).astype(np.float64)
    return x, xp, fp


CASES = [(10_000, 1_000)] if SMOKE else [(10_000, 1_000), (100_000, 1_000), (1_000_000, 10_000)]

for nq, grid in CASES:
    x, xp, fp = build(rng, nq, grid)
    suite.measure(
        case=f"interp_nq{nq}_grid{grid}",
        params={"nq": nq, "grid": grid},
        baseline=("interp", lambda x=x, xp=xp, fp=fp: np.interp(x, xp, fp)),
        candidates={
            "direct_index": lambda x=x, xp=xp, fp=fp: interp_direct_index(x, xp, fp)
        },
        check=close_scaled(1e-9),
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
