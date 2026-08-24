"""percentile_dense calibration: np.percentile with a dense q array.

OPP-000022 shipped quantile_dense_sort (sort + numpy's exact lerp
arithmetic, 1.8-918x); np.percentile is documented as quantile with
q/100 and shares the same internal machinery, so the sibling path was
flagged in that record's roadmap notes. House law: a path ships only on
ITS OWN measured cells, so this battery runs the sort route against
stock np.percentile at a subset of the QUANTILE-CAL grid.

Candidate = the shipped quantile route verbatim, with q scaled by 1/100
exactly as numpy's percentile does before it hits the shared quantile
core; the check demands BIT-EXACT agreement (if numpy's own q/100 and
ours ever rounded differently, these cells would fail and the sibling
is dead).

Result JSON: benchmarks/results/PERCENTILE-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_percentile_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv


def sort_lerp_percentile(a, q, axis):
    q01 = np.true_divide(q, 100)  # exactly numpy's own scaling
    moved = np.moveaxis(a, axis, 0)
    n = moved.shape[0]
    srt = np.sort(moved, axis=0)
    virt = (n - 1) * q01
    prev_i = np.floor(virt).astype(np.intp)
    next_i = prev_i + 1
    above = virt >= n - 1
    prev_i[above] = -1
    next_i[above] = -1
    gamma = (virt - prev_i).reshape(virt.shape + (1,) * (srt.ndim - 1))
    below_v = srt[prev_i]
    above_v = srt[next_i]
    diff = above_v - below_v
    result = np.where(gamma >= 0.5, above_v - diff * (1 - gamma), below_v + diff * gamma)
    has_nan = np.isnan(srt[-1, ...])
    if has_nan.any():
        result[..., has_nan] = np.nan
    return result


def exact(c, b):
    return c.shape == b.shape and c.dtype == b.dtype and bool(
        np.array_equal(c, b, equal_nan=True)
    )


suite = BenchSuite("PERCENTILE-CAL", "sort+lerp vs stock percentile, dense q")
rng = np.random.default_rng(32187)

# (label, slices, m, nq)
if SMOKE:
    CASES = [("smoke", 8, 256, 64)]
    SAMPLES = 3
else:
    CASES = [
        ("smallq", 300, 2048, 4),
        ("smallq", 300, 2048, 64),
        ("dense", 300, 2048, 512),
        ("m512", 300, 512, 128),
        ("oned", None, 2048, 512),
        ("oned", None, 65536, 16384),
        ("fewslices", 4, 2048, 512),
    ]
    SAMPLES = 7

for label, slices, m, nq in CASES:
    shape = (m,) if slices is None else (slices, m)
    a = rng.standard_normal(shape)
    q = np.linspace(0.0, 100.0, nq)
    axis = 0 if slices is None else -1
    suite.measure(
        case=f"{label}_s{slices or 1}_m{m}_nq{nq}",
        params={"slices": slices or 1, "m": m, "nq": nq},
        baseline=(
            "numpy.percentile",
            lambda a=a, q=q, ax=axis: np.percentile(a, q, axis=ax, method="linear"),
        ),
        candidates={"sort_lerp": lambda a=a, q=q, ax=axis: sort_lerp_percentile(a, q, ax)},
        check=exact,
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
