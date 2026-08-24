"""quantile_dense_sort calibration: the regime edges OPP-000022 left open.

The OPP-000022 reproducer measured the cliff at one reduced length
(m=2048, 300 slices, axis=-1): wins 4.3x from nq=102 and 34-44x past the
nq ~= m/4 cliff. A dispatch predicate needs the edges:

- reduced-length sweep: does nq/m >= 0.05 still win at m = 512 / 8192 /
  65536? (Sorting is n log n; partition is ~n per kth batch - the
  crossover could move.)
- small-nq witnesses: np.quantile(a, [0.25, 0.5, 0.75]) style calls must
  stay on stock - measure nq in {4, 16, 64} to see where the win dies.
- 1-D input (a single slice): the most common call shape of all, absent
  from the reproducer.
- few-slices (4) vs many-slices (300).

Candidate = the reproducer's sort_lerp_quantile, verbatim (bit-exact vs
stock method='linear' including NaN propagation - exact_equal checks).

Result JSON: benchmarks/results/QUANTILE-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_quantile_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv


def sort_lerp_quantile(a, q, axis):
    moved = np.moveaxis(a, axis, 0)
    n = moved.shape[0]
    srt = np.sort(moved, axis=0)
    virt = (n - 1) * q
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
    return c.shape == b.shape and c.dtype == b.dtype and bool(np.array_equal(c, b, equal_nan=True))


suite = BenchSuite("QUANTILE-CAL", "sort+lerp vs stock quantile: nq/m regime edges")
rng = np.random.default_rng(32187)

# (label, slices, m, nq); slices None = 1-D input
if SMOKE:
    CASES = [("smoke", 8, 256, 64)]
    SAMPLES = 3
else:
    CASES = [
        ("smallq", 300, 2048, 4),
        ("smallq", 300, 2048, 16),
        ("smallq", 300, 2048, 64),
        ("m512", 300, 512, 32),
        ("m512", 300, 512, 128),
        ("m8192", 300, 8192, 410),
        ("m8192", 300, 8192, 2048),
        ("m65536", 32, 65536, 3277),
        ("m65536", 32, 65536, 16384),
        ("oned", None, 2048, 102),
        ("oned", None, 2048, 512),
        ("oned", None, 65536, 3277),
        ("oned", None, 65536, 16384),
        ("oned_smallq", None, 65536, 16),
        ("fewslices", 4, 2048, 512),
    ]
    SAMPLES = 7

for label, slices, m, nq in CASES:
    shape = (m,) if slices is None else (slices, m)
    a = rng.standard_normal(shape)
    q = np.linspace(0.0, 1.0, nq)
    axis = 0 if slices is None else -1
    samples = SAMPLES if slices != 300 or m <= 2048 else max(3, SAMPLES - 2)
    suite.measure(
        case=f"{label}_s{slices or 1}_m{m}_nq{nq}",
        params={"slices": slices or 1, "m": m, "nq": nq, "ratio": round(nq / m, 4)},
        baseline=("numpy.quantile", lambda a=a, q=q, ax=axis: np.quantile(a, q, axis=ax, method="linear")),
        candidates={"sort_lerp": lambda a=a, q=q, ax=axis: sort_lerp_quantile(a, q, ax)},
        check=exact,
        samples=samples,
    )

if not SMOKE:
    suite.save()
