"""OPP-000013: nanquantile axis-reduction overhead vs quantile / a masked
fast path.

numpy/numpy#16575 claims np.nanquantile(a, q, axis=...) is far slower than
np.quantile(a, q, axis=...) on 2-D+ arrays reduced along one axis, because
nanquantile loops per-slice in Python (seberg, numpy maintainer) while
quantile amortizes that overhead. Reporter's own repro: (27, 100) float64,
axis=0, q=0.8 -> np.quantile 147us, np.nanquantile 10.3ms (70.07x).
lbferreira's fastnanquantile benchmark used (50, 100, 100), q=0.6, axis=0
(0.2658s stock nanquantile).

This reproducer measures the two routes named in the opportunity record's
"What PyOverdrive should do next" step 1:

  - route 1 "skip_to_quantile": when the whole input has zero NaNs, an
    O(n) np.isnan(a).any() scan routes straight to np.quantile instead of
    nanquantile's per-slice Python path. Bit-identical by construction
    (nanquantile with no NaNs present equals quantile), and the isnan scan
    is INCLUDED in the timed call so the measured speedup is net of the
    check's own cost, not an idealized number.
  - route 2 "masked_route2": for arrays that do contain NaNs, a vectorized
    masked-reduction reimplementation (sort each slice so NaNs land at the
    end for floats, per-slice valid-count from np.isnan, linear-interpolate
    at q*(valid_count-1) via np.take_along_axis) standing in for the
    fastnanquantile/numbagg-style candidates the record cites. All-NaN
    slices are masked back to NaN afterward, matching nanquantile's
    contract.

Neither candidate imports pyoverdrive, and both call only ordinary public
numpy names (np.quantile, np.isnan, np.sort, np.take_along_axis, ...), none
of which is np.nanquantile itself -- so neither risks the self-patch
recursion the house rule warns about (a fast path for nanquantile calling
nanquantile again as its own fallback). route 1 has no fallback branch at
all in this script: it is only ever measured on the zero-NaN cases where it
is valid, never combined into an all-cases router that would need to call
np.nanquantile to handle the NaN-present case.

Shapes: (27, 100) is the reporter's own repro; (50, 100, 100) is
lbferreira's fastnanquantile benchmark shape. The report's third example,
(1000, 1000), is shrunk here to (500, 500) to keep the full (non-smoke)
battery under roughly 90s wall clock -- a 1000x1000 array reduced along
axis=0 pushes nanquantile's per-slice Python loop across 1000 slices of
length 1000 each, which is not needed to demonstrate the effect once
(27, 100) and (50, 100, 100) already bracket "few slices" and "many
slices"; this follows the task's explicit allowance to shrink the largest
size and say so here. Sample counts are also lowered for the two larger,
slower shapes (still within the 5-11 range asked for).

q in {0.6, 0.8}, matching the thread. NaN fraction is applied per-element
(independent Bernoulli draw at the given probability, not a fixed
per-slice count) at 0%, 1%, 10%, to separate route 1 (valid only at 0%
NaNs) from route 2 (the NaN-present regime).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 16575
SMOKE = "--smoke" in sys.argv


def make_array(rng, shape, nan_frac, dtype=np.float64):
    a = rng.uniform(size=shape).astype(dtype)
    if nan_frac > 0:
        mask = rng.random(size=shape) < nan_frac
        a[mask] = np.nan
    return a


def skip_to_quantile(a, q, axis):
    """Route 1: whole-array isnan scan, then numpy's own quantile fast
    path. Only ever exercised on zero-NaN arrays in this script (see module
    docstring); the isnan scan is included in the timed call so the
    measured speedup already accounts for its cost."""
    if np.isnan(a).any():
        raise AssertionError("skip_to_quantile candidate exercised on an array with NaNs")
    return np.quantile(a, q, axis=axis)


def masked_route2(a, q, axis):
    """Route 2: vectorized masked-reduction nanquantile, method='linear'
    (numpy's default), reimplemented without a per-slice Python loop."""
    a_moved = np.moveaxis(a, axis, 0)
    n_total = a_moved.shape[0]
    is_nan = np.isnan(a_moved)
    valid_count = n_total - is_nan.sum(axis=0)
    sorted_a = np.sort(a_moved, axis=0)  # NaN sorts to the end for floats
    vc_safe = np.maximum(valid_count, 1)
    virt = q * (vc_safe - 1).astype(np.float64)
    lo = np.clip(np.floor(virt).astype(np.intp), 0, vc_safe - 1)
    hi = np.clip(np.ceil(virt).astype(np.intp), 0, vc_safe - 1)
    gamma = virt - lo
    lo_val = np.take_along_axis(sorted_a, lo[np.newaxis, ...], axis=0)[0]
    hi_val = np.take_along_axis(sorted_a, hi[np.newaxis, ...], axis=0)[0]
    result = lo_val + gamma * (hi_val - lo_val)
    return np.where(valid_count == 0, np.nan, result)


def close_equal_nan(actual, expected):
    """allclose with equal_nan=True and an ULP-scale tolerance: both
    candidates use the same linear-interpolation formula as numpy's
    nanquantile/quantile, but reach it via a different code path (vectorized
    sort + take_along_axis, or numpy's direct quantile path, vs
    nanquantile's per-slice Python loop -- and numpy's own lerp has a
    t>=0.5 special case for numerical stability that this reimplementation
    does not replicate). Mathematically the same interpolation, so the
    tolerance is ULP-scale, not a semantic relaxation."""
    return np.allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True)


if SMOKE:
    shapes = [("27x100", (27, 100), 0)]
    q_values = [0.8]
    nan_fracs_by_shape = {"27x100": [0.0, 0.10]}
else:
    shapes = [
        ("27x100", (27, 100), 0),
        ("50x100x100", (50, 100, 100), 0),
        ("500x500", (500, 500), 0),
    ]
    q_values = [0.6, 0.8]
    nan_fracs_by_shape = {
        "27x100": [0.0, 0.01, 0.10],
        "50x100x100": [0.0, 0.01, 0.10],
        "500x500": [0.0, 0.01, 0.10],
    }
    samples_by_shape = {"27x100": 11, "50x100x100": 5, "500x500": 5}

suite = BenchSuite(
    "OPP-000013",
    "nanquantile axis-reduction vs quantile skip-path / masked route",
)

rng = np.random.default_rng(SEED)

for shape_name, shape, axis in shapes:
    samples = 3 if SMOKE else samples_by_shape[shape_name]
    for q in q_values:
        for nan_frac in nan_fracs_by_shape[shape_name]:
            a = make_array(rng, shape, nan_frac)
            case = f"{shape_name}_q{q}_nan{int(round(nan_frac * 100))}pct"
            params = {
                "dtype": "float64",
                "shape": list(shape),
                "axis": axis,
                "q": q,
                "nan_frac": nan_frac,
            }
            if nan_frac == 0.0:
                candidates = {
                    "skip_to_quantile": lambda a=a, q=q, axis=axis: skip_to_quantile(a, q, axis),
                }
            else:
                candidates = {
                    "masked_route2": lambda a=a, q=q, axis=axis: masked_route2(a, q, axis),
                }
            suite.measure(
                case=case,
                params=params,
                baseline=(
                    "numpy.nanquantile",
                    lambda a=a, q=q, axis=axis: np.nanquantile(a, q, axis=axis),
                ),
                candidates=candidates,
                check=close_equal_nan,
                samples=samples,
            )

if not SMOKE:
    suite.save()
