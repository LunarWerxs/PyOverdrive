"""Batch-11 calibration battery: apply_along_axis_reduce (OPP-000054)
and vectorize_ufunc_direct (OPP-000055).

Both paths remove a PYTHON LOOP, so their margin scales with the loop's
trip count, not with the array's byte size: the apply_along_axis cells
sweep slice COUNT (and slice length separately, to show it barely
matters), and the vectorize cells sweep element count. That is also why
the floors here are set for dispatch-tax safety (ADR-0003) rather than
at a 1.3x crossing - both are far past the bar everywhere they apply.

Candidates are the shipped routes exactly: the reducer's own axis= call
for apply_along_axis, the stock ufunc for vectorize. Both are checked
BIT-IDENTICAL by the battery, which is the contract.

Result JSON: benchmarks/results/BATCH11-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_batch11_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from lab.dyno import BenchSuite

from pyoverdrive.fastpaths.apply_along_axis_reduce import (
    _EXACT_ANY_AXIS,
    _ORDER_SENSITIVE_LAST_AXIS,
    SLICE_MIN,
)
from pyoverdrive.fastpaths.vectorize_ufunc import _SERVED_NAMES

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite(
    "BATCH11-CAL",
    "apply_along_axis known-reducer interception; vectorize(ufunc) direct call",
)
rng = np.random.default_rng(20260825)


def exact(c, b):
    c = np.asarray(c)
    b = np.asarray(b)
    return (
        c.shape == b.shape
        and c.dtype == b.dtype
        and bool(np.array_equal(c, b, equal_nan=True))
    )


# --- 1. apply_along_axis: margin vs SLICE COUNT ----------------------------

_REDUCERS = {name: getattr(np, name) for name in _EXACT_ANY_AXIS + _ORDER_SENSITIVE_LAST_AXIS}

SLICE_COUNTS = (
    [SLICE_MIN, 2_000] if SMOKE else [SLICE_MIN, 64, 256, 1_000, 5_000, 20_000]
)
SLICE_LEN = 50

for nslices in SLICE_COUNTS:
    a = rng.standard_normal((nslices, SLICE_LEN))
    for name in ("mean", "sum", "max", "median"):
        f = _REDUCERS[name]
        suite.measure(
            case=f"aaa_{name}_slices{nslices}_len{SLICE_LEN}",
            params={"reducer": name, "slices": nslices, "slice_len": SLICE_LEN,
                    "axis": -1, "slice_min": SLICE_MIN},
            baseline=(
                "numpy.apply_along_axis",
                lambda a=a, f=f: np.apply_along_axis(f, -1, a),
            ),
            candidates={"axis_reduction": lambda a=a, f=f: f(a, axis=-1)},
            check=exact,
            samples=SAMPLES,
        )

# slice LENGTH sweep at fixed count: the margin should barely move, which is
# the evidence that the Python loop - not the arithmetic - is the cost
if not SMOKE:
    for slice_len in (10, 100, 1_000):
        a = rng.standard_normal((2_000, slice_len))
        suite.measure(
            case=f"aaa_mean_slices2000_len{slice_len}",
            params={"reducer": "mean", "slices": 2_000, "slice_len": slice_len, "axis": -1},
            baseline=("numpy.apply_along_axis", lambda a=a: np.apply_along_axis(np.mean, -1, a)),
            candidates={"axis_reduction": lambda a=a: np.mean(a, axis=-1)},
            check=exact,
            samples=SAMPLES,
        )

    # non-last axis, exact reducers only (the order-sensitive ones refuse
    # there by construction - see the module's split)
    a = rng.standard_normal((100, 2_000))
    for name in ("max", "argmax", "median"):
        f = _REDUCERS[name]
        suite.measure(
            case=f"aaa_{name}_axis0_2000slices",
            params={"reducer": name, "slices": 2_000, "slice_len": 100, "axis": 0},
            baseline=("numpy.apply_along_axis", lambda a=a, f=f: np.apply_along_axis(f, 0, a)),
            candidates={"axis_reduction": lambda a=a, f=f: f(a, axis=0)},
            check=exact,
            samples=SAMPLES,
        )

    # 3-D, and the below-floor witness cell
    a3 = rng.standard_normal((40, 30, 25))
    suite.measure(
        case="aaa_mean_3d_40x30x25",
        params={"reducer": "mean", "slices": 1_200, "slice_len": 25, "axis": -1},
        baseline=("numpy.apply_along_axis", lambda a=a3: np.apply_along_axis(np.mean, -1, a3)),
        candidates={"axis_reduction": lambda a=a3: np.mean(a3, axis=-1)},
        check=exact,
        samples=SAMPLES,
    )
    a_small = rng.standard_normal((SLICE_MIN - 1, SLICE_LEN))
    suite.measure(
        case=f"aaa_mean_slices{SLICE_MIN - 1}_below_floor",
        params={"reducer": "mean", "slices": SLICE_MIN - 1, "slice_len": SLICE_LEN,
                "axis": -1, "below_floor": True},
        baseline=("numpy.apply_along_axis", lambda a=a_small: np.apply_along_axis(np.mean, -1, a_small)),
        candidates={"axis_reduction": lambda a=a_small: np.mean(a_small, axis=-1)},
        check=exact,
        samples=SAMPLES,
    )

# --- 2. vectorize(ufunc): margin vs ELEMENT COUNT --------------------------

VEC_NS = [10_000, 1_000_000] if SMOKE else [100, 1_000, 10_000, 100_000, 1_000_000]
VEC_FUNCS = ("sin", "exp", "sqrt") if SMOKE else ("sin", "exp", "sqrt", "log", "tanh", "rint")

for n in VEC_NS:
    x = np.abs(rng.standard_normal(n)) + 1e-6
    for name in VEC_FUNCS:
        uf = getattr(np, name)
        v = np.vectorize(uf)
        suite.measure(
            case=f"vectorize_{name}_n{n}",
            params={"ufunc": name, "n": n, "served_count": len(_SERVED_NAMES)},
            baseline=("numpy.vectorize.__call__", lambda v=v, x=x: v(x)),
            candidates={"ufunc_direct": lambda uf=uf, x=x: uf(x)},
            check=exact,
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
