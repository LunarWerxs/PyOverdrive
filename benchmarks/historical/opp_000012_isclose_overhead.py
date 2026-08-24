"""OPP-000012: np.isclose wrapper/call overhead.

numpy/numpy#16160 claims np.isclose(a, b, rtol=, atol=) carries meaningfully
more per-call overhead than a hand-written tolerance expression, because of
repeated asanyarray conversions of a/b/atol/rtol, an
errstate(invalid='ignore') context manager, and two separate
np.all(np.isfinite(...)) reductions. The reporter's own 2024 apples-to-apples
retest on NumPy 2.0 (lab/corpus/OPP-000012.yaml claim) measured 20.7us vs
7.87us on two 1000-element float64 arrays, about 2.63x, using
np.abs(a-b) <= atol + rtol*np.abs(b) as the candidate expression. Per the
report's "What PyOverdrive should do next" step 1, this reproducer measures:

  - baseline: np.isclose(a, b, rtol=1e-05, atol=1e-08) (numpy's defaults, the
    call users write today), swept over dtype in {float32, float64} and
    size in {1, 10, 100, 1000, 10000, 100000}, plus the scalar-pair case
    (a=0.5, b=0) from the original 2020 report.
  - candidate "fused": a hand-rolled fast path that skips the redundant
    asanyarray/errstate/double-np.all wrapper machinery for the common case
    (finite scalar atol/rtol, equal_nan=False), folding the two
    np.all(np.isfinite(...)) reductions into one np.all(xfin & yfin) and
    dropping errstate entirely, since the fast path never triggers an
    invalid-value warning when atol/rtol are both finite (see docstring on
    fused_isclose below). It falls back to calling numpy's own isclose,
    unmodified, for the narrow cases the report flags as unsafe to
    fast-path: non-finite atol/rtol (WarrenWeckesser's 2020-05-09 edge case)
    and equal_nan=True.

Report step 4 asked to check numpy/numpy#26252 (the errstate-only-when-needed
PR) against current main before building anything: this repo's installed
numpy is 2.4.5, which already ships that fix and NEP 50 weak scalar
promotion, so the baseline being measured here is already the *improved*
stock isclose, not the 2020 original; any speedup this reproducer sees is on
top of that, smaller than the original 61.8x/11.5x numbers and possibly
smaller than the 2.63x 2024 retest too.

Report step 3 asked for a differential-correctness sweep on non-finite
atol/rtol and on NaN/Inf entries in a/b; those are the "edge_*" cases below,
run through the same suite.measure/check machinery as the perf cases so a
CORRECTNESS-FAIL would show up the same way.

Runtime: dyno's BenchSuite auto-calibrates each timed block to >=5ms
regardless of how fast a single call is, so total runtime scales with
(number of cases x variants x samples), not with array size; the full sweep
below (13 perf cases + 4 edge cases, 1 candidate each, samples=11) is
expected to finish in a few seconds, well under the ~90s budget, so nothing
here needed shrinking from the report's stated size list.
"""

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 16160
SMOKE = "--smoke" in sys.argv

# Candidate helper bindings, captured once here at import time (setup, never
# timed). House rule: a fast-path candidate must not resolve a public numpy
# name at call time inside its timed body when an equivalent can be captured
# up front, because PyOverdrive's own fix for this exact opportunity would
# ship by patching numpy.isclose (and friends) in place; a candidate that
# looks the name up fresh on every call would silently start measuring that
# future patch instead of stock numpy on a rerun. Binding here freezes this
# process's stock numpy 2.4.5 objects.
_asarray = np.asarray
_isfinite = np.isfinite
_zeros_like = np.zeros_like
_ones_like = np.ones_like


def fused_isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Fast-path candidate per OPP-000012's report.

    Skips isclose's repeated asanyarray(a)/asanyarray(b) conversions (atol
    and rtol are used directly as Python/numpy scalars, no conversion at
    all) and its errstate(invalid='ignore') guard, and folds the two
    np.all(np.isfinite(...)) reductions into one np.all(xfin & yfin) (valid
    whenever xfin/yfin are broadcastable, which they must be for isclose to
    be defined at all: all(xfin) and all(yfin) == all(xfin & yfin) under
    broadcasting). The errstate guard is safe to drop unconditionally here
    only because this fast path requires atol/rtol to both be finite
    (checked below) - that is precisely the case WarrenWeckesser (2020-05-09)
    identified as the one where errstate's suppression is load-bearing, so
    non-finite atol/rtol falls back to stock isclose instead, matching stock
    behavior (including its silent warning suppression) exactly. equal_nan
    and non-floating dtypes also fall back: the report names both as outside
    the sweep this fast path was justified against. The fallback calls
    numpy's own isclose directly; this script never imports pyoverdrive, so
    that name is guaranteed to be stock numpy in this process.
    """
    if equal_nan or not (math.isfinite(atol) and math.isfinite(rtol)):
        return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    x = _asarray(a)
    y = _asarray(b)
    if x.dtype.kind != "f" or y.dtype.kind != "f":
        return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    finite = _isfinite(x) & _isfinite(y)
    if finite.all():
        return (abs(x - y) <= atol + rtol * abs(y))[()]

    # Non-finite entries present: mirror stock isclose's masked branch
    # exactly (broadcast x/y up to the shared shape via the same
    # multiply-by-ones-like trick stock isclose uses, then compare within
    # tolerance where finite and by equality where not).
    cond = _zeros_like(finite, subok=True)
    xb = x * _ones_like(cond)
    yb = y * _ones_like(cond)
    cond[finite] = abs(xb[finite] - yb[finite]) <= atol + rtol * abs(yb[finite])
    cond[~finite] = xb[~finite] == yb[~finite]
    return cond[()]


def make_array_pair(rng, n, dtype):
    """n-length (a, b) pair with a realistic mix of within-tolerance and
    far-apart elements at numpy's default rtol=1e-05/atol=1e-08, so the
    boolean output mask is neither all-True nor all-False."""
    dt = np.dtype(dtype)
    a = rng.standard_normal(n).astype(dt)
    near = (rng.standard_normal(n) * 1e-9).astype(dt)
    far = (rng.standard_normal(n) * 1e-2).astype(dt)
    use_near = rng.random(n) < 0.5
    b = np.where(use_near, a + near, a + far).astype(dt)
    return a, b


if SMOKE:
    dtypes = [np.float64]
    sizes = [10, 1000]
else:
    dtypes = [np.float32, np.float64]
    sizes = [1, 10, 100, 1000, 10000, 100000]

suite = BenchSuite("OPP-000012", "np.isclose wrapper/call overhead vs fused tolerance expression")
samples = 3 if SMOKE else 11

rng = np.random.default_rng(SEED)

# Scalar-pair case, matching the original 2020 report's a=0.5, b=0 input.
suite.measure(
    case="scalar_pair",
    params={"dtype": "python_float", "a": 0.5, "b": 0.0},
    baseline=("numpy.isclose", lambda: np.isclose(0.5, 0.0)),
    candidates={"fused": lambda: fused_isclose(0.5, 0.0)},
    check=np.array_equal,
    samples=samples,
)

# Array-pair perf sweep.
for dtype in dtypes:
    for n in sizes:
        a, b = make_array_pair(rng, n, dtype)
        case = f"array_pair_{np.dtype(dtype).name}_n{n}"
        suite.measure(
            case=case,
            params={"dtype": np.dtype(dtype).name, "n": n},
            baseline=("numpy.isclose", lambda a=a, b=b: np.isclose(a, b)),
            candidates={"fused": lambda a=a, b=b: fused_isclose(a, b)},
            check=np.array_equal,
            samples=samples,
        )

# Differential-correctness edge cases (report step 3): non-finite atol/rtol,
# and NaN/Inf entries in a/b. Small arrays; the point is the CORRECT/
# CORRECTNESS-FAIL verdict dyno prints, not the timing.
edge_a = np.array([1.0, np.nan, np.inf, -np.inf, 2.0, 0.0, -3.5, 100.0])
edge_b = np.array([1.0, np.nan, np.inf, 5.0, 2.0000001, -0.0, -3.5, 100.0001])

suite.measure(
    case="edge_atol_inf",
    params={"dtype": "float64", "n": len(edge_a), "atol": "inf"},
    baseline=("numpy.isclose", lambda: np.isclose(edge_a, edge_b, atol=np.inf)),
    candidates={"fused": lambda: fused_isclose(edge_a, edge_b, atol=np.inf)},
    check=np.array_equal,
    samples=samples,
)

suite.measure(
    case="edge_rtol_neg_inf",
    params={"dtype": "float64", "n": len(edge_a), "rtol": "-inf"},
    baseline=("numpy.isclose", lambda: np.isclose(edge_a, edge_b, rtol=-np.inf)),
    candidates={"fused": lambda: fused_isclose(edge_a, edge_b, rtol=-np.inf)},
    check=np.array_equal,
    samples=samples,
)

suite.measure(
    case="edge_atol_nan",
    params={"dtype": "float64", "n": len(edge_a), "atol": "nan"},
    baseline=("numpy.isclose", lambda: np.isclose(edge_a, edge_b, atol=np.nan)),
    candidates={"fused": lambda: fused_isclose(edge_a, edge_b, atol=np.nan)},
    check=np.array_equal,
    samples=samples,
)

suite.measure(
    case="edge_nan_inf_entries_default_tol",
    params={"dtype": "float64", "n": len(edge_a)},
    baseline=("numpy.isclose", lambda: np.isclose(edge_a, edge_b)),
    candidates={"fused": lambda: fused_isclose(edge_a, edge_b)},
    check=np.array_equal,
    samples=samples,
)

suite.measure(
    case="edge_equal_nan_true",
    params={"dtype": "float64", "n": len(edge_a), "equal_nan": True},
    baseline=("numpy.isclose", lambda: np.isclose(edge_a, edge_b, equal_nan=True)),
    candidates={"fused": lambda: fused_isclose(edge_a, edge_b, equal_nan=True)},
    check=np.array_equal,
    samples=samples,
)

if not SMOKE:
    suite.save()
