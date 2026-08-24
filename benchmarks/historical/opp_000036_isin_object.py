"""OPP-000036: isin on object arrays, Python-set route vs stock O(n*m).

numpy/numpy#14997 (chaburkland, 2019): object dtype unconditionally
takes in1d's O(n*m) broadcast-equality path; the reporter's own numbers
are "around 10/15 seconds" vs "no longer than ~0.1 seconds" for a set
route (DERIVED ~100-150x). Sibling of the shipped isin_string_hash
(OPP-000023, 2317x end-to-end): the mechanism is identical, the
operands are object arrays instead of StringDType.

What this reproducer measures:

  1. n x m grid of hashable-object arrays (strings-as-objects and
     ints-as-objects): element n in {1_000, 30_000}, test m in
     {100, 3_000}, plus a small anti-regime cell (100 x 5).
  2. The guarded candidate a shipped predicate would actually run:
     the same set route WITH the pre-scan the NaN hazard demands (see
     3), timed honestly.
  3. Semantics probes, each its own case with an exact check:
     - nan_object: element and test both contain float('nan') objects.
       Python's `in` matches NaN by IDENTITY where stock's
       equality-based path says nan != nan, so if stock and the set
       route disagree the check FAILS - that failure is the measured
       boundary of the regime (the shipped predicate must refuse or
       special-case NaN-likes), not a script bug. The DISTINCT-nan
       variant (different nan objects in element vs test) probes the
       identity-vs-equality split from the other side.
     - int_float_mix: 1 vs 1.0 vs True (equal by __eq__ AND by hash, so
       both routes must agree).
  4. Unhashable elements (lists) are NOT timed: the shipped route falls
     back internally on TypeError; the differential battery owns that.

Correctness: bit-identical bool arrays wherever the check passes.

House rules: never imports pyoverdrive; the candidate uses only
set/tolist/fromiter, so a patched dispatch could not recurse.

Result JSON: benchmarks/results/OPP-000036/.
Run: .venv/Scripts/python benchmarks/historical/opp_000036_isin_object.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 14997
SMOKE = "--smoke" in sys.argv


def set_route(element, test):
    lookup = set(test.tolist())
    return np.fromiter(
        (s in lookup for s in element.tolist()), dtype=bool, count=element.size
    )


def _is_nanlike(x):
    try:
        return x != x
    except Exception:
        return False


def set_route_guarded(element, test):
    """What a shipped path would run: scan for NaN-like objects (identity
    membership diverges from stock's equality semantics) before trusting
    the set. The scan cost is part of the honest timing."""
    el = element.tolist()
    te = test.tolist()
    if any(_is_nanlike(x) for x in te) or any(_is_nanlike(x) for x in el):
        raise AssertionError("NaN-like present: predicate would refuse (not this data)")
    lookup = set(te)
    return np.fromiter((s in lookup for s in el), dtype=bool, count=element.size)


def exact(cand, base):
    return cand.dtype == base.dtype and cand.shape == base.shape and bool(
        np.array_equal(cand, base)
    )


suite = BenchSuite("OPP-000036", "isin on object arrays: python set vs O(n*m)")
rng = np.random.default_rng(SEED)


def obj_strings(n, vocab):
    return np.array([vocab[i] for i in rng.integers(0, len(vocab), size=n)], dtype=object)


def obj_ints(n, hi):
    return np.array([int(x) for x in rng.integers(0, hi, size=n)], dtype=object)


VOCAB = [f"key_{i:05d}" for i in range(5_000)]

if SMOKE:
    GRID = [("str", 1_000, 100)]
    SAMPLES = 3
else:
    GRID = [
        ("str", 1_000, 100),
        ("str", 30_000, 3_000),
        ("int", 1_000, 100),
        ("int", 30_000, 3_000),
        ("str", 100, 5),
    ]
    SAMPLES = 7

for kind, n, m in GRID:
    if kind == "str":
        el = obj_strings(n, VOCAB)
        te = np.array(
            [VOCAB[i] for i in rng.choice(len(VOCAB), size=m, replace=False)], dtype=object
        )
    else:
        el = obj_ints(n, 10_000)
        te = obj_ints(m, 10_000)
    suite.measure(
        case=f"isin_object_{kind}_n{n}_m{m}",
        params={"kind": kind, "n": n, "m": m},
        baseline=("numpy.isin", lambda e=el, t=te: np.isin(e, t)),
        candidates={
            "set_route": lambda e=el, t=te: set_route(e, t),
            "set_route_nan_guarded": lambda e=el, t=te: set_route_guarded(e, t),
        },
        check=exact,
        samples=SAMPLES,
    )

if not SMOKE:
    # semantics probes: SAME nan object on both sides, then DISTINCT nans
    nan = float("nan")
    el = np.array([nan, "a", "b"] + ["c"] * 300, dtype=object)
    te = np.array([nan, "a"], dtype=object)
    suite.measure(
        case="isin_object_same_nan_object_probe",
        params={"probe": "identity-vs-equality, same nan object"},
        baseline=("numpy.isin", lambda e=el, t=te: np.isin(e, t)),
        candidates={"set_route": lambda e=el, t=te: set_route(e, t)},
        check=exact,
        samples=5,
    )
    el2 = np.array([float("nan"), "a"] + ["c"] * 300, dtype=object)
    te2 = np.array([float("nan"), "a"], dtype=object)
    suite.measure(
        case="isin_object_distinct_nan_objects_probe",
        params={"probe": "identity-vs-equality, distinct nan objects"},
        baseline=("numpy.isin", lambda e=el2, t=te2: np.isin(e, t)),
        candidates={"set_route": lambda e=el2, t=te2: set_route(e, t)},
        check=exact,
        samples=5,
    )
    mix_el = np.array([1, 1.0, True, 2, "x"] + list(range(10, 310)), dtype=object)
    mix_te = np.array([1.0, "x"], dtype=object)
    suite.measure(
        case="isin_object_int_float_bool_mix",
        params={"probe": "cross-type __eq__/__hash__ agreement"},
        baseline=("numpy.isin", lambda e=mix_el, t=mix_te: np.isin(e, t)),
        candidates={"set_route": lambda e=mix_el, t=mix_te: set_route(e, t)},
        check=exact,
        samples=5,
    )
    suite.save()
