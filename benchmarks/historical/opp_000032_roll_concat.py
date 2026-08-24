"""OPP-000032 (+ OPP-000031's transparent ceiling): np.roll vs cheaper routes.

numpy/numpy#10848 (ucyo, numpy 1.13.1): np.roll(d, 1) on 99 int elements
at 15 us vs np.concatenate([d[-1:], d[:-1]]) at 1.26 us - DERIVED 11.90x.
seberg closed it: the gap is np.roll's Python-level overhead, "should be
pretty small nowadays"; a real roll perf issue was already fixed by 1.13.
So the live question is purely how much fixed overhead np.roll still
carries on CURRENT numpy, and below what size a concatenate route wins.

numpy/numpy#12389 (grlee77): np.roll(r, (), axis=()) on 256^3 float64
spends 54.2 ms doing a copy through the full index machinery; the 10-20us
"fix" in-thread is returning the input UNCOPIED, which the roll contract
forbids (seberg's aliasing objection; see the OPP-000031 record). The
transparent ceiling is a plain order-preserving copy, measured here as
copy_K on the degenerate-argument cases.

Cases:

  1. 1-D scalar-shift size sweep 99 .. 1e6, int64 and float64, shift 1
     and a larger shift: np.roll vs concatenate of two slices (shift
     normalized modulo n exactly as roll does). Exact-equality checks.
  2. Degenerate arguments (OPP-000031): shift=0 on 1-D, and
     shift=(), axis=() on the thread's 256x256x256 float64, vs a.copy
     (order='K' to preserve layout as np.empty_like does). Expected
     near-parity at large sizes (the copy dominates both routes);
     whatever margin remains IS the transparent ceiling, honestly
     measured. An F-order variant checks result-order fidelity of the
     copy route (np.roll preserves input order via empty_like).

House rules: never imports pyoverdrive; candidates use concatenate /
ndarray.copy, neither of which a roll predicate touches, so a patched
dispatch could not recurse.

Result JSON: benchmarks/results/OPP-000032/ (both records cite it).
Run: .venv/Scripts/python benchmarks/historical/opp_000032_roll_concat.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 10848
SMOKE = "--smoke" in sys.argv


def roll_concat(d, s):
    s = s % d.size
    if s == 0:
        return d.copy(order="K")
    return np.concatenate((d[-s:], d[:-s]))


def exact(cand, base):
    return (
        cand.dtype == base.dtype
        and cand.shape == base.shape
        and cand.flags.c_contiguous == base.flags.c_contiguous
        and cand.flags.f_contiguous == base.flags.f_contiguous
        and bool(np.array_equal(cand, base))
    )


suite = BenchSuite("OPP-000032", "np.roll vs concatenate/copy routes")
rng = np.random.default_rng(SEED)

SIZES = [99, 1_000] if SMOKE else [99, 1_000, 10_000, 100_000, 1_000_000]
SAMPLES = 3 if SMOKE else 11

for n in SIZES:
    for label, dtype in (("int64", np.int64), ("float64", np.float64)):
        if np.issubdtype(dtype, np.integer):
            d = rng.integers(-(2**40), 2**40, size=n, dtype=dtype)
        else:
            d = rng.random(n)
        for s in (1, 17 % n or 1):
            suite.measure(
                case=f"roll_1d_n{n}_{label}_shift{s}",
                params={"n": n, "dtype": label, "shift": s},
                baseline=("numpy.roll", lambda d=d, s=s: np.roll(d, s)),
                candidates={"concat_slices": lambda d=d, s=s: roll_concat(d, s)},
                check=exact,
                samples=SAMPLES if n <= 100_000 else max(5, SAMPLES - 4),
            )

# degenerate arguments (OPP-000031's transparent ceiling)
d1 = rng.random(100_000)
suite.measure(
    case="roll_1d_n100000_float64_shift0",
    params={"n": 100_000, "dtype": "float64", "shift": 0},
    baseline=("numpy.roll", lambda d=d1: np.roll(d1, 0)),
    candidates={"copy_K": lambda d=d1: d1.copy(order="K")},
    check=exact,
    samples=SAMPLES,
)

if not SMOKE:
    big = rng.random((256, 256, 256))
    suite.measure(
        case="roll_emptyaxes_256cubed_float64",
        params={"shape": [256, 256, 256], "shift": [], "axis": []},
        baseline=("numpy.roll", lambda a=big: np.roll(a, (), axis=())),
        candidates={"copy_K": lambda a=big: a.copy(order="K")},
        check=exact,
        samples=7,
    )
    bigF = np.asfortranarray(rng.random((512, 512)))
    suite.measure(
        case="roll_emptyaxes_512x512_ForderF_order_fidelity",
        params={"shape": [512, 512], "order": "F", "shift": [], "axis": []},
        baseline=("numpy.roll", lambda a=bigF: np.roll(a, (), axis=())),
        candidates={"copy_K": lambda a=bigF: a.copy(order="K")},
        check=exact,
        samples=9,
    )
    suite.save()
