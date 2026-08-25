"""Batch-13 calibration battery: 1-D constant-mode np.pad (OPP-000057).

EVERY CELL IS MEASURED TWICE, bare and consumed, and that is the point of
this battery rather than a detail of it.

The no-constant route allocates with np.zeros, i.e. calloc, so the pages it
hands back have not been faulted in yet. Timing only the pad call therefore
banks work the caller has not paid for: at a small input with a huge pad the
bare ratio reaches 5090x while the same cell, summed before the clock stops,
is 0.87x - a REGRESSION reported as the largest speedup in the project. The
`_used` cells are the honest ones and OUTPUT_CAP is set from them.

That is also why the cap is on the length of the RESULT and not of the
input: a tiny array with an enormous pad is exactly the shape that looks
best bare and performs worst used. The over-cap cells below are kept
deliberately, as the standing evidence for where the crossing is, so a
future NumPy or allocator that moves it shows up here rather than in a
user's results.

Result JSON: benchmarks/results/BATCH13-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_batch13_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from lab.dyno import BenchSuite

from pyoverdrive.fastpaths.pad_1d_constant import OUTPUT_CAP, _run

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite(
    "BATCH13-CAL",
    "np.pad on 1-D arrays in constant mode, via np.zeros / np.empty plus one assignment",
)


def byte_exact(c, b):
    c, b = np.asarray(c), np.asarray(b)
    if c.dtype != b.dtype or c.shape != b.shape:
        return False
    # bytes, not array_equal: this path's whole contract is that a
    # caller-supplied -0.0 keeps its sign bit and a NaN matches a NaN
    return c.tobytes() == b.tobytes()


def same_float(c, b):
    """For the consumed cells, where both sides return a summed scalar."""
    return (c == b) or (np.isnan(c) and np.isnan(b))


def arr(n, dtype=np.float64, seed=13):
    return np.ascontiguousarray(
        (np.random.default_rng(seed).standard_normal(n) * 4).astype(dtype)
    )


# --- 1. margin vs OUTPUT length, both routes, bare and consumed -------------

CELLS = (
    [(64, 3), (4_000, 3)] if SMOKE
    else [(8, 3), (64, 3), (256, 3), (1_000, 3), (4_000, 3), (8_000, 3),
          (16_000, 3), (64, 1_000), (8, 8_000)]
)

for n, pad in CELLS:
    a = arr(n)
    out_len = n + 2 * pad
    common = {"n": n, "pad": pad, "out_len": out_len, "output_cap": OUTPUT_CAP,
              "served": out_len <= OUTPUT_CAP}

    suite.measure(
        case=f"pad_zero_n{n}_p{pad}",
        params={**common, "route": "np.zeros", "constant": "absent", "timing": "bare"},
        baseline=("numpy.pad", lambda a=a, p=pad: np.pad(a, (p, p))),
        candidates={"zeros_assign": lambda a=a, p=pad: _run(a, (p, p))},
        check=byte_exact,
        samples=SAMPLES,
    )
    suite.measure(
        case=f"pad_zero_n{n}_p{pad}_used",
        params={**common, "route": "np.zeros", "constant": "absent",
                "timing": "consumed", "note": "the honest number; sets OUTPUT_CAP"},
        baseline=("numpy.pad+sum", lambda a=a, p=pad: float(np.pad(a, (p, p)).sum())),
        candidates={"zeros_assign+sum": lambda a=a, p=pad: float(_run(a, (p, p)).sum())},
        check=same_float,
        samples=SAMPLES,
    )
    suite.measure(
        case=f"pad_const_n{n}_p{pad}",
        params={**common, "route": "np.empty", "constant": 5.0, "timing": "bare"},
        baseline=("numpy.pad", lambda a=a, p=pad: np.pad(a, (p, p), constant_values=5.0)),
        candidates={"empty_assign": lambda a=a, p=pad: _run(a, (p, p), constant_values=5.0)},
        check=byte_exact,
        samples=SAMPLES,
    )
    suite.measure(
        case=f"pad_const_n{n}_p{pad}_used",
        params={**common, "route": "np.empty", "constant": 5.0, "timing": "consumed"},
        baseline=("numpy.pad+sum",
                  lambda a=a, p=pad: float(np.pad(a, (p, p), constant_values=5.0).sum())),
        candidates={"empty_assign+sum":
                    lambda a=a, p=pad: float(_run(a, (p, p), constant_values=5.0).sum())},
        check=same_float,
        samples=SAMPLES,
    )

# --- 2. past the cap: the cells that justify refusing ----------------------
# Kept on purpose. Bare these look spectacular and consumed they are a loss;
# both numbers are recorded so the gap is evidence rather than a claim.

OVER = [(8, 120_000)] if SMOKE else [(8, 32_000), (8, 120_000), (8, 1_000_000)]

def _unguarded_zero_route(a, pad):
    """The route WITHOUT the predicate, so a refused shape can be measured.

    _run() consults the same _parse() the predicate does and would refuse
    these outright, which is the correct shipped behaviour and useless for
    measuring what that refusal is worth. This is the identical two lines
    with the cap taken off.
    """
    out = np.zeros(a.shape[0] + 2 * pad, dtype=a.dtype)
    out[pad:pad + a.shape[0]] = a
    return out


for n, pad in OVER:
    a = arr(n)
    out_len = n + 2 * pad
    assert out_len > OUTPUT_CAP, "this block is for refused shapes only"
    suite.measure(
        case=f"overcap_zero_n{n}_p{pad}",
        params={"n": n, "pad": pad, "out_len": out_len, "served": False,
                "timing": "bare", "note": "calloc pages not yet faulted in"},
        baseline=("numpy.pad", lambda a=a, p=pad: np.pad(a, (p, p))),
        candidates={"zeros_assign": lambda a=a, p=pad: _unguarded_zero_route(a, p)},
        check=byte_exact,
        samples=SAMPLES,
    )
    suite.measure(
        case=f"overcap_zero_n{n}_p{pad}_used",
        params={"n": n, "pad": pad, "out_len": out_len, "served": False,
                "timing": "consumed", "note": "the same cell, honestly: a loss"},
        baseline=("numpy.pad+sum", lambda a=a, p=pad: float(np.pad(a, (p, p)).sum())),
        candidates={"zeros_assign+sum":
                    lambda a=a, p=pad: float(_unguarded_zero_route(a, p).sum())},
        check=same_float,
        samples=SAMPLES,
    )

# --- 3. dtype spread at the headline shape ---------------------------------
# The saving is fixed Python machinery, not element work, so every measured
# dtype should land in the same band; a dtype that does not belongs out of
# the allowlist.

DTYPES = ([np.float64, np.int64] if SMOKE
          else [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8,
                np.uint64, np.uint32, np.uint16, np.uint8, np.bool_,
                np.complex128, np.complex64])

for dt in DTYPES:
    a = arr(64, dtype=dt)
    suite.measure(
        case=f"dtype_{np.dtype(dt).name}_n64",
        params={"n": 64, "pad": 3, "dtype": np.dtype(dt).name, "timing": "consumed"},
        baseline=("numpy.pad+sum", lambda a=a: float(np.pad(a, (3, 3)).sum())),
        candidates={"zeros_assign+sum": lambda a=a: float(_run(a, (3, 3)).sum())},
        check=same_float,
        samples=SAMPLES,
    )

# --- 4. pad_width spellings, to price the predicate's normalization --------

a = arr(64)
for label, pw in (("scalar", 3), ("pair", (2, 3)), ("nested", ((2, 3),)),
                  ("array", np.array([2, 3]))):
    suite.measure(
        case=f"spelling_{label}",
        params={"n": 64, "spelling": label, "timing": "bare"},
        baseline=("numpy.pad", lambda pw=pw: np.pad(a, pw)),
        candidates={"zeros_assign": lambda pw=pw: _run(a, pw)},
        check=byte_exact,
        samples=SAMPLES,
    )

suite.save()
