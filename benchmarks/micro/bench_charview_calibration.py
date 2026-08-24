"""char_view calibration: the size floors OPP-000024 left open.

The OPP-000024 reproducer measured np.sort on U1 only at 10_000 (43.2x)
and 1e6 (25.1x), unique(return_counts=True) only at 10_000 (26.6x), and
index+inverse only at 10_000 (2.02x). S1 was never measured for sort or
counts at all. A dispatch predicate needs the floors below those points,
so this battery sweeps the small-n region for every route the fast path
would take:

- sort U1 (int32 view) at n in {100, 1_000, 3_000, 10_000}
- sort S1 (uint8 view) at n in {1_000, 10_000, 100_000} - first S1 sort
  measurements anywhere in this record
- unique+counts U1 at n in {300, 1_000, 3_000} and S1 at {1_000, 10_000}
- unique+index+inverse U1 at n in {1_000, 10_000}
- one hibyte-S1 sort case (bytes >= 0x80) as the int8-vs-uint8 trap
  detector on the sort route (the reproducer ran it only through unique)

Plain unique (no flags) is NOT re-measured: the reproducer already swept
it 10..1e6 (U1 1.31x at n=1000, 1.43-1.55x from 10_000; S1 1.54-1.73x).

Candidates are the exact routes the fast path will ship: view to
int32/uint8, stock sort/unique on the ints, view values back. The view is
a monotone bijection (U1 orders by its single UCS4 unit, non-negative in
int32; S1 orders as unsigned bytes), so every check here is EXACT
equality on every returned array, dtype included.

Result JSON: benchmarks/results/CHARVIEW-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_charview_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv

ALPHA10_U = np.array(list("ASDFGHJKLZ"), dtype="U1")
ALPHA10_S = np.frombuffer(b"ASDFGHJKLZ", dtype="S1")
HIBYTE10_S = np.frombuffer(
    bytes([0x00, 0x41, 0x5A, 0x7F, 0x80, 0x9C, 0xB5, 0xC8, 0xE0, 0xFF]),
    dtype="S1",
)

_VIEW = {"U1": np.int32, "S1": np.uint8}


def sort_view(x):
    return np.sort(x.view(_VIEW[str(x.dtype)[-2:]])).view(x.dtype)


def unique_view_counts(x):
    vals, counts = np.unique(x.view(_VIEW[str(x.dtype)[-2:]]), return_counts=True)
    return vals.view(x.dtype), counts


def unique_view_index_inverse(x):
    vals, idx, inv = np.unique(
        x.view(_VIEW[str(x.dtype)[-2:]]), return_index=True, return_inverse=True
    )
    return vals.view(x.dtype), idx, inv


def exact_equal(cand, base):
    if isinstance(base, tuple):
        return (
            isinstance(cand, tuple)
            and len(cand) == len(base)
            and cand[0].dtype == base[0].dtype
            and all(np.array_equal(c, b) for c, b in zip(cand, base))
        )
    return cand.dtype == base.dtype and np.array_equal(cand, base)


def draw(rng, alphabet, n):
    return alphabet[rng.integers(0, len(alphabet), size=n)]


def samples_for(n):
    if SMOKE:
        return 3
    return 7 if n >= 100_000 else 11


suite = BenchSuite("CHARVIEW-CAL", "U1/S1 int-view sort+unique: size floors")
rng = np.random.default_rng(24821)

# (route, dtype_key, alphabet, n)
if SMOKE:
    CASES = [("sort", "U1", ALPHA10_U, 1_000)]
else:
    CASES = (
        [("sort", "U1", ALPHA10_U, n) for n in (100, 1_000, 3_000, 10_000)]
        + [("sort", "S1", ALPHA10_S, n) for n in (1_000, 10_000, 100_000)]
        + [("sort", "S1hi", HIBYTE10_S, 10_000)]
        + [("counts", "U1", ALPHA10_U, n) for n in (300, 1_000, 3_000)]
        + [("counts", "S1", ALPHA10_S, n) for n in (1_000, 10_000)]
        + [("idxinv", "U1", ALPHA10_U, n) for n in (1_000, 10_000)]
        + [("idxinv", "S1", ALPHA10_S, n) for n in (1_000, 10_000)]
    )

for route, key, alphabet, n in CASES:
    x = draw(rng, alphabet, n)
    if route == "sort":
        baseline = ("numpy.sort", lambda x=x: np.sort(x))
        candidates = {"sort_view": lambda x=x: sort_view(x)}
    elif route == "counts":
        baseline = ("numpy.unique", lambda x=x: np.unique(x, return_counts=True))
        candidates = {"unique_view_counts": lambda x=x: unique_view_counts(x)}
    else:
        baseline = (
            "numpy.unique",
            lambda x=x: np.unique(x, return_index=True, return_inverse=True),
        )
        candidates = {"unique_view_idxinv": lambda x=x: unique_view_index_inverse(x)}
    suite.measure(
        case=f"{route}_{key}_n{n}",
        params={"route": route, "dtype": key, "n": n},
        baseline=baseline,
        candidates=candidates,
        check=exact_equal,
        samples=samples_for(n),
    )

if not SMOKE:
    suite.save()
