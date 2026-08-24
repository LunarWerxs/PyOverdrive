"""OPP-000040: unique(axis=0) on integer rows, void-view route vs stock.

numpy/numpy#11136: unique(axis=0) is needlessly slow; bjornmadsen (2026,
numpy 2.3.5) measured a raw void-view route at 9.54M rows/s vs stock
axis=0 at 4.87M rows/s (DERIVED 1.96x) on 10k (int, int) rows; nschloe's
2018 "factor of 3" started the thread. Stock's own axis=0 machinery
ALSO goes through a void view internally - but a PRE-BATTERY PROBE
(2026-08-24, numpy 2.4.5) showed stock's OUTPUT ORDER is numeric
lexicographic ([-5, 2] before [0, 0]) while a raw little-endian void
view memcmp-sorts to garbage on signed ints, so the raw void route is
DEAD for bit-identity and the candidate here is the LEXSORT route
instead: np.lexsort on the columns (last column least significant),
gather, adjacent-diff row mask - numeric lexicographic by
construction, exactly stock's order. Negative values stay salted in as
the standing order-semantics witness.

Float/string rows are OUT of scope: void equality is bit-pattern
equality, which diverges from value equality on NaN and negative zero
(eric-wieser's in-thread hazard).

Candidates:

  - lexsort_unique: np.lexsort over the columns, gather the sorted
    rows, adjacent-diff mask across columns, slice. return_counts
    variant via diff of the mask indices.

Cases: (n, k) grids n in {1_000, 10_000, 100_000, 1_000_000}, k in
{2, 4, 8}, int64; int32 at one point; low-cardinality (many duplicate
rows) and high-cardinality variants; negative-salted correctness cells.

House rules: never imports pyoverdrive. The candidate calls np.lexsort and
elementwise compares only (np.unique never), so a patched dispatch
could not recurse.

Result JSON: benchmarks/results/OPP-000040/.
Run: .venv/Scripts/python benchmarks/historical/opp_000040_unique_rows_void.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 11136
SMOKE = "--smoke" in sys.argv


def lexsort_unique(a, return_counts=False):
    n, k = a.shape
    order = np.lexsort(tuple(a[:, j] for j in range(k - 1, -1, -1)))
    srt = a[order]
    mask = np.empty(n, dtype=bool)
    mask[0] = True
    np.any(srt[1:] != srt[:-1], axis=1, out=mask[1:])
    if not return_counts:
        return srt[mask]
    idx = np.flatnonzero(mask)
    counts = np.diff(np.append(idx, n))
    return srt[mask], counts


def exact(cand, base):
    if isinstance(base, tuple):
        return (
            isinstance(cand, tuple)
            and len(cand) == len(base)
            and all(
                c.dtype == b.dtype and np.array_equal(c, b) for c, b in zip(cand, base)
            )
        )
    return cand.dtype == base.dtype and cand.shape == base.shape and bool(
        np.array_equal(cand, base)
    )


suite = BenchSuite("OPP-000040", "unique(axis=0) int rows: void view vs stock")
rng = np.random.default_rng(SEED)

if SMOKE:
    GRID = [(10_000, 2, "lo")]
    SAMPLES = 3
else:
    GRID = [(n, k, card) for n in (1_000, 10_000, 100_000, 1_000_000)
            for k in (2, 4) for card in ("lo",)]
    GRID += [(100_000, 8, "lo"), (100_000, 2, "hi"), (100_000, 4, "hi")]
    SAMPLES = 9


def rows(n, k, card):
    hi = 50 if card == "lo" else 2**40
    # negative values salted: two's-complement memcmp order is the
    # decisive order-semantics probe (see docstring)
    return rng.integers(-hi, hi, size=(n, k), dtype=np.int64)


for n, k, card in GRID:
    a = rows(n, k, card)
    suite.measure(
        case=f"unique_axis0_n{n}_k{k}_{card}card_int64",
        params={"n": n, "k": k, "cardinality": card, "dtype": "int64"},
        baseline=("numpy.unique", lambda a=a: np.unique(a, axis=0)),
        candidates={"lexsort_unique": lambda a=a: lexsort_unique(a)},
        check=exact,
        samples=SAMPLES if n <= 100_000 else 5,
    )

if not SMOKE:
    a = rows(100_000, 3, "lo").astype(np.int32)
    suite.measure(
        case="unique_axis0_n100000_k3_locard_int32",
        params={"n": 100_000, "k": 3, "cardinality": "lo", "dtype": "int32"},
        baseline=("numpy.unique", lambda a=a: np.unique(a, axis=0)),
        candidates={"lexsort_unique": lambda a=a: lexsort_unique(a)},
        check=exact,
        samples=9,
    )
    a = rows(100_000, 2, "lo")
    suite.measure(
        case="unique_axis0_n100000_k2_counts",
        params={"n": 100_000, "k": 2, "returns": "values+counts", "dtype": "int64"},
        baseline=("numpy.unique", lambda a=a: np.unique(a, axis=0, return_counts=True)),
        candidates={"lexsort_unique": lambda a=a: lexsort_unique(a, return_counts=True)},
        check=exact,
        samples=9,
    )
    suite.save()
