"""OPP-000034: argmax along the slow (non-contiguous) axis of a C-order 2-D array.

numpy/numpy#9182: argmax does not support strided data, so
np.argmax(a, axis=0) on a C-order array first copies to make the reduced
axis contiguous (juliantaylor's root cause; cipri-tom traced the copy to
calculation.c). The thread's strongest number is from the final comment:
np.argmax(frames, axis=0) on a 10000x10000 float64 C-order array at
1.9 s/loop vs 96.7 ms/loop after np.asfortranarray(frames) - a DERIVED
19.65x (1900 / 96.7) that excludes the conversion cost. The reporter's
own per-column Python loop was only marginally faster (21.6 vs 22.9 s
for 10 runs, 1.06x) - the loop is NOT the candidate here.

Candidates:

  - relayout_argmax: np.argmax(np.ascontiguousarray(a.T), axis=1) -
    one explicit transpose-copy (the same copy stock performs
    internally, but done via the fast blocked relayout) followed by a
    fast-axis argmax. Result is EXACTLY stock's: same reduction axis,
    same scan order along it (increasing index), so first-occurrence
    ties and first-NaN semantics are preserved; check is exact equality
    dtype included.
  - fortran_control: np.argmax on a PRE-CONVERTED F-order copy of the
    same data (conversion NOT timed) - reproduces the thread's 19.65x
    framing and bounds the headroom; a transparent fast path cannot
    reach this number because it must pay the conversion, so this row
    is a reference ceiling, not a shippable candidate.

Cases: the thread's own 10000x10000, smaller/rectangular shapes to find
the floor, an axis=1 control (already the fast axis: candidate expected
to LOSE there, which is the anti-regime a predicate must refuse),
int64, float32, and a NaN-salted correctness case (stock argmax returns
the first NaN's index; the relayout route must match exactly).

House rules: never imports pyoverdrive. relayout_argmax calls
np.ascontiguousarray, which IS a patched name (relayout_blocked) - in
this script numpy is unpatched so it is a plain stock call; a shipped
fast path would call it through GEARBOX.stock_fn or benefit from the
relayout fast path deliberately (composition, to be decided at ship
time from these numbers).

Result JSON: benchmarks/results/OPP-000034/.
Run: .venv/Scripts/python benchmarks/historical/opp_000034_argmax_axis0.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 9182
SMOKE = "--smoke" in sys.argv


def relayout_argmax(a):
    return np.argmax(np.ascontiguousarray(a.T), axis=1)


def blocked_relayout_argmax(a, block=128):
    """The composed route a shipped path would take: the transpose copy
    done as a cache-blocked tile loop (the same technique PyOverdrive's
    relayout_blocked path ships, reimplemented here so this script stays
    pyoverdrive-free), then a fast-axis argmax. Stock's internal copy and
    plain ascontiguousarray(a.T) both walk the transpose cache-hostilely;
    the tiles are what the smoke run showed the route needs to win."""
    rows, cols = a.shape
    out = np.empty((cols, rows), dtype=a.dtype)
    for j0 in range(0, cols, block):
        j1 = min(j0 + block, cols)
        for i0 in range(0, rows, block):
            i1 = min(i0 + block, rows)
            out[j0:j1, i0:i1] = a[i0:i1, j0:j1].T
    return np.argmax(out, axis=1)


def exact(cand, base):
    return cand.dtype == base.dtype and cand.shape == base.shape and bool(
        np.array_equal(cand, base)
    )


suite = BenchSuite("OPP-000034", "argmax axis=0 on C-order 2-D: relayout route vs stock")
rng = np.random.default_rng(SEED)

# (rows, cols, dtype_label, dtype)
if SMOKE:
    CASES = [(1000, 1000, "float64", np.float64)]
    SAMPLES = 3
else:
    CASES = [
        (1000, 1000, "float64", np.float64),
        (10_000, 10_000, "float64", np.float64),
        (100_000, 100, "float64", np.float64),
        (100, 100_000, "float64", np.float64),
        (10_000, 10_000, "float32", np.float32),
        (10_000, 10_000, "int64", np.int64),
    ]
    SAMPLES = 9


def build(rows, cols, dtype):
    if np.issubdtype(dtype, np.integer):
        return rng.integers(-(2**40), 2**40, size=(rows, cols), dtype=dtype)
    return rng.random(size=(rows, cols), dtype=dtype)


for rows, cols, label, dtype in CASES:
    a = build(rows, cols, dtype)
    f = np.asfortranarray(a)
    samples = SAMPLES if a.size <= 10_000_000 else max(5, SAMPLES - 4)
    suite.measure(
        case=f"argmax_axis0_{rows}x{cols}_{label}",
        params={"rows": rows, "cols": cols, "dtype": label, "axis": 0},
        baseline=("numpy.argmax", lambda a=a: np.argmax(a, axis=0)),
        candidates={
            "relayout_argmax": lambda a=a: relayout_argmax(a),
            "blocked_relayout_argmax": lambda a=a: blocked_relayout_argmax(a),
            "fortran_control": lambda f=f: np.argmax(f, axis=0),
        },
        check=exact,
        samples=samples,
    )

if not SMOKE:
    # anti-regime: axis=1 is already the fast axis; the relayout route
    # (transposing to reduce along what becomes the slow axis) must lose
    a = build(10_000, 10_000, np.float64)
    suite.measure(
        case="argmax_axis1_10000x10000_float64_antiregime",
        params={"rows": 10_000, "cols": 10_000, "dtype": "float64", "axis": 1},
        baseline=("numpy.argmax", lambda a=a: np.argmax(a, axis=1)),
        candidates={
            "relayout_argmax_axis1": lambda a=a: np.argmax(
                np.ascontiguousarray(a), axis=1
            )
        },
        check=exact,
        samples=5,
    )

    # NaN correctness witness: stock returns the FIRST NaN index per column
    a = build(20_000, 200, np.float64)
    nan_rows = rng.integers(0, 20_000, size=200)
    a[nan_rows, np.arange(200)] = np.nan
    a[rng.integers(0, 20_000, size=200), np.arange(200)] = np.nan  # some cols get 2
    suite.measure(
        case="argmax_axis0_20000x200_nan_salted",
        params={"rows": 20_000, "cols": 200, "dtype": "float64", "nan": True},
        baseline=("numpy.argmax", lambda a=a: np.argmax(a, axis=0)),
        candidates={"relayout_argmax": lambda a=a: relayout_argmax(a)},
        check=exact,
        samples=9,
    )
    suite.save()
