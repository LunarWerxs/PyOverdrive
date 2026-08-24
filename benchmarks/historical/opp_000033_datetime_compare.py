"""OPP-000033: datetime64 comparison/subtraction vs the int64 view of the
same buffers.

numpy/numpy#7487 (dmbelov, 2016, numpy 1.9.3 era): comparing a 2-D M8[m]
array against a COLUMN-broadcast slice of itself took 100 ms vs 14.4 ms
through .view('i8') - a DERIVED 6.94x - while the same-shape comparison
was only 1.23x and the ROW-broadcast one 1.02x, a shape dependency no
maintainer explained. Subtraction showed 119 vs 35.2 ms (DERIVED 3.38x).
Mechanism per shoyer: datetime loops lack the SIMD/unrolling tricks the
int64 loops have (plus a NaT check); both hypotheses unconfirmed.

What this reproducer measures on CURRENT numpy:

  1. The thread's exact construction: dates 1990-01-02 .. 2005-12-31
     daily (5842 days) x 1440 minutes, M8[m]; comparison and subtraction
     in all three shapes the thread timed (column-broadcast,
     row-broadcast, same-shape), stock vs i8 view.
  2. The i8 route AS A SHIPPED PREDICATE WOULD RUN IT: with a NaT scan
     of both operands included in the candidate timing (np.isnat().any()
     on each), since NaT diverges under the view (INT64_MIN compares
     normally; real NaT compares False everywhere) and the predicate
     must refuse NaT-bearing inputs. The scan-free view is ALSO timed as
     the ceiling.
  3. 1-D contiguous arrays at 1e6 (the common time-series shape the
     thread never measured).

NaT correctness is NOT probed here (a check would rightly fail; the
divergence is the documented reason for the guard) - it belongs to the
differential tests if this ships. All arrays here are NaT-free by
construction, matching the predicate's regime.

Comparison ufuncs return bool (no view-back); subtraction returns
m8[m] from the i8 route via .view('m8[m]'), bit-identical when NaT-free
and same-unit. Checks are exact equality, dtype included.

House rules: never imports pyoverdrive. Candidates call np.greater /
np.subtract on int64 operands only, which a datetime-dtype predicate
refuses, so a patched dispatch could not recurse.

Result JSON: benchmarks/results/OPP-000033/.
Run: .venv/Scripts/python benchmarks/historical/opp_000033_datetime_compare.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 7487
SMOKE = "--smoke" in sys.argv

I8 = np.dtype(np.int64)


def greater_view(a, b):
    return np.greater(a.view(I8), b.view(I8))


def greater_view_guarded(a, b):
    if np.isnat(a).any() or np.isnat(b).any():
        raise AssertionError("NaT present: predicate would refuse (not this data)")
    return np.greater(a.view(I8), b.view(I8))


_NAT_I8 = np.iinfo(np.int64).min


def greater_view_min_guarded(a, b):
    """The cheaper guard: NaT is INT64_MIN, so a SIMD int64 min-reduce
    detects it without materializing an isnat bool array. This is the
    guard a shipped predicate would actually use."""
    ai = a.view(I8)
    bi = b.view(I8)
    if ai.min() == _NAT_I8 or bi.min() == _NAT_I8:
        raise AssertionError("NaT present: predicate would refuse (not this data)")
    return np.greater(ai, bi)


def subtract_view(a, b, unit):
    return np.subtract(a.view(I8), b.view(I8)).view(unit)


def exact(cand, base):
    return cand.dtype == base.dtype and cand.shape == base.shape and bool(
        np.array_equal(cand, base)
    )


suite = BenchSuite("OPP-000033", "datetime64 compare/subtract: i8 view vs stock")

# the thread's own construction
date_d = np.arange(np.datetime64("1990-01-02"), np.datetime64("2005-12-31"))
time_t = np.arange(1440).view("m8[m]")
utc = date_d[:, None] + time_t[None, :]
if SMOKE:
    utc = utc[:200]
M8M = utc.dtype  # M8[m]
TDM = np.dtype("m8[m]")

SAMPLES = 3 if SMOKE else 9

SHAPES = [
    ("colbcast", utc, utc[:, 0:1]),   # the 6.94x claim shape
    ("rowbcast", utc, utc[0:1, :]),   # measured 1.02x in-thread
    ("sameshape", utc, utc),          # measured 1.23x in-thread
]

for label, a, b in SHAPES:
    suite.measure(
        case=f"greater_M8m_{label}_{a.shape[0]}x{a.shape[1]}",
        params={"shape": list(a.shape), "bshape": list(b.shape), "kind": label},
        baseline=("numpy.greater", lambda a=a, b=b: np.greater(a, b)),
        candidates={
            "i8_view": lambda a=a, b=b: greater_view(a, b),
            "i8_view_nat_guarded": lambda a=a, b=b: greater_view_guarded(a, b),
            "i8_view_min_guarded": lambda a=a, b=b: greater_view_min_guarded(a, b),
        },
        check=exact,
        samples=SAMPLES,
    )

suite.measure(
    case=f"subtract_M8m_colbcast_{utc.shape[0]}x{utc.shape[1]}",
    params={"shape": list(utc.shape), "kind": "colbcast", "op": "subtract"},
    baseline=("numpy.subtract", lambda a=utc: np.subtract(a, a[:, 0:1])),
    candidates={"i8_view": lambda a=utc: subtract_view(a, a[:, 0:1], TDM)},
    check=exact,
    samples=SAMPLES,
)

if not SMOKE:
    # 1-D contiguous at 1e6: the common time-series shape
    rng = np.random.default_rng(SEED)
    base_ns = np.datetime64("2020-01-01", "ns").astype("i8")
    stamps = (base_ns + np.sort(rng.integers(0, 10**15, size=1_000_000))).view("M8[ns]")
    pivot = np.full(1, stamps[500_000], dtype="M8[ns]")
    suite.measure(
        case="greater_M8ns_1d_n1000000",
        params={"n": 1_000_000, "kind": "1d-scalar-ish"},
        baseline=("numpy.greater", lambda a=stamps, b=pivot: np.greater(a, b)),
        candidates={
            "i8_view": lambda a=stamps, b=pivot: greater_view(a, b),
            "i8_view_nat_guarded": lambda a=stamps, b=pivot: greater_view_guarded(a, b),
            "i8_view_min_guarded": lambda a=stamps, b=pivot: greater_view_min_guarded(a, b),
        },
        check=exact,
        samples=9,
    )
    suite.save()
