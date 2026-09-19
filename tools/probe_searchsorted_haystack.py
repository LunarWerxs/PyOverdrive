"""Does sorting the queries pay because of the QUERY count, or the HAYSTACK?

searchsorted_sortqueries gates on the query count - len(v) >= 10_000 for
float64, >= 100_000 for int64 - and asks only that the haystack clear a flat
10_000. But the mechanism it claims is a CACHE effect: consecutive sorted
queries land near each other, so the binary search stops missing. That effect
cannot exist while the haystack fits in cache, and a 10_000-element int64
haystack is 80 KB, which fits in L2 on every machine this project targets.

So the gate may be measuring the wrong axis. That exact defect shipped once
already this project - np.histogram2d gated on BIN count while its cost
scaled with SAMPLES, and lost 0.75x in the corner the gate was not looking
at (batch 14). The row sweep then found searchsorted's int64 row at 0.36x
at its own floor and 0.20x at 100x it, against an evidence comment that
records 1.51x at that same query count - a contradiction that only makes
sense if the two measurements used different haystacks.

This probe answers it directly: the ratio over a grid of (haystack, queries,
dtype), with the query count held while the haystack moves and vice versa.

Method is the sweep's: through the public API, result consumed, the two
sides interleaved, one cell per fresh process, and on a hybrid CPU only on
a fast core.

    .venv/Scripts/python tools/probe_searchsorted_haystack.py

VERDICT (2026-08-25, idle Intel box, fp 9bbe7063c555): THE HAYSTACK, and it
is not close. See docs/research/batch16-notes.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402

import pyoverdrive  # noqa: E402
from pyoverdrive import _cpuclass as cpuclass  # noqa: E402
from verify_no_pessimization import _measure  # noqa: E402

HAYSTACKS = (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)
QUERIES = (10_000, 100_000, 1_000_000)
DTYPES = ("float64", "int64")
SKEW_CAP = 10  # the shipped gate's own len(v) <= 10 * len(a)


def _build(n_a: int, n_v: int, dtype: str):
    rng = np.random.default_rng(4242)
    if dtype == "int64":
        span = 10 * max(n_a, n_v)
        a = np.sort(rng.integers(-span, span, size=n_a).astype(np.int64))
        v = rng.integers(-span, span, size=n_v).astype(np.int64)
    else:
        a = np.sort(rng.standard_normal(n_a))
        v = rng.standard_normal(n_v)
    return a, v


def _one(spec: str, fast_under: float | None) -> str:
    if fast_under is not None and cpuclass.probe_us() > fast_under:
        return "SLOW-CORE"
    n_a, n_v, dtype = spec.split(",")
    a, v = _build(int(n_a), int(n_v), dtype)
    pyoverdrive.enable()
    from pyoverdrive.dispatcher.gearbox import GEARBOX

    chosen, reason = GEARBOX.decide("numpy.searchsorted", (a, v), {})
    if chosen != "searchsorted_sortqueries":
        return f"NODISPATCH {reason}"
    ratio = _measure("numpy.searchsorted", (a, v), {}, rounds=9)
    return "UNMEASURABLE" if ratio is None else f"RATIO {ratio:.4f}"


def main(argv: list[str]) -> int:
    if len(argv) > 2 and argv[1] == "--one":
        under = float(argv[3]) if len(argv) > 3 else None
        print(_one(argv[2], under))
        return 0

    classes = cpuclass.classify()
    print(cpuclass.describe(classes))
    cutoff = cpuclass.fast_cutoff(classes)

    print(f"\n{'dtype':>8s} {'haystack':>10s} " +
          " ".join(f"{q:>12,d}" for q in QUERIES) + "   (queries)")
    for dtype in DTYPES:
        for n_a in HAYSTACKS:
            row = []
            for n_v in QUERIES:
                if n_v > SKEW_CAP * n_a:
                    row.append(f"{'skew':>12s}")
                    continue
                cmd = [sys.executable, str(Path(__file__)), "--one",
                       f"{n_a},{n_v},{dtype}"]
                if cutoff is not None:
                    cmd += [f"{cutoff:.3f}"]
                for _ in range(6):
                    proc = subprocess.run(cmd, capture_output=True, text=True,
                                          cwd=str(REPO))
                    line = ((proc.stdout or "").strip().splitlines()[-1:]
                            or [""])[0]
                    if line.strip() != "SLOW-CORE":
                        break
                if line.startswith("RATIO"):
                    row.append(f"{float(line.split()[1]):>11.2f}x")
                elif line.startswith("NODISPATCH"):
                    row.append(f"{'-':>12s}")
                else:
                    row.append(f"{line.split()[0][:12]:>12s}")
            print(f"{dtype:>8s} {n_a:>10,d} " + " ".join(row))
    print("\n'-' = the shipped predicate refuses that cell; 'skew' = beyond "
          "the gate's own 10:1 query/haystack cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
