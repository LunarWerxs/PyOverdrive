"""Is the SINGLE-THREADED baseline reproducible on this machine?

Committed evidence for np.sin float64 n=1e5 on fingerprint 9bbe7063c555
(idle, contended=false) records stock at 490.8 us and pyrallel_4t at 307.8 us
- a 1.59x win. Re-measured on the same box with the same code one day later,
the CANDIDATE is unchanged (~302 us) and STOCK is 337.7 us, which turns the
same cell into 1.10x. Only the baseline moved, and it moved by 45%.

A threading speedup is a RATIO, so an unstable baseline is not a small
problem: it is the whole number. And this box is an i7-12700K - 8
performance cores plus 4 efficiency cores. A single-threaded loop is one
thread the scheduler may place on either kind, and an E-core is far slower
per clock than a P-core. The threaded candidate spans several cores and
averages over the difference, so the same hardware quirk moves one side of
the ratio and not the other.

This measures the same single-threaded call in N fresh processes and prints
the distribution. A tight cluster means the baseline is trustworthy and the
day-to-day gap has some other cause. A bimodal split, or a spread of tens of
percent, means no ratio measured against an unpinned baseline on this machine
means anything until it is pinned.

Usage:
    .venv/Scripts/python tools/probe_baseline_stability.py [--runs 25]
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import timeit
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

DOMAINS = {"sin": (0.0, 2 * np.pi), "tanh": (-4.0, 4.0), "sqrt": (0.0, 100.0)}


def one(op: str, n: int, stat: str) -> float:
    lo, hi = DOMAINS[op]
    u = getattr(np, op)
    x = np.linspace(lo, hi, n)

    def fn():
        u(x)

    for _ in range(20):
        fn()
    t = timeit.timeit(fn, number=5) / 5
    number = max(1, min(500, int(0.01 / max(t, 1e-9))))
    times = [timeit.timeit(fn, number=number) / number for _ in range(9)]
    chosen = min(times) if stat == "min" else statistics.median(times)
    return chosen * 1e6


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="sin")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--runs", type=int, default=25)
    ap.add_argument("--stat", default="median", choices=("median", "min"))
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv[1:])

    if args.one:
        print(f"{one(args.op, args.n, args.stat):.3f}")
        return 0

    vals: list[float] = []
    for _ in range(args.runs):
        p = subprocess.run(
            [sys.executable, str(Path(__file__)), "--one", "--op", args.op,
             "--n", str(args.n), "--stat", args.stat],
            capture_output=True, text=True, cwd=str(REPO),
        )
        try:
            vals.append(float((p.stdout or "").strip().splitlines()[-1]))
        except Exception:  # noqa: BLE001
            pass

    vals.sort()
    if not vals:
        print("no measurements")
        return 1
    lo, hi = vals[0], vals[-1]
    med = statistics.median(vals)
    print(f"stock np.{args.op} float64 n={args.n}, {len(vals)} fresh processes, "
          f"{args.stat} of 9 rounds each")
    print(f"  min {lo:8.1f} us   median {med:8.1f} us   max {hi:8.1f} us   "
          f"spread {hi / lo:.2f}x")
    print("  sorted: " + "  ".join(f"{v:.0f}" for v in vals))
    # A gap of >15% between consecutive sorted values is the signature of two
    # populations rather than one noisy one.
    gaps = [(vals[i + 1] / vals[i], i) for i in range(len(vals) - 1)]
    ratio, idx = max(gaps)
    if ratio > 1.15:
        low, high = vals[:idx + 1], vals[idx + 1:]
        print(f"\n  BIMODAL: {len(low)} runs near {statistics.median(low):.0f} us, "
              f"{len(high)} near {statistics.median(high):.0f} us "
              f"({statistics.median(high) / statistics.median(low):.2f}x apart).")
        print("  A single-threaded baseline landing in two populations is what a "
              "hybrid\n  P-core/E-core scheduler looks like. Every ratio measured "
              "against it\n  inherits the split.")
    else:
        print(f"\n  UNIMODAL (largest consecutive gap {ratio:.2f}x): the baseline "
              "is stable here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
