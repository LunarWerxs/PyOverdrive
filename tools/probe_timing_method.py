"""Do the two timing methods in this repo disagree on a THREADED candidate?

The committed PyRallel table says np.sin float64 wins 1.63x at n=1e5. Measured
end to end through the public API on an idle machine, the same cell is 1.10x.
Both numbers are from the same box. Before re-deriving anything, find out
whether the gap is the ROUTE (candidate vs dispatched) or the CLOCK (how
lab/dyno times against how tools/calibrate_dispatch.py times), because those
call for completely different repairs.

The two methods differ in three ways, and each is measured separately here:

  ORDER      lab/dyno times all of the baseline's samples, then all of the
             candidate's. Anything that drifts across the run lands on one
             side. calibrate_dispatch interleaves.
  STATISTIC  both report a median of per-sample means, but lab/dyno also
             records min_ns, and a threaded candidate's minimum is the run
             where every worker happened to start at once - a best case a
             caller does not get.
  LOOPS      lab/dyno calibrates an inner loop count per variant;
             calibrate_dispatch derives one from the baseline and uses it
             for both.

Usage:
    .venv/Scripts/python tools/probe_timing_method.py [--op sin] [--n 100000]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import timeit
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from lab.dyno import _time_variant  # noqa: E402
from pyoverdrive.fastpaths._pyrallel_common import threads_for  # noqa: E402
from pyoverdrive.parallel import parallel_unary  # noqa: E402

DOMAINS = {
    "sin": (0.0, 2 * np.pi), "cos": (0.0, 2 * np.pi), "tan": (-1.5, 1.5),
    "exp": (-5.0, 5.0), "log": (0.1, 100.0), "log10": (0.1, 100.0),
    "tanh": (-4.0, 4.0), "sqrt": (0.0, 100.0),
}


def interleaved(s_fn, c_fn, rounds: int):
    for _ in range(10):
        s_fn()
        c_fn()
    t = timeit.timeit(s_fn, number=3) / 3
    number = max(1, min(500, int(0.01 / max(t, 1e-9))))
    st, ct = [], []
    for _ in range(rounds):
        st.append(timeit.timeit(s_fn, number=number) / number)
        ct.append(timeit.timeit(c_fn, number=number) / number)
    return (statistics.median(st) / statistics.median(ct),
            min(st) / min(ct))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="sin")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args(argv[1:])

    lo, hi = DOMAINS[args.op]
    u = getattr(np, args.op)
    x = np.linspace(lo, hi, args.n, dtype=np.dtype(args.dtype))
    t = threads_for(x.nbytes)
    print(f"{args.op} {args.dtype} n={args.n} threads={t}   "
          f"(repeated {args.repeat}x; the battery's own numbers are 'blocked')")
    print(f"{'run':>4s} {'blocked/median':>15s} {'blocked/min':>12s} "
          f"{'interleaved/med':>16s} {'interleaved/min':>16s}")

    for r in range(args.repeat):
        # the battery's shape: all of the baseline, THEN all of the candidate
        _, base = _time_variant(lambda: u(x), args.samples, 3)
        _, cand = _time_variant(lambda: parallel_unary(u, x, t), args.samples, 3)
        b_med = base["median_ns"] / cand["median_ns"]
        b_min = base["min_ns"] / cand["min_ns"]
        i_med, i_min = interleaved(lambda: u(x),
                                   lambda: parallel_unary(u, x, t),
                                   rounds=args.samples)
        print(f"{r + 1:>4d} {b_med:14.2f}x {b_min:11.2f}x "
              f"{i_med:15.2f}x {i_min:15.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
