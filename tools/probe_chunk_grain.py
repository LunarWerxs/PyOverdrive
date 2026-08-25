"""Can the sorted-data loss be recovered, and at what price?

PyRallel gives each thread ONE contiguous range. On sorted input each range
covers a different sub-range of the domain, so when the kernel's cost depends
on its argument the threads get unequal work and the call waits for the
slowest. Measured per-range at n=100k with 4 threads: np.tan on
linspace over [-1.5, 1.5] is 167/58/58/167 us - a 2.9x spread that caps the
achievable speedup at 2.7x before a thread is even created. Shuffle the same
values and the spread falls to 1.3x.

Sorted input is not a corner case for these ops. ``np.sin(np.linspace(...))``
is close to the canonical way the function gets called.

Two repairs are worth measuring, and they are not equally cheap:

  OVERSUBSCRIBE  cut the array into grain*threads ranges and submit them all,
                 letting the executor hand them out as workers free up. True
                 dynamic balancing, but every extra range costs a submit, a
                 Future and a wait - order 5-10 us each, which at n=100k is
                 a large fraction of the whole call.

  BLOCK-CYCLIC   cut into the same grain*threads ranges but assign them
                 statically, worker w taking blocks w, w+T, w+2T, ... Each
                 worker still gets ONE task, so the submit count does not
                 change at all; the only new cost is grain-1 extra ufunc
                 calls per worker, order 1-2 us each. It balances any smooth
                 cost profile because every worker is spread across the whole
                 domain, and unlike a shared work queue it needs no lock and
                 no GIL-atomicity assumption - which matters, because a
                 free-threaded build would silently duplicate or skip ranges
                 if the scheduling relied on that.

Both are bit-identical to stock by construction (elementwise kernels have no
cross-element data flow); the probe asserts it per cell rather than assuming.

VERDICT (2026-08-24, idle reference box): REJECTED, one chunk per thread
stays. Oversubscribe collapses where it matters - at n=1e5 with 4 threads it
runs 1.13x, 0.97x, 0.56x, 0.36x, 0.20x as the grain rises, because a submit
plus a Future costs 5-10 us against a ~300 us call. Block-cyclic is far
better behaved and still does not pay: across 8 ops x 2 orders at n=3e5 a
grain of 2 beat grain 1 in 3 cases and lost in 5, and a grain of 4 beat it
in 2 and lost in 6, with the apparent wins inside the run-to-run spread. The
imbalance is real but it is worst exactly where stock is also fastest, so
the absolute time on the table is small next to the per-piece overhead.
Kept as the instrument that produced that answer, and to re-check it on
hardware with a different core count or memory system.

Usage:
    .venv/Scripts/python tools/probe_chunk_grain.py [--n 100000] [--ops tan,sin]
"""

from __future__ import annotations

import argparse
import sys
import timeit
from concurrent.futures import wait as _wait_all
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from pyoverdrive.fastpaths._pyrallel_common import threads_for  # noqa: E402
from pyoverdrive.parallel import pyrallel  # noqa: E402

DOMAINS = {
    "sin": (0.0, 2 * np.pi), "cos": (0.0, 2 * np.pi), "tan": (-1.5, 1.5),
    "exp": (-5.0, 5.0), "log": (0.1, 100.0), "log10": (0.1, 100.0),
    "tanh": (-4.0, 4.0), "sqrt": (0.0, 100.0),
}
GRAINS = (1, 2, 4, 8, 16)


def _blocks(size: int, n: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, size, n + 1).astype(np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


def _run_ranges(ufunc, xf, of, ranges, err) -> None:
    with np.errstate(**err):
        for a, b in ranges:
            ufunc(xf[a:b], out=of[a:b])


def par_cyclic(ufunc, x, threads: int, grain: int):
    """Block-cyclic: grain*threads blocks, worker w takes w::threads.
    Submit count stays at `threads` no matter how fine the blocks get."""
    out = np.empty(x.shape, dtype=ufunc.resolve_dtypes((x.dtype, None))[-1])
    xf, of = x.reshape(-1), out.reshape(-1)
    blocks = _blocks(of.size, threads * grain)
    err = np.geterr()
    ex = pyrallel._pool()
    futures = [ex.submit(_run_ranges, ufunc, xf, of, blocks[w::threads], err)
               for w in range(threads)]
    _wait_all(futures)
    for f in futures:
        f.result()
    return out


def par_oversub(ufunc, x, threads: int, grain: int):
    """Oversubscribed: one submit per block, dynamic hand-out by the pool."""
    out = np.empty(x.shape, dtype=ufunc.resolve_dtypes((x.dtype, None))[-1])
    xf, of = x.reshape(-1), out.reshape(-1)
    blocks = _blocks(of.size, threads * grain)
    err = np.geterr()
    ex = pyrallel._pool()
    futures = [ex.submit(_run_ranges, ufunc, xf, of, [b], err) for b in blocks]
    _wait_all(futures)
    for f in futures:
        f.result()
    return out


def t_us(fn, reps=5) -> float:
    for _ in range(15):
        fn()
    t = timeit.timeit(fn, number=3) / 3
    number = max(1, min(300, int(0.01 / max(t, 1e-9))))
    return min(timeit.timeit(fn, number=number) / number for _ in range(reps)) * 1e6


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--ops", default=",".join(DOMAINS))
    ap.add_argument("--orders", default="sorted,shuffled")
    args = ap.parse_args(argv[1:])

    n = args.n
    threads = threads_for(n * 8)
    print(f"n={n}  threads={threads}   speedup over stock; grain 1 == what ships today")
    head = "  ".join(f"{'cyc' + str(g):>7s}" for g in GRAINS)
    head2 = "  ".join(f"{'ovr' + str(g):>7s}" for g in GRAINS)
    print(f"{'op':6s} {'order':>8s} {'stock us':>9s}  {head}   {head2}")

    for op_name in args.ops.split(","):
        u = getattr(np, op_name)
        lo, hi = DOMAINS[op_name]
        for order in args.orders.split(","):
            x = np.linspace(lo, hi, n)
            if order == "shuffled":
                np.random.default_rng(0).shuffle(x)
            x = np.ascontiguousarray(x)
            expect = u(x)
            stock_us = t_us(lambda: u(x))

            cells = []
            for fn in (par_cyclic, par_oversub):
                row = []
                for g in GRAINS:
                    got = fn(u, x, threads, g)
                    if not np.array_equal(got, expect, equal_nan=True):
                        row.append("DIFFER")
                        continue
                    row.append(f"{stock_us / t_us(lambda f=fn, g=g: f(u, x, threads, g)):6.2f}x")
                cells.append("  ".join(f"{c:>7s}" for c in row))
            print(f"{op_name:6s} {order:>8s} {stock_us:9.1f}  {cells[0]}   {cells[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
