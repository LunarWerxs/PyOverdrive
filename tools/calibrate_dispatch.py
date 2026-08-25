"""Re-derive the PyRallel thresholds from what a CALLER experiences.

Covers both PyRallel families: --family unary (parallel_ufunc.py, the
default) and --family binary (parallel_binary.py).

Their committed tables came from benchmarks/micro/bench_pyrallel_calibration.py
and its binary twin, which time ``parallel_unary(u, x, t)`` against ``u(x)``.
That is the candidate, and three things separate it from the call a user
makes. Two of them were suspects and turned out not to matter; the third
invalidated the whole table.

1. IT PICKS THE THREAD COUNT BY HAND. The battery sweeps t in {2,4,8,16} and
   the derivation reads the row for the scheduled count. Dispatch instead
   calls threads_for(nbytes). Those agree today, but only because someone
   keeps them agreeing.
2. IT NEVER GOES THROUGH THE PREDICATE. det 2x2 shipped at 0.70x for months
   because its guard recomputed the run's work, and no candidate-level
   battery could have seen it (tools/verify_no_pessimization.py). Measured
   here: for these paths candidate and dispatched agree, so the predicate is
   not where the error was.
3. ITS BASELINE IS A COIN FLIP ON A HYBRID CPU. This is the one. See the
   METHOD note below and docs/research/hybrid-cpu-baseline-coin-flip.md.

A fourth thing is not a defect in the battery but a genuine gap in what it
covers: its input is np.linspace, which is SORTED, and the speedup of a
threaded elementwise path depends on the data's ORDER as well as its size.
NumPy's own trig kernels run ~2.2x faster on sorted data (sin at n=1e5:
338 us sorted vs 753 us shuffled), leaving threading far less to win back;
and because PyRallel splits into CONTIGUOUS chunks, sorted input hands each
chunk a different sub-range of the domain and so unequal work - np.tan over
[-1.5, 1.5] at n=1e5 in 4 chunks measures 167/58/58/167 us, a 2.9x spread
capping the achievable speedup at 2.7x before a thread is created, against
1.3x for the same values shuffled. Neither order is the corner case:
np.sin(np.linspace(...)) is close to the canonical call. So this tool
measures both and sets each threshold from the WORSE one.

METHOD, and every clause of it was forced by a wrong number:

- Through the PUBLIC name, with pyoverdrive enabled, against
  GEARBOX.stock_fn - the dispatched route, predicate and all.
- ONE CELL PER PROCESS. Measuring many cells in one process reported a
  threaded path at 0.89x on four independent re-measurements when it was
  1.15x in isolation; other cells' allocations reach a threaded one through
  cache and allocator state. This is the same rule tools/verify_no_pessimization.py
  is built on.
- Rounds INTERLEAVED, never all-of-A then all-of-B, so drift cannot land on
  one side.
- Both data orders, and both bare and consumed. The reported ratio for a
  cell is the MINIMUM over {sorted, shuffled} x {bare, consumed}: all four
  are things a caller can hit, so the threshold has to hold for the worst.
- An IDLE machine. Threaded candidates genuinely lose under contention, so a
  busy box turns every marginal cell red - a true statement about a machine
  nobody ships on. The tool refuses to run above --max-load.
- MEASURED ONLY ON A FAST CORE, and this one invalidates the whole committed
  table on its own. On a hybrid CPU the scheduler gives a single-threaded
  process a P-core or an E-core and it stays there; asked for the same
  np.sin float64 n=1e5 in 25 fresh processes this box answered 344 us
  fifteen times and 497 us ten times, 1.44x apart, with almost no spread
  inside either group. A THREADED candidate spans cores and averages over
  the split, so it moves the denominator of every ratio and not the
  numerator. The committed evidence for that cell (stock 490.8 us,
  pyrallel_4t 307.8 us, 1.59x, recorded on a box its own conditions block
  calls idle) is a run that drew an E-core; re-measured with the candidate
  unchanged at ~302 us and stock at 344 us the same cell is 1.10x.
  Affinity is NOT the fix - pinning to the fast class costs the candidate
  its parallelism outright (the measured table in src/pyoverdrive/_cpuclass.py), so
  a pinned ratio is wrong in the other direction. Instead each cell probes
  the core it was handed and declines if it is a slow one, and the parent
  re-draws. The fast class is also the conservative choice: stock is
  quickest there, so a threaded path has the LEAST to offer, and a cell that
  clears the target there clears it everywhere.

Usage:
    .venv/Scripts/python tools/calibrate_dispatch.py [--target 1.30] [--json OUT]
    .venv/Scripts/python tools/calibrate_dispatch.py --ops sin,tan --sizes 100000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import timeit
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from pyoverdrive import _cpuclass as cpuclass  # noqa: E402

DOMAINS = {
    "sin": (0.0, 2 * np.pi),
    "cos": (0.0, 2 * np.pi),
    "tan": (-1.5, 1.5),
    "exp": (-5.0, 5.0),
    "log": (0.1, 100.0),
    "log10": (0.1, 100.0),
    "tanh": (-4.0, 4.0),
    "sqrt": (0.0, 100.0),
}
# The LAST size is a sentinel, not a candidate: derive() refuses a threshold
# that lands on it, because such a row rests on one point with nothing above
# it. So the sweep has to reach past the range you actually want to ship.
SIZES = (30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000,
         30_000_000)
DTYPES = ("float64", "float32")
ORDERS = ("sorted", "shuffled")

# The binary family (parallel_binary.py) came from the same battery and so
# inherits the same coin-flip baseline. Its kernels are bandwidth bound
# rather than compute bound, so data ORDER should not matter to them - both
# orders are measured anyway, because "should not matter" is a prediction.
BINARY_OPS = ("add", "subtract", "multiply", "divide", "maximum", "minimum")
BINARY_DTYPES = ("float64", "float32", "int64")
# Divisor kept away from zero; int64 divide is not in the shipped table.
BINARY_DOMAIN = (1.0, 2.0)


def make_args(op: str, n: int, dtype: str, order: str,
              family: str) -> tuple:
    """Sorted = np.linspace, what the old battery used. Shuffled = the same
    VALUES in random order, so the two differ in layout and in nothing else -
    identical multiset, identical dtype, identical size."""
    dt = np.dtype(dtype)
    lo, hi = BINARY_DOMAIN if family == "binary" else DOMAINS[op]

    def one(seed_shift: int) -> np.ndarray:
        if dt.kind == "i":
            x = np.arange(1, n + 1, dtype=dt) % 1000 + 1
        else:
            x = np.linspace(lo, hi, n, dtype=dt)
        if order == "shuffled":
            np.random.default_rng(seed_shift).shuffle(x)
        return np.ascontiguousarray(x)

    return (one(0), one(1)) if family == "binary" else (one(0),)


# --------------------------------------------------------------------------
# child: one cell, its own process
# --------------------------------------------------------------------------

def _ratio(stock_fn, cand_fn, rounds: int) -> float:
    for _ in range(10):
        stock_fn()
        cand_fn()
    t = timeit.timeit(stock_fn, number=3) / 3
    number = max(1, min(500, int(0.01 / max(t, 1e-9))))
    st: list[float] = []
    ct: list[float] = []
    for _ in range(rounds):
        st.append(timeit.timeit(stock_fn, number=number) / number)
        ct.append(timeit.timeit(cand_fn, number=number) / number)
    return sorted(st)[rounds // 2] / sorted(ct)[rounds // 2]


def _force_dispatch(op: str, dtype: str, family: str = "unary") -> None:
    """Lower this op's threshold to 1 element for the length of this process.

    Without it the tool could only ever ratchet a threshold UP: below the
    shipped floor the predicate refuses, the cell reports "no dispatch", and
    a size that would now win stays invisible. Re-deriving a table has to be
    able to move a number in both directions, so the predicate is rebuilt
    from a lowered copy of the row. Nothing else about the route changes -
    same run, same guard, same public name - and it lives in a calibration
    tool that never writes code.
    """
    from pyoverdrive.dispatcher.gearbox import GEARBOX
    from pyoverdrive.fastpaths import parallel_binary as PB
    from pyoverdrive.fastpaths import parallel_ufunc as PU

    mod = PB if family == "binary" else PU
    for path in GEARBOX._paths.get(f"numpy.{op}", []):
        if path.name == f"pyrallel_{op}":
            path.applicable = mod._make_applicable({np.dtype(dtype): 1})


def measure_one(op: str, n: int, dtype: str, order: str, rounds: int,
                force: bool = False, fast_under: float | None = None,
                family: str = "unary") -> dict:
    import pyoverdrive
    from pyoverdrive.dispatcher.gearbox import GEARBOX

    # On a hybrid CPU a fresh process is given a P-core or an E-core and
    # stays there, a 1.44x fork in the denominator of every ratio. Affinity
    # cannot be used to settle it (pinning strips the candidate of its
    # parallelism - the table in src/pyoverdrive/_cpuclass.py), so instead this
    # process asks which class it drew and declines the cell if it is the
    # slow one. The parent then retries and gets a new draw.
    if fast_under is not None:
        got = cpuclass.probe_us()
        if got > fast_under:
            return {"dispatch": None, "slow_core": round(got, 1)}

    args = make_args(op, n, dtype, order, family)
    pyoverdrive.enable()
    if force:
        _force_dispatch(op, dtype, family)
    name = f"numpy.{op}"
    patched = getattr(np, op)
    stock = GEARBOX.stock_fn(name)
    decided = GEARBOX.decide(name, args, {})[0]
    # decide() reports the string "stock" when no path takes the call - NOT
    # None. Testing for None here meant a below-threshold cell was timed
    # stock-against-stock and reported as a ~1.0x measurement OF THE FAST
    # PATH, which never ran. A cell that reads as a measured loss when
    # nothing was measured is the worst kind of wrong number.
    if decided != f"pyrallel_{op}":
        return {"dispatch": None, "declined": decided}

    bare = _ratio(lambda: stock(*args), lambda: patched(*args), rounds)
    used = _ratio(lambda: float(stock(*args).sum()),
                  lambda: float(patched(*args).sum()), rounds)
    if not np.array_equal(stock(*args), patched(*args), equal_nan=True):
        return {"dispatch": decided, "error": "results differ"}
    return {"dispatch": decided, "bare": bare, "consumed": used}


# --------------------------------------------------------------------------
# parent
# --------------------------------------------------------------------------

def cpu_load() -> float | None:
    """Average CPU load percent, or None if it cannot be read."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | "
             "Measure-Object -Property LoadPercentage -Average).Average"],
            capture_output=True, text=True, timeout=30,
        )
        return float(r.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


_ALIAS = {"float64": "_F64", "float32": "_F32", "int64": "_I64"}


def derive(rows: dict, target: float,
           dtypes: tuple[str, ...] = DTYPES) -> dict[str, dict[str, int]]:
    """Smallest size that clears TARGET and stays clear at every larger
    measured size. A single win with a loss above it is not a threshold: the
    table is read as 'at or above this size', so one red cell above the
    candidate floor disqualifies it."""
    table: dict[str, dict[str, int]] = {}
    for op in sorted({k[0] for k in rows}):
        for dtype in dtypes:
            sizes = sorted({k[2] for k in rows if k[0] == op and k[1] == dtype})
            floor = None
            for i, n in enumerate(sizes):
                if all(rows[(op, dtype, m)] >= target for m in sizes[i:]):
                    floor = n
                    break
            # A threshold that lands on the LARGEST size measured rests on a
            # single point with nothing above it to show the win persists.
            # House rule is that hardware decides and nobody extrapolates, so
            # that is not evidence enough to ship a row - extend the sweep
            # instead. Three float32 rows were caught by this on the first
            # unary run; all three ran at 1.06-1.15x at their old thresholds,
            # so nothing of value was lost.
            if floor is not None and len(sizes) > 1 and floor == sizes[-1]:
                floor = None
            if floor is not None:
                table.setdefault(op, {})[dtype] = floor
    return table


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=1.30)
    ap.add_argument("--family", default="unary", choices=("unary", "binary"))
    ap.add_argument("--ops")
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--dtypes")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--max-load", type=float, default=25.0)
    ap.add_argument("--json", help="write the raw cells here")
    ap.add_argument("--force-dispatch", action="store_true",
                    help="measure sizes BELOW the shipped floor too, so a "
                         "threshold can be lowered and not only raised")
    ap.add_argument("--any-core", action="store_true",
                    help="accept cells measured on a slow core too. Off by "
                         "default: on a hybrid CPU that makes the baseline a "
                         "coin flip and inflates every ratio by the class "
                         "ratio.")
    ap.add_argument("--retries", type=int, default=6,
                    help="re-draws allowed per cell when a process lands on a "
                         "slow core")
    ap.add_argument("--fast-under", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--one", help=argparse.SUPPRESS)
    args = ap.parse_args(argv[1:])

    if args.ops is None:
        args.ops = ",".join(BINARY_OPS if args.family == "binary" else DOMAINS)
    if args.dtypes is None:
        args.dtypes = ",".join(BINARY_DTYPES if args.family == "binary"
                               else DTYPES)

    if args.one:
        op, dtype, n, order = args.one.split(":")
        print(json.dumps(measure_one(op, int(n), dtype, order, args.rounds,
                                     force=args.force_dispatch,
                                     fast_under=args.fast_under,
                                     family=args.family)))
        return 0

    load = cpu_load()
    if load is not None and load > args.max_load:
        print(f"REFUSING: CPU load is {load:.0f}%, above --max-load "
              f"{args.max_load:.0f}%.\nThreaded candidates genuinely lose under "
              f"contention; a busy box turns every marginal cell red.\n"
              f"Run this on an idle machine, or raise --max-load deliberately.")
        return 2
    print(f"cpu load {load if load is None else f'{load:.0f}%'}   "
          f"target {args.target:g}x   rounds {args.rounds}")

    classes = cpuclass.classify()
    print(cpuclass.describe(classes))
    cutoff = None if args.any_core else cpuclass.fast_cutoff(classes)
    if cutoff is not None:
        print(f"rejecting any cell whose process draws a slow core "
              f"(probe > {cutoff:.0f} us); up to {args.retries} re-draws each")
    elif classes.get("hybrid"):
        print("WARNING: hybrid CPU and --any-core: the single-threaded "
              "baseline is a coin flip and every ratio below inherits it.")

    ops = args.ops.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    dtypes = args.dtypes.split(",")

    raw: list[dict] = []
    worst: dict[tuple, float] = {}
    unmeasured: list[tuple] = []
    hdr = f"{'op':6s} {'dtype':8s} {'n':>9s}  " + "  ".join(
        f"{o[:4]+'/'+w[:4]:>11s}" for o in ORDERS for w in ("bare", "consumed")
    ) + f"  {'WORST':>7s}"
    print(hdr)
    for op in ops:
        for dtype in dtypes:
            # np.divide on integer operands produces float64, so the shipped
            # predicate refuses it (out dtype != operand dtype) and it has no
            # row to re-derive. --force-dispatch would push it through a path
            # that was never written for it, so it is skipped outright rather
            # than measured into a number nobody should act on.
            if op == "divide" and np.dtype(dtype).kind in "iu":
                continue
            for n in sizes:
                cells: dict[str, dict] = {}
                for order in ORDERS:
                    cmd = [sys.executable, str(Path(__file__)),
                           "--one", f"{op}:{dtype}:{n}:{order}",
                           "--rounds", str(args.rounds),
                           "--family", args.family]
                    if args.force_dispatch:
                        cmd.append("--force-dispatch")
                    if cutoff is not None:
                        cmd += ["--fast-under", f"{cutoff:.3f}"]
                    # Re-draw until this cell lands on a fast core. A process
                    # keeps whatever class it was given, so the only way to
                    # choose is to spawn again.
                    for _ in range(max(1, args.retries)):
                        p = subprocess.run(cmd, capture_output=True, text=True,
                                           cwd=str(REPO))
                        try:
                            got = json.loads((p.stdout or "").strip().splitlines()[-1])
                        except Exception:  # noqa: BLE001
                            got = {"dispatch": None, "error": (p.stderr or "")[-160:]}
                        if "slow_core" not in got:
                            break
                    cells[order] = got
                vals = [c[k] for c in cells.values()
                        for k in ("bare", "consumed") if k in c]
                if not vals:
                    why = next((c.get("error") or c.get("slow_core")
                                or c.get("declined") for c in cells.values()
                                if c.get("dispatch") is None), "no dispatch")
                    # A cell that produced nothing drops OUT of the size list
                    # derive() walks, so the threshold would be chosen from a
                    # sequence with an invisible hole in it. Record it and say
                    # so before the table: a gap must never read as a pass.
                    unmeasured.append((op, dtype, n, str(why)[:60]))
                    print(f"{op:6s} {dtype:8s} {n:9d}  "
                          f"{'(not measured: ' + str(why)[:32] + ')':>50s}")
                    continue
                w = min(vals)
                worst[(op, dtype, n)] = w
                raw.append({"op": op, "dtype": dtype, "n": n, "cells": cells,
                            "worst": w})
                shown = "  ".join(
                    f"{cells[o].get(k, float('nan')):10.2f}x"
                    for o in ORDERS for k in ("bare", "consumed")
                )
                flag = " " if w >= args.target else "!"
                print(f"{op:6s} {dtype:8s} {n:9d}  {shown}  {w:6.2f}x{flag}")

    table = derive(worst, args.target, tuple(dtypes))
    if unmeasured:
        print(f"\n!! {len(unmeasured)} cell(s) produced no measurement. Each "
              f"is a HOLE in the size sequence a threshold is read from, so "
              f"treat any row below that spans one as unproven:")
        for op, dtype, n, why in unmeasured:
            print(f"   {op} {dtype} n={n}: {why}")
    print(f"\nthresholds at >= {args.target:g}x on the WORST of "
          f"{{sorted, shuffled}} x {{bare, consumed}}:")
    print("SUPPORTED: dict[str, dict[np.dtype, int]] = {")
    print("# rows whose only qualifying size was the largest measured are "
          "EXCLUDED:\n# one point, nothing above it. Extend --sizes to ship "
          "them.")
    for op in ops:
        if op not in table:
            print(f'    # "{op}": no size clears the target -> stays on stock')
            continue
        entries = ", ".join(
            f"{_ALIAS[d]}: {table[op][d]:_}" for d in dtypes if d in table[op]
        )
        print(f'    "{op}": {{{entries}}},')
    print("}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"target": args.target, "load": load,
             "fast_core_only": cutoff is not None,
             "cpu_classes": classes, "cells": raw, "table": table},
            separators=(",", ":")), encoding="utf-8")
        print(f"\nraw cells -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
