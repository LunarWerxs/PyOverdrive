"""Prove no fast path is SLOWER than stock on its own canonical input.

A fast path that dispatches into a loss is the worst failure this project
can have, and nothing else catches it. The tests check correctness. The
benchmark batteries time the CANDIDATE, which is not the same thing as the
dispatched route - the candidate has no predicate in front of it and no
guard inside it. So a path can advertise a large win, pass every test, and
still make a user's call slower.

That is not hypothetical here. Sweeping end to end found det 2x2 running at
0.70x AT ITS OWN ADVERTISED FLOOR, slogdet 3x3 at 0.96x, and solve at 0.91x
and 0.94x - four dispatched losses across three shipped paths, all of them
because a guard recomputed work the run was about to do anyway.

Method: every path's OWN selfcheck input (the canonical input its author
chose), timed through the PUBLIC API with pyoverdrive enabled, against the
same call with it disabled, with the result consumed so nothing hides in a
lazy allocation. A path is only judged if it actually dispatches on that
input.

ISOLATION IS THE WHOLE METHOD. Every path is measured in its own fresh
process, and that is not caution, it is what makes the number mean
anything. Measuring all 69 in one process reported pyrallel_tan at 0.89x -
reproducibly, four independent re-measures in a row - when the same path on
the same array on the same idle machine is 1.15x in isolation, in every
configuration tried (only-tan enabled, everything enabled, thread pool
already warm). Sixty-eight other paths' allocations reach the threaded ones
through cache and allocator state, and no amount of re-measuring inside
that process escapes it.

The two sides are also timed INTERLEAVED rather than in separate blocks, so
anything that drifts across a measurement cannot land on one side only.
That alone fixed two earlier false reds.

AND, on a hybrid CPU, ONLY ON A FAST CORE - because otherwise this tool
returns FALSE GREENS. A single-threaded stock call runs on whatever core
class the process was handed and stays there; a threaded fast path spans
cores and averages over the difference. So a process that draws an
efficiency core has a slow stock side and a normal patched side, which
flatters the ratio by the class ratio - 1.44x on the box this was found on.
A threaded path that genuinely runs at 0.9x reports 1.3x and passes. Every
process here probes the core it was given and re-draws if it is a slow one
(src/pyoverdrive/_cpuclass.py); the fast class is also the honest one to judge on,
since stock is quickest there and a fast path has the least to offer.

Two limits worth stating even so.

First, this catches gross losses, not the difference between 0.98x and
1.02x. Run it on an idle machine, and settle anything within a few percent
of 1.0 with a dedicated probe.

Second, coverage is per CELL, not per path, and the difference matters. It
used to probe one canonical input per path, which is not coverage where a
path's table spans several dtypes: the selfcheck input picks float64 when
there is one, and pyrallel_subtract passed at 1.13x on its float64 floor
while its float32 row was running at 0.97x. The two PyRallel families are
the only modules here with a dtype-keyed threshold table, so they now
contribute one cell per dtype, each at that dtype's own floor.

What is still NOT covered, and should not be mistaken for covered: every
other path is judged on a single input, and no path is judged at more than
one SIZE. A loss that only appears at some shape inside a regime will not
show up here. The instrument for that is a full sweep
(tools/calibrate_dispatch.py), not this.

Usage:
    .venv/Scripts/python tools/verify_no_pessimization.py [--min 1.0] [-v]

Exit 0 = every dispatching path is at least --min. Exit 1 = at least one
dispatches into a loss.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import timeit
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from pyoverdrive import _cpuclass as cpuclass  # noqa: E402

import pyoverdrive  # noqa: E402
from pyoverdrive import diagnostics as D  # noqa: E402
from pyoverdrive.dispatcher.gearbox import GEARBOX  # noqa: E402


def _consume(value) -> None:
    """Touch the result so a lazily allocated one cannot look free."""
    if isinstance(value, tuple):
        for v in value:
            _consume(v)
        return
    arr = np.asarray(value)
    if arr.dtype.kind in "biufc" and arr.size:
        float(np.asarray(arr.sum()).real)


def _resolve(op: str):
    holder = np
    parts = op.split(".")
    for p in parts[1:-1]:
        holder = getattr(holder, p)
    return holder, parts[-1]


def _measure(op: str, args, kwargs, rounds: int) -> float | None:
    """Median ratio, measured with the two sides INTERLEAVED.

    Timing all of stock's rounds and then all of the patched rounds looks
    equivalent and is not: anything that drifts over the run - thermal
    state, a thread pool spinning down, another process arriving - lands on
    one side only. That produced two false reds here, reporting threaded
    paths at 0.91x and 0.99x that an alternating probe measured at 1.17x
    and 1.18x on the same idle machine. A pessimization detector that cries
    wolf is worse than none, so the rounds alternate.
    """
    holder, name = _resolve(op)
    stock = GEARBOX.stock_fn(op)
    patched = getattr(holder, name)

    def run_stock():
        _consume(stock(*args, **kwargs))

    def run_patched():
        _consume(patched(*args, **kwargs))

    try:
        run_stock()
        run_patched()
    except Exception:
        return None  # a path whose canonical input raises is selfcheck's problem

    t = timeit.timeit(run_stock, number=1)
    number = 1 if t > 0.02 else max(1, int(0.02 / max(t, 1e-9)))
    for _ in range(max(2, number // 4)):
        run_stock()
        run_patched()

    stock_times, patched_times = [], []
    for _ in range(rounds):
        stock_times.append(timeit.timeit(run_stock, number=number) / number)
        patched_times.append(timeit.timeit(run_patched, number=number) / number)
    s = sorted(stock_times)[rounds // 2]
    c = sorted(patched_times)[rounds // 2]
    return s / c if c > 0 else None


def cells() -> list[str]:
    """Every cell to judge, as "path" or "path@dtype".

    One canonical input per path is not coverage where a path's table spans
    several dtypes: the selfcheck input picks float64 when there is one, and
    pyrallel_subtract passed at 1.13x on float64 while its float32 row was
    running at 0.97x. The two PyRallel families are the only modules here
    with a dtype-keyed threshold table, so they get one cell PER DTYPE, each
    at that dtype's own floor - the weakest size its predicate admits.
    """
    from pyoverdrive.fastpaths import parallel_binary, parallel_ufunc

    tabled: dict[str, list[str]] = {}
    for mod in (parallel_ufunc, parallel_binary):
        for op, row in mod.SUPPORTED.items():
            tabled[f"pyrallel_{op}"] = [np.dtype(d).name for d in row]

    out = []
    for name in sorted(D._selfcheck_inputs()):
        dtypes = tabled.get(name)
        out.extend([f"{name}@{d}" for d in dtypes] if dtypes else [name])
    return out


def _inputs_for(cell: str):
    """(path name, maker) for a cell, expanding a "path@dtype" cell onto the
    row's own floor rather than whatever the selfcheck input would pick."""
    if "@" not in cell:
        return cell, D._selfcheck_inputs().get(cell)
    name, dtype_name = cell.split("@", 1)
    from pyoverdrive.fastpaths import parallel_binary, parallel_ufunc

    op = name[len("pyrallel_"):]
    for mod, builder in ((parallel_ufunc, D._inputs_ufunc),
                         (parallel_binary, D._inputs_binary)):
        row = mod.SUPPORTED.get(op)
        if row is None:
            continue
        want = {d: n for d, n in row.items() if np.dtype(d).name == dtype_name}
        if want:
            return name, builder(op, want)
    return name, None


def _measure_one(cell: str, fast_under: float | None = None) -> str:
    """Measure a single cell. Runs as its own process; prints one line."""
    if fast_under is not None and cpuclass.probe_us() > fast_under:
        return "SLOW-CORE"
    pyoverdrive.enable()
    name, make = _inputs_for(cell)
    if make is None:
        return "SKIP no-input"
    try:
        call_args, call_kwargs = make()
    except Exception as exc:  # noqa: BLE001
        return f"SKIP input-error {exc!r}"

    paths = [
        p
        for lst in GEARBOX._paths.values()
        for p in (lst if isinstance(lst, list) else [lst])
        if p.name == name
    ]
    if not paths:
        return "SKIP unknown"
    op = paths[0].op
    if GEARBOX.decide(op, call_args, call_kwargs)[0] != name:
        return "SKIP no-dispatch"

    ratio = _measure(op, call_args, call_kwargs, rounds=9)
    if ratio is None:
        return "SKIP unmeasurable"
    return f"RATIO {ratio:.4f} {op}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=1.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--any-core", action="store_true",
                    help="accept measurements taken on a slow core too; on a "
                         "hybrid CPU that can hide a threaded pessimization")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--fast-under", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--one", help=argparse.SUPPRESS)  # internal: one path, own process
    args = ap.parse_args(argv[1:])

    if args.one:
        print(_measure_one(args.one, args.fast_under))
        return 0

    # EVERY PATH GETS A FRESH PROCESS. Measuring 69 paths in one process
    # produced a red that reproduced four times and was still wrong:
    # pyrallel_tan reported 0.89x in the sweep and 1.15x in isolation, in
    # every configuration tried (only-tan enabled, everything enabled, pool
    # already warm). Sixty-eight other paths' allocations get to the
    # threaded ones through cache and allocator state, and no amount of
    # re-measuring inside that process escapes it. A per-path subprocess is
    # slower and is the only way the number means anything.
    pyoverdrive.enable()
    live = {
        p.name
        for lst in GEARBOX._paths.values()
        for p in (lst if isinstance(lst, list) else [lst])
        if p.enabled
    }
    names = [c for c in cells() if c.split("@", 1)[0] in live]
    pyoverdrive.disable()

    classes = cpuclass.classify()
    print(cpuclass.describe(classes))
    cutoff = None if args.any_core else cpuclass.fast_cutoff(classes)
    if cutoff is not None:
        print(f"measuring only on the fast class (probe <= {cutoff:.0f} us), "
              f"up to {args.retries} re-draws per path")

    losses: list[tuple[str, str, float]] = []
    judged = skipped = 0
    for name in names:
        cmd = [sys.executable, str(Path(__file__)), "--one", name]
        if cutoff is not None:
            cmd += ["--fast-under", f"{cutoff:.3f}"]
        for _ in range(max(1, args.retries)):
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=str(REPO))
            line = (proc.stdout or "").strip().splitlines()[-1:] or [""]
            if line[0].strip() != "SLOW-CORE":
                break
        parts = line[0].split()
        if not parts or parts[0] != "RATIO":
            skipped += 1
            if args.verbose:
                print(f"  {'skip':>8s}  {name:28s} {' '.join(parts[1:])}")
            continue
        judged += 1
        ratio, op = float(parts[1]), parts[2]
        if ratio < args.min:
            losses.append((name, op, ratio))
        if args.verbose:
            print(f"  {ratio:6.2f}x  {name:28s} {op}")

    print(f"\njudged {judged} dispatching paths in their own processes, "
          f"skipped {skipped} (no canonical input, or it does not dispatch)")
    if losses:
        print(f"PESSIMIZATION: {len(losses)} path(s) below {args.min:g}x")
        for name, op, ratio in sorted(losses, key=lambda r: r[2]):
            print(f"  !! {ratio:5.2f}x  {name}  ({op})")
        return 1
    print(f"no dispatching path is below {args.min:g}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
