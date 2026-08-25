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

A limit worth stating even so: this catches gross losses, not the
difference between 0.98x and 1.02x. Run it on an idle machine, and settle
anything within a few percent of 1.0 with a dedicated probe.

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
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

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


def _measure_one(name: str) -> str:
    """Measure a single path. Runs as its own process; prints one line."""
    pyoverdrive.enable()
    make = D._selfcheck_inputs().get(name)
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
    ap.add_argument("--one", help=argparse.SUPPRESS)  # internal: one path, own process
    args = ap.parse_args(argv[1:])

    if args.one:
        print(_measure_one(args.one))
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
    names = sorted(
        p.name
        for lst in GEARBOX._paths.values()
        for p in (lst if isinstance(lst, list) else [lst])
        if p.enabled
    )
    pyoverdrive.disable()

    losses: list[tuple[str, str, float]] = []
    judged = skipped = 0
    for name in names:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), "--one", name],
            capture_output=True, text=True, cwd=str(REPO),
        )
        line = (proc.stdout or "").strip().splitlines()[-1:] or [""]
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
