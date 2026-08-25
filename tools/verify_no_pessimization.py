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

--sizes closes the other half: every cell is also judged at 3x, 10x, 30x
and 100x its canonical size AND at a third, a tenth, a thirtieth and a
hundredth of it. Both directions were needed and both found real losses.
Upward, three det/slogdet cells were losing at the TOP of their window
(1.01x, 0.84x, 0.85x) while passing at their canonical input. Downward,
np.inner had no size gate at all and ran at 0.38x on operands smaller than
its canonical one - which was, it turned out, the smallest shape in the
sweep that wins.

That is the pattern to distrust: a hand-picked canonical input is evidence
about whoever picked it, and they picked one where the path works. A path
with a real floor simply reports NODISPATCH as the cells shrink, which
costs nothing; a path without one keeps accepting, and that is the case
worth finding.

--shapes closes the aspect-ratio class that scaling can never reach. At
roughly constant volume, each cell with a 2-D-or-deeper input is re-judged
with its trailing axis grown 4x and 16x while the leading axis shrinks by
the same factor (long rows / long contraction, few of them - exactly the
regime where np.inner ran at 0.38x), and the reverse. Operands that share
their trailing length move together; matmul-shaped pairs get a chain
variant that grows the shared inner dimension instead, so (m,k)x(k,n)
stays a valid product. A path whose predicate refuses the reshaped input
reports NODISPATCH, which costs nothing and is the correct answer for a
path with a real shape gate.

What is STILL not covered even so: regimes that live on a KEYWORD axis
rather than a shape (histogram bin counts - that one bit already and now
has a sample floor), per-dimension cells for the small-matrix linalg
families (the selfcheck input picks one d; other d's have their own
calibration rows but only one is swept), and layout - every reshape here
lands C-contiguous, so an F-contiguity-gated path skips its shape cells.

Usage:
    .venv/Scripts/python tools/verify_no_pessimization.py [--min 1.0] [-v]

Exit 0 = every dispatching path is at least --min. Exit 1 = at least one
dispatches into a loss.
"""

from __future__ import annotations

import argparse
import re
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


def cells(mults: tuple = (), divs: tuple = (), shapes: tuple = ()) -> list[str]:
    """Every cell to judge, as "path", "path@dtype", optionally "*mult",
    "/div", ">f" (trailing axis f-fold longer, leading f-fold shorter) or
    "<f" (the reverse).

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
    if not (mults or divs or shapes):
        return out
    scaled = []
    for c in out:
        for m in mults:
            scaled.append(c if m == 1 else f"{c}*{m}")
        for d in divs:
            scaled.append(f"{c}/{d}")
        for f in shapes:
            scaled.append(f"{c}>{f}")
            scaled.append(f"{c}<{f}")
    return scaled


# Scaling factors for the deep sweep. Each path's canonical input sits near
# the BOTTOM of what it accepts, so the unsampled axis is upward - and that
# is where the losses were: det 3x3 measured 1.01x at 1e5 and slogdet 3x3
# 0.84x, both far above their canonical cell, both passing the shallow sweep.
SIZE_MULTS = (1, 3, 10, 30, 100)

# ...and DOWNWARD, which is the half that actually bit. np.inner had no size
# gate at all and ran at 0.38x on operands smaller than its canonical input,
# for months. A path with a floor simply stops dispatching as these shrink,
# which reports as NODISPATCH and costs nothing; a path WITHOUT one keeps
# accepting, and that is exactly what needs finding.
SIZE_DIVS = (3, 10, 30, 100)

# Aspect-ratio factors for --shapes. Scaling moves volume; these move SHAPE
# at (roughly) constant volume, which is the axis scaling can never sample.
# np.inner's 0.38x corner was exactly this class - a long contraction
# against few rows has the same element count as its canonical input, so no
# multiple or fraction of that input would ever have produced it.
ASPECT_FACTORS = (4, 16)

# A scaled cell that would allocate more than this is skipped rather than
# risking an OOM on someone's machine; the skip is printed, not silent.
MAX_ELEMENTS = 120_000_000


def _scaled(args: tuple, mult: int, div: int = 1) -> tuple | None:
    """Every array argument on the SIZE axis, grown by `mult`.

    Which axis is the size axis is not knowable in general, so the rule is:
    grow the leading axis of every ndarray whose leading axis is already the
    longest among the arguments. That keeps paired operands in step
    (solve's a and b, a binary ufunc's two inputs) while leaving small
    parameter arrays alone - a 128-element quantile vector beside a
    1e6-element sample must not be "scaled" into something else entirely.
    """
    if mult == 1 and div == 1:
        return args
    arrays = [a for a in args if isinstance(a, np.ndarray) and a.ndim >= 1]
    if not arrays:
        return None
    lead = max(a.shape[0] for a in arrays)
    total = 0
    out = []
    for a in args:
        if isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[0] == lead:
            if div > 1:
                grown = np.ascontiguousarray(a[: max(1, a.shape[0] // div)])
            else:
                grown = np.concatenate([a] * mult, axis=0)
            total += grown.size
            out.append(grown)
        else:
            out.append(a)
    return None if total > MAX_ELEMENTS else tuple(out)


def _reshaped(args: tuple, f: int, trail_heavy: bool) -> list[tuple]:
    """Candidate reshapes of `args` at ~constant volume, aspect moved by `f`.

    trail_heavy grows every trailing axis f-fold and shrinks the leading
    axis f-fold (long rows / long contraction, few of them); the reverse
    direction does the opposite. Two rules, tried in turn, because operand
    coupling is not knowable in general:

    - shared-trailing: every array whose last axis matches the longest last
      axis moves together. That keeps np.inner's stacked operands and a
      matrix-vector product valid, and leaves small parameter arrays (a
      quantile vector beside its sample) alone.
    - chain: two 2-D operands with a.shape[-1] == b.shape[0] are a product
      chain, so the SHARED inner dimension is what grows while the outer
      dimensions shrink - the shared-trailing rule would break the chain.

    A rule that produces an input the op cannot run, or that the path's
    predicate refuses, costs a printed skip and nothing else.
    """
    arrays = [a for a in args if isinstance(a, np.ndarray) and a.ndim >= 1]
    if not any(a.ndim >= 2 for a in arrays):
        return []

    def grow(a, axis):
        return np.concatenate([a] * f, axis=axis)

    def cut(a, axis):
        idx = [slice(None)] * a.ndim
        idx[axis] = slice(0, max(1, a.shape[axis] // f))
        return np.ascontiguousarray(a[tuple(idx)])

    variants = []
    t = max(a.shape[-1] for a in arrays)
    out = []
    for a in args:
        if isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[-1] == t:
            if trail_heavy:
                a = grow(a, -1)
                if a.ndim >= 2:
                    a = cut(a, 0)
            else:
                a = cut(a, -1)
                if a.ndim >= 2:
                    a = grow(a, 0)
        out.append(a)
    variants.append(tuple(out))

    two = [a for a in args if isinstance(a, np.ndarray) and a.ndim == 2]
    if len(two) == 2 and two[0].shape[-1] == two[1].shape[0]:
        a, b = two
        if trail_heavy:
            na, nb = cut(grow(a, -1), 0), cut(grow(b, 0), -1)
        else:
            na, nb = grow(cut(a, -1), 0), grow(cut(b, 0), -1)
        repl = {id(a): na, id(b): nb}
        variants.append(tuple(repl.get(id(x), x) for x in args))

    return [v for v in variants
            if sum(x.size for x in v if isinstance(x, np.ndarray))
            <= MAX_ELEMENTS]


def _inputs_for(cell: str):
    """(path name, maker) for a cell, expanding a "path@dtype" cell onto the
    row's own floor rather than whatever the selfcheck input would pick."""
    cell = re.split(r"[*/><]", cell, maxsplit=1)[0]
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

    variants = [call_args]
    marker = re.search(r"([*/><])(\d+)$", cell)
    if marker:
        kind, val = marker.group(1), int(marker.group(2))
        if kind in "*/":
            scaled = _scaled(call_args,
                             val if kind == "*" else 1,
                             val if kind == "/" else 1)
            if scaled is None:
                return "SKIP too-big"
            variants = [scaled]
        else:
            variants = _reshaped(call_args, val, kind == ">")
            if not variants:
                return "SKIP not-shaped"

    paths = [
        p
        for lst in GEARBOX._paths.values()
        for p in (lst if isinstance(lst, list) else [lst])
        if p.name == name
    ]
    if not paths:
        return "SKIP unknown"
    op = paths[0].op

    dispatched = False
    for cargs in variants:
        try:
            if GEARBOX.decide(op, cargs, call_kwargs)[0] != name:
                continue
        except Exception:
            continue
        dispatched = True
        ratio = _measure(op, cargs, call_kwargs, rounds=9)
        if ratio is None:
            continue
        return f"RATIO {ratio:.4f} {op}"
    return "SKIP unmeasurable" if dispatched else "SKIP no-dispatch"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=1.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--any-core", action="store_true",
                    help="accept measurements taken on a slow core too; on a "
                         "hybrid CPU that can hide a threaded pessimization")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--sizes", action="store_true",
                    help="also judge each cell at 3x, 10x, 30x and 100x its "
                         "canonical size. Every canonical input sits near the "
                         "BOTTOM of what its path accepts, so upward is the "
                         "axis nothing was sampling - and three shipped losses "
                         "were hiding up there.")
    ap.add_argument("--shapes", action="store_true",
                    help="also judge each 2-D-or-deeper cell with its aspect "
                         "ratio moved 4x and 16x in both directions at ~constant "
                         "volume. Scaling moves volume; this moves SHAPE, which "
                         "is the axis np.inner's 0.38x corner lived on.")
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
    names = [c for c in cells(SIZE_MULTS if args.sizes else (),
                              SIZE_DIVS if args.sizes else (),
                              ASPECT_FACTORS if args.shapes else ())
             if re.split(r"[@*/><]", c)[0] in live]

    # A shape cell needs a 2-D-or-deeper input to have an aspect ratio at
    # all; filtering the 1-D cells here saves a subprocess each, and the
    # count printed below stays a count of cells that could ever judge.
    shaped_ok: dict[str, bool] = {}

    def _has_matrix(cell: str) -> bool:
        base = re.split(r"[*/><]", cell, maxsplit=1)[0]
        if base not in shaped_ok:
            try:
                a, _ = _inputs_for(base)[1]()
                shaped_ok[base] = any(
                    isinstance(x, np.ndarray) and x.ndim >= 2 for x in a)
            except Exception:
                shaped_ok[base] = False
        return shaped_ok[base]

    names = [c for c in names if (">" not in c and "<" not in c)
             or _has_matrix(c)]
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
            # MAKE THE RED REPRODUCE. A sweep this wide will throw the odd
            # unlucky cell, and a detector that cries wolf gets ignored -
            # which is how the real one then gets missed. A loss is only
            # reported when a second, independent process agrees; the worse
            # of the two is what gets printed.
            again = subprocess.run(cmd, capture_output=True, text=True,
                                   cwd=str(REPO))
            reparts = ((again.stdout or "").strip().splitlines()[-1:] or [""])[0].split()
            if reparts and reparts[0] == "RATIO" and float(reparts[1]) < args.min:
                losses.append((name, op, min(ratio, float(reparts[1]))))
            elif args.verbose:
                print(f"  {ratio:6.2f}x  {name:28s} {op}  (did not reproduce; "
                      f"second reading "
                      f"{reparts[1] if len(reparts) > 1 else '?'})")
                continue
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
