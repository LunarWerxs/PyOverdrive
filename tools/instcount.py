"""User-space instruction counts for one benchmark cell (Linux only).

WHY: wall-clock ratios on a shared box move with foreign load; for a
SINGLE-THREADED cell the number of user-space instructions retired is
exact for a fixed amount of work, so tools/ratchet.py judges those cells
on it. Windows has no such counter here: the field is None (NOT MEASURED),
never 0, and the caller falls back to the wall ratio.

Counter, in order: `perf stat -e instructions:u` when perf may open the
event (checked by running it on `true`; containers often forbid it), else
`valgrind --tool=callgrind` (Ir total). Neither: not measured.

A cell is measured as a fixed number of public calls on one side (stock or
patched) in a fresh process with every thread pool pinned to 1. The fixed
startup cost (imports, input construction) is removed by differencing a
run of N calls against a run of 0 calls, so the result is instructions per
N calls of that side only.

Usage (Linux):
    python tools/instcount.py --cell <cell> [--calls 20] [--json]
Child (internal): python tools/instcount.py --child <cell> --side stock|patched --calls N
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

_PIN = {"PYOVERDRIVE_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1"}


def counter() -> str | None:
    """"perf", "callgrind", or None when this host cannot count (Windows included)."""
    if not sys.platform.startswith("linux"):
        return None
    if shutil.which("perf"):
        probe = subprocess.run(["perf", "stat", "-x,", "-e", "instructions:u", "true"],
                               capture_output=True, text=True)
        if probe.returncode == 0 and _perf_value(probe.stderr + probe.stdout) is not None:
            return "perf"
    if shutil.which("valgrind"):
        return "callgrind"
    return None


def _perf_value(text: str) -> int | None:
    for line in text.splitlines():
        fields = line.split(",")
        if len(fields) > 2 and "instructions" in fields[2] and fields[0].strip().isdigit():
            return int(fields[0])
    return None


def count_command(cmd: list[str], tool: str, env: dict | None = None) -> int | None:
    """Instructions retired in user space by `cmd`, or None when not measurable."""
    env = {**os.environ, **(env or {})}
    if tool == "perf":
        proc = subprocess.run(["perf", "stat", "-x,", "-e", "instructions:u", *cmd],
                              capture_output=True, text=True, env=env)
        return _perf_value(proc.stderr) if proc.returncode == 0 else None
    if tool == "callgrind":
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "callgrind.out"
            proc = subprocess.run(["valgrind", "--tool=callgrind", f"--callgrind-out-file={out}",
                                   *cmd], capture_output=True, text=True, env=env)
            hit = re.search(r"Collected\s*:\s*(\d+)", proc.stderr)
            return int(hit.group(1)) if proc.returncode == 0 and hit else None
    return None


def is_single_threaded(path_name: str) -> bool:
    """The parallel_* fast paths are the multi-threaded cells; everything else is one thread."""
    return not path_name.startswith("parallel_")


def measure_cell(cell: str, calls: int = 20) -> dict:
    """{"tool", "stock", "patched", "calls"} with None counts when not measured."""
    tool = counter()
    result = {"tool": tool, "calls": calls, "stock": None, "patched": None}
    if tool is None:
        return result
    base = [sys.executable, str(Path(__file__).resolve()), "--child", cell, "--calls"]
    for side in ("stock", "patched"):
        loaded = count_command([*base, str(calls), "--side", side], tool, _PIN)
        idle = count_command([*base, "0", "--side", side], tool, _PIN)
        if loaded is not None and idle is not None and loaded > idle:
            result[side] = loaded - idle
    return result


def _child(cell: str, side: str, calls: int) -> int:
    import verify_no_pessimization as V

    V.pyoverdrive.enable()
    name, make = V._inputs_for(cell)
    if make is None:
        print("SKIP no-input")
        return 2
    call_args, call_kwargs = make()
    variants, skip = V._cell_variants(cell, call_args, make)
    if skip is not None:
        print(skip)
        return 2
    path = next(p for lst in V.GEARBOX._paths.values()
                for p in (lst if isinstance(lst, list) else [lst]) if p.name == name)
    op = path.op
    for cargs in variants:
        try:
            chosen, _ = V.GEARBOX.decide(op, cargs, call_kwargs)
        except Exception:  # noqa: BLE001
            continue
        if chosen != name:
            continue
        holder, attr = V._resolve(op)
        fn = V.GEARBOX.stock_fn(op) if side == "stock" else getattr(holder, attr)
        fn(*cargs, **call_kwargs)  # warm: first-call caches belong to startup
        for _ in range(calls):
            V._consume(fn(*cargs, **call_kwargs))
        return 0
    print("SKIP no-dispatch")
    return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/instcount.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--child")
    ap.add_argument("--side", choices=("stock", "patched"), default="patched")
    args = ap.parse_args(argv)
    if args.child:
        return _child(args.child, args.side, args.calls)
    if not args.cell:
        ap.error("--cell is required")
    result = measure_cell(args.cell, args.calls)
    if args.json:
        print(json.dumps(result))
    elif result["tool"] is None:
        print(f"{args.cell}: instructions NOT MEASURED (no perf/valgrind on this host)")
    else:
        print(f"{args.cell}: stock={result['stock']} patched={result['patched']} "
              f"instructions per {args.calls} calls ({result['tool']})")
    return 0 if result["stock"] is not None and result["patched"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
