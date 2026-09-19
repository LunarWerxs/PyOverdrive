"""Which cells of two probe runs AGREE, and which do not reproduce?

A number that does not reproduce is not a measurement. This project already
applies that to a red - `verify_no_pessimization.py` re-runs a losing cell in
a second process and drops it if the second reading disagrees - and the same
standard has to apply to the numbers a THRESHOLD is derived from, because a
floor set from an unstable cell is wrong in whichever direction the noise
went.

Eyeballing two nine-table grids for disagreement is exactly the job that
invents a pattern where there is none. This does it arithmetically:

    .venv/Scripts/python tools/compare_probe_tables.py RUN_A.log RUN_B.log
    .venv/Scripts/python tools/compare_probe_tables.py A.log B.log --tol 0.15

A cell is UNSTABLE when the two runs differ by more than --tol relative to
the smaller reading, and it is called out loudest when the two runs
disagree about the SIGN of the verdict - one above 1.0x and the other
below - because that is a cell that would flip a floor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from propose_cardinality_floors import parse  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--tol", type=float, default=0.15,
                    help="relative difference above which a cell is unstable")
    args = ap.parse_args(argv[1:])

    ta = parse(args.a.read_text(encoding="utf-8", errors="replace"))
    tb = parse(args.b.read_text(encoding="utf-8", errors="replace"))
    shared = [k for k in ta if k in tb]
    if not shared:
        print("no tables in common")
        return 1

    flips, unstable, stable = [], [], 0
    for key in shared:
        cards_a, rows_a = ta[key]
        _, rows_b = tb[key]
        rb = {n: cells for n, cells in rows_b}
        for n, cells in rows_a:
            if n not in rb:
                continue
            for i, va in enumerate(cells):
                vb = rb[n][i] if i < len(rb[n]) else None
                if va is None or vb is None:
                    continue
                card = cards_a[i] if i < len(cards_a) else "?"
                lo, hi = min(va, vb), max(va, vb)
                rel = (hi - lo) / lo if lo > 0 else float("inf")
                if (va < 1.0) != (vb < 1.0):
                    flips.append((key, n, card, va, vb))
                elif rel > args.tol:
                    unstable.append((key, n, card, va, vb, rel))
                else:
                    stable += 1

    print(f"{stable} cell(s) reproduce within {args.tol:.0%}\n")
    if flips:
        print(f"{len(flips)} cell(s) DISAGREE ABOUT THE VERDICT - one run "
              f"says win, the other says loss. A floor must not rest on "
              f"these:")
        for (op, dt), n, card, va, vb in flips:
            print(f"  !! {op:>14s} {dt:>6s}  n={n:>9,d}  card={card:>8s}  "
                  f"{va:.2f}x vs {vb:.2f}x")
        print()
    if unstable:
        print(f"{len(unstable)} cell(s) agree on the verdict but move more "
              f"than {args.tol:.0%}:")
        for (op, dt), n, card, va, vb, rel in sorted(
                unstable, key=lambda r: -r[5])[:20]:
            print(f"   ~ {op:>14s} {dt:>6s}  n={n:>9,d}  card={card:>8s}  "
                  f"{va:.2f}x vs {vb:.2f}x  ({rel:.0%})")
    if not flips and not unstable:
        print("every shared cell reproduces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
