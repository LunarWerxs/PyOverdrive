"""Turn a probe_cardinality table into the floors it implies. No judgement.

The rule is not this file's invention - `tools/calibrate_dispatch.py` states
it for its own variants: "all four are things a caller can hit, so the
threshold has to hold for the worst". Cardinality is exactly such a variant.
A caller passing `np.unique` an array of 16 distinct values is not doing
anything exotic, and nothing in the gate can see the difference cheaply at
these sizes, so **the ratio that decides a floor is the MINIMUM across the
cardinality row**, not the one the calibration battery happened to draw.

Applying that by eye across nine tables is how an arithmetic slip becomes a
shipped threshold, so it is applied here instead:

    .venv/Scripts/python tools/propose_cardinality_floors.py TABLE.log
    .venv/Scripts/python tools/propose_cardinality_floors.py TABLE.log --target 1.0

--target is the bar a floor must clear. 1.30 is the project's min-win for a
NEW threshold (calibrate_dispatch.py's default); 1.0 is the weaker "never
slower than stock" bar that tools/verify_no_pessimization.py enforces. Both
are printed because they are different questions: 1.0 says the shipped
promise is not violated, 1.30 says the row is worth having at all.

A row where NO measured size clears the bar is reported as such, loudly:
that is a row a floor cannot rescue, and it needs withdrawal or a different
mechanism rather than a bigger number.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEAD = re.compile(r"^=== (\w+), (\w+) =+\s*$")
COLS = re.compile(r"^\s*n\s+(.*?)\s+\(distinct values\)\s*$")
ROW = re.compile(r"^\s*([\d,]+)\s+(.*?)\s*$")
CELL = re.compile(r"^(?:([\d.]+)x|(-|n/a|\S+))$")


def parse(text: str) -> dict[tuple[str, str], tuple[list[str], list[tuple[int, list]]]]:
    tables: dict[tuple[str, str], tuple[list[str], list[tuple[int, list]]]] = {}
    key = None
    cards: list[str] = []
    for line in text.splitlines():
        m = HEAD.match(line)
        if m:
            key = (m.group(1), m.group(2))
            tables[key] = ([], [])
            continue
        if key is None:
            continue
        m = COLS.match(line)
        if m:
            cards = m.group(1).split()
            tables[key] = (cards, [])
            continue
        m = ROW.match(line)
        if m and cards and line.strip() and not line.startswith("==="):
            try:
                n = int(m.group(1).replace(",", ""))
            except ValueError:
                continue
            cells: list[float | None] = []
            for tok in m.group(2).split():
                c = CELL.match(tok)
                if c and c.group(1):
                    cells.append(float(c.group(1)))
                else:
                    cells.append(None)
            if len(cells) == len(cards):
                tables[key][1].append((n, cells))
    return tables


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("table", type=Path)
    ap.add_argument("--target", type=float, default=1.30)
    args = ap.parse_args(argv[1:])

    tables = parse(args.table.read_text(encoding="utf-8", errors="replace"))
    if not tables:
        print("no tables parsed - is this a probe_cardinality log?")
        return 1

    print(f"floor = the smallest measured n whose WORST cardinality clears "
          f"the bar,\nand every larger measured n clears it too (a floor that "
          f"stops holding\nfurther up is not a floor).\n")
    unrescuable = 0
    for (op, dtype), (cards, rows) in tables.items():
        if not rows:
            continue
        print(f"=== {op}, {dtype}")
        for bar in (1.0, args.target):
            floor = None
            for i, (n, cells) in enumerate(rows):
                measured = [c for c in cells if c is not None]
                if not measured:
                    continue
                if min(measured) < bar:
                    floor = None
                    continue
                # every LARGER measured size must hold too
                if all(min([c for c in rows[j][1] if c is not None] or [0]) >= bar
                       for j in range(i, len(rows))):
                    floor = n
                    break
            worst_at = {n: min([c for c in cells if c is not None] or [float("nan")])
                        for n, cells in rows}
            if floor is None:
                unrescuable += (bar == args.target)
                span = ", ".join(f"{n:,}:{w:.2f}x" for n, w in worst_at.items())
                print(f"  bar {bar:.2f}x  NO FLOOR RESCUES THIS ROW - worst "
                      f"per size: {span}")
            else:
                print(f"  bar {bar:.2f}x  floor = {floor:,} "
                      f"(worst cardinality there: {worst_at[floor]:.2f}x)")
        print()

    if unrescuable:
        print(f"{unrescuable} row(s) clear no bar at any measured size. A "
              f"bigger number does not fix those - they need withdrawal, a "
              f"different algorithm, or a gate on something observable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
