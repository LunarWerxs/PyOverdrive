"""What did an instrument change do to every cell it measures?

A change to the sweep is a change to what every number in this repo MEANS,
and the only honest way to land one is to run the whole sweep before and
after and diff it cell by cell. Batch 16's size-axis repair is the case this
was written for: `--sizes` grew arrays by tiling, which for any path whose
cost depends on the value distribution measured "the same values, repeated"
under the name of size. Four cells that were red were not path losses at
all - but the same tiling sits behind an unknown number of the `*N` numbers
batches 14 and 15 ACTED on, and "unknown" is not a state to leave that in.

    .venv/Scripts/python tools/compare_sweep_logs.py BEFORE.log AFTER.log
    .venv/Scripts/python tools/compare_sweep_logs.py BEFORE.log AFTER.log --tol 0.2

The output is ordered by what a reader has to act on:

  FIXED     - was below 1.0x, now above. The red was the instrument.
  BROKEN    - was above 1.0x, now below. The green was the instrument, and
              this is a REAL loss that was being hidden. Loudest of all.
  APPEARED  - measured now, skipped before (and the reverse), because a cell
              that changes whether it dispatches has changed what it is.
  MOVED     - same verdict, ratio moved more than --tol.

Exit 1 if anything is BROKEN or newly skipped, so this can gate a change to
the instrument the way the sweep gates a change to a path.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# "   1.46x  unique_sort#int64            numpy.unique"
RATIO = re.compile(r"^\s*([\d.]+)x\s+(\S+)\s+(\S+)\s*$")
# "     skip  unique_sort#int64/3          no-dispatch"
SKIP = re.compile(r"^\s*skip\s+(\S+)\s+(.*?)\s*$")


def parse(text: str) -> tuple[dict[str, float], dict[str, str]]:
    ratios: dict[str, float] = {}
    skips: dict[str, str] = {}
    for line in text.splitlines():
        if line.lstrip().startswith("!!"):
            continue  # the summary block repeats cells already counted
        m = RATIO.match(line)
        if m:
            ratios[m.group(2)] = float(m.group(1))
            continue
        m = SKIP.match(line)
        if m:
            skips[m.group(1)] = m.group(2)
    return ratios, skips


def _classify(
    rb: dict[str, float], sb: dict[str, str],
    ra: dict[str, float], sa: dict[str, str],
    tol: float, min_ratio: float,
) -> tuple[list, list, list, list, list]:
    """Sort every cell into FIXED / BROKEN / MOVED / APPEARED / VANISHED."""
    fixed, broken, moved, appeared, vanished = [], [], [], [], []
    for cell, va in ra.items():
        if cell in rb:
            vb = rb[cell]
            if vb < min_ratio <= va:
                fixed.append((cell, vb, va))
            elif va < min_ratio <= vb:
                broken.append((cell, vb, va))
            else:
                lo, hi = min(va, vb), max(va, vb)
                if lo > 0 and (hi - lo) / lo > tol:
                    moved.append((cell, vb, va, (hi - lo) / lo))
        elif cell in sb:
            appeared.append((cell, sb[cell], va))
    for cell, vb in rb.items():
        if cell not in ra and cell in sa:
            vanished.append((cell, vb, sa[cell]))
    return fixed, broken, moved, appeared, vanished


def _report(
    rb: dict[str, float], sb: dict[str, str],
    ra: dict[str, float], sa: dict[str, str],
    fixed: list, broken: list, moved: list, appeared: list, vanished: list,
    tol: float,
) -> None:
    print(f"before: {len(rb)} measured / {len(sb)} skipped")
    print(f"after:  {len(ra)} measured / {len(sa)} skipped\n")

    if broken:
        print(f"!! {len(broken)} cell(s) BROKEN - passed before, fail now. The "
              f"old green was the instrument, and these are real losses it "
              f"was hiding:")
        for cell, vb, va in sorted(broken, key=lambda r: r[2]):
            print(f"   !! {va:6.2f}x  {cell:34s} (was {vb:.2f}x)")
        print()
    if vanished:
        print(f"!! {len(vanished)} cell(s) stopped dispatching - a cell that "
              f"changes whether it dispatches has changed what it is:")
        for cell, vb, why in vanished:
            print(f"   !! {cell:34s} was {vb:.2f}x, now skipped: {why}")
        print()
    if fixed:
        print(f"{len(fixed)} cell(s) FIXED - failed before, pass now. Those "
              f"reds were the instrument:")
        for cell, vb, va in sorted(fixed, key=lambda r: -r[2]):
            print(f"    + {va:6.2f}x  {cell:34s} (was {vb:.2f}x)")
        print()
    if appeared:
        print(f"{len(appeared)} cell(s) now measured that were skipped before:")
        for cell, why, va in appeared[:20]:
            print(f"    ~ {va:6.2f}x  {cell:34s} (was skipped: {why})")
        print()
    if moved:
        print(f"{len(moved)} cell(s) same verdict, moved more than "
              f"{tol:.0%} (top 25 by size of move):")
        for cell, vb, va, rel in sorted(moved, key=lambda r: -r[3])[:25]:
            print(f"    ~ {vb:7.2f}x -> {va:7.2f}x  {rel:6.0%}  {cell}")
        print()
    if not (fixed or broken or moved or appeared or vanished):
        print("every shared cell reproduces within tolerance.")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--tol", type=float, default=0.20)
    ap.add_argument("--min", type=float, default=1.0)
    args = ap.parse_args(argv[1:])

    rb, sb = parse(args.before.read_text(encoding="utf-8", errors="replace"))
    ra, sa = parse(args.after.read_text(encoding="utf-8", errors="replace"))
    if not rb or not ra:
        print("one of these does not look like a sweep log")
        return 1

    fixed, broken, moved, appeared, vanished = _classify(rb, sb, ra, sa, args.tol, args.min)
    _report(rb, sb, ra, sa, fixed, broken, moved, appeared, vanished, args.tol)
    return 1 if (broken or vanished) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
