"""Is relayout_blocked's floor in the wrong UNIT - elements, not bytes?

The path's three shipped rows sit at 262_144 elements for float64 and
float32 and 1_048_576 for int64. Two of those three thresholds are equal in
ELEMENTS while float32 moves half the data, and the row sweep found exactly
that row losing: float32 measured 0.70x at its own floor, where float64
measured 1.33x and int64 1.34x at theirs.

What the path does is a blocked transpose-copy, and what it beats is
numpy's strided copy. Both move BYTES, and the tiling wins by keeping a
tile resident in cache, so a threshold in elements says different things to
different dtypes - the same defect class as np.histogram2d gating on bin
count while its cost scaled with samples (batch 14).

If one byte floor explains all three rows, the repair is that floor, not
three re-measured element numbers. If it does not, the rows are genuinely
independent and each needs its own measurement. This probe distinguishes
them: square n x n at matched BYTE sizes across the three dtypes.

Method is the sweep's: through the public API, result consumed, the two
sides interleaved, one cell per fresh process, fast core only.

    .venv/Scripts/python tools/probe_relayout_floor.py

VERDICT (2026-08-25, idle Intel box, fp 9bbe7063c555): see
docs/research/batch16-notes.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402

import pyoverdrive  # noqa: E402
from pyoverdrive import _cpuclass as cpuclass  # noqa: E402
from verify_no_pessimization import _measure  # noqa: E402

# Target byte sizes, from below the smallest shipped floor (1 MB) to well
# above it. Each is realised as the nearest square n x n for the dtype, so
# the aspect ratio never varies with the size.
TARGET_MB = (0.5, 1, 2, 3, 4, 6, 8, 12, 16, 32, 64)
DTYPES = ("float32", "float64", "int64")


def _side(mb: float, dtype: str) -> int:
    itemsize = np.dtype(dtype).itemsize
    return max(2, int((mb * 1024 * 1024 / itemsize) ** 0.5))


def _one(spec: str, fast_under: float | None) -> str:
    if fast_under is not None and cpuclass.probe_us() > fast_under:
        return "SLOW-CORE"
    # the SIDE is passed, never a byte count the child re-derives: at the
    # boundary this probe exists to test, n and n+1 differ by one element and
    # a float round-trip through MB could land on the wrong side of it
    side, dtype = spec.split(",")
    n = int(side)
    rng = np.random.default_rng(7171)
    a = rng.standard_normal((n, n))
    a = (a * 1000).astype(dtype) if np.dtype(dtype).kind in "iu" else a.astype(dtype)
    a = a.T  # F-contiguous view, which is what the path is for
    pyoverdrive.enable()
    from pyoverdrive.dispatcher.gearbox import GEARBOX

    chosen, reason = GEARBOX.decide("numpy.ascontiguousarray", (a,), {})
    if chosen != "relayout_blocked":
        return f"NODISPATCH {reason}"
    ratio = _measure("numpy.ascontiguousarray", (a,), {}, rounds=9)
    return "UNMEASURABLE" if ratio is None else f"RATIO {ratio:.4f}"


# The matched-byte grid put every >= 4M-element cell at 3.02-5.16x and every
# smaller one at 0.89-2.69x, which is not a byte boundary at all - it is
# where threads_for() switches from 8 threads to 16. These sides straddle
# that switch by one element in each direction, at deliberately
# non-power-of-two n so the aliasing spike cannot be what is being seen.
BOUNDARY_SIDES = (1447, 1449, 2047, 2049, 2447, 2449)


def _sides_mode(cutoff, sides) -> None:
    # 4M elements was the 8-vs-16 thread switch when this probe was written.
    # The A/B below refuted it as the mechanism, and the floor then moved
    # onto that same number, which collapsed the schedule to a constant 16 -
    # so the boundary is hardcoded here rather than read from the module,
    # and the columns still straddle exactly what was tested.
    switch = 4 * 1024 * 1024
    print(f"\nthe refuted 8 -> 16 thread switch sat at {switch:,d} elements; "
          f"relayout now runs a constant 16\n")
    print(f"{'side':>6s} {'elements':>12s} {'thr':>4s} " +
          " ".join(f"{d:>9s}" for d in DTYPES))
    for n in sides:
        row = []
        for dtype in DTYPES:
            cmd = [sys.executable, str(Path(__file__)), "--one", f"{n},{dtype}"]
            if cutoff is not None:
                cmd += [f"{cutoff:.3f}"]
            for _ in range(6):
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      cwd=str(REPO))
                line = ((proc.stdout or "").strip().splitlines()[-1:] or [""])[0]
                if line.strip() != "SLOW-CORE":
                    break
            if line.startswith("RATIO"):
                row.append(f"{float(line.split()[1]):>8.2f}x")
            else:
                row.append(f"{line.split()[0][:9]:>9s}")
        thr = 16 if n * n >= switch else 8
        print(f"{n:>6d} {n * n:>12,d} {thr:>4d} " + " ".join(row))


def main(argv: list[str]) -> int:
    if len(argv) > 2 and argv[1] == "--one":
        under = float(argv[3]) if len(argv) > 3 else None
        print(_one(argv[2], under))
        return 0

    classes = cpuclass.classify()
    print(cpuclass.describe(classes))
    cutoff = cpuclass.fast_cutoff(classes)

    if "--sides" in argv:
        i = argv.index("--sides")
        explicit = argv[i + 1] if len(argv) > i + 1 else ""
        sides = (tuple(int(s) for s in explicit.split(","))
                 if explicit and not explicit.startswith("-") else BOUNDARY_SIDES)
        _sides_mode(cutoff, sides)
        return 0

    from pyoverdrive.fastpaths import relayout_blocked

    print("\nshipped element floors:", {np.dtype(d).name: v for d, v
                                        in relayout_blocked.SUPPORTED.items()})
    print(f"\n{'MB':>6s} " + " ".join(f"{d:>22s}" for d in DTYPES))
    for mb in TARGET_MB:
        row = []
        for dtype in DTYPES:
            n = _side(mb, dtype)
            cmd = [sys.executable, str(Path(__file__)), "--one", f"{n},{dtype}"]
            if cutoff is not None:
                cmd += [f"{cutoff:.3f}"]
            for _ in range(6):
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      cwd=str(REPO))
                line = ((proc.stdout or "").strip().splitlines()[-1:] or [""])[0]
                if line.strip() != "SLOW-CORE":
                    break
            tag = f"{n}^2={n * n / 1e6:.2f}M"
            if line.startswith("RATIO"):
                row.append(f"{tag:>13s} {float(line.split()[1]):>7.2f}x")
            elif line.startswith("NODISPATCH"):
                row.append(f"{tag:>13s} {'refused':>8s}")
            else:
                row.append(f"{tag:>13s} {line.split()[0][:8]:>8s}")
        print(f"{mb:>6g} " + " ".join(row))
    print("\n'refused' = below that dtype's shipped element floor. Each column "
          "is the SAME number of bytes, so a byte-shaped cost model would put "
          "the crossing on one row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
