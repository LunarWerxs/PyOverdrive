"""Where does the isin hash route stop paying as the two operands' RATIO moves?

Both isin hash paths gate on ONE number - `element.size + test.size >= 300` -
and that number cannot see the axis the four-axis sweep just found them
losing on. At constant combined size, moved 4x and 16x toward MANY elements
against FEW test_elements, they measured:

    isin_string_hash>4   0.35x      isin_string_hash>16   0.05x
    isin_object_hash>4   0.46x      isin_object_hash>16   0.09x

0.05x is the worst reading in the package, and it is not a worse kernel - it
is the wrong algorithm for that corner. numpy's own `in1d` switches methods
on exactly this ratio: with a small enough `test_elements` it evaluates a
CHAIN OF VECTORIZED EQUALITIES (`ar1 == ar2[0]` | `ar1 == ar2[1]` | ...),
which runs the comparison in C across the whole array. The hash route
instead pays one Python-level hash and lookup PER ELEMENT of the big
operand, because these are object and StringDType arrays. The fewer distinct
things there are to test against, the better stock's method gets and the
worse ours does, and nothing in the shipped predicate notices.

This walks the ratio to find where the crossing actually is, in both
directions, so a gate can be written from the measurement rather than from
numpy's heuristic - which is a different quantity (it was tuned for
fixed-width numeric dtypes, where its per-element cost is nothing like the
Python-object cost that decides these two paths).

Method is the sweep's: through the public API with the result consumed, the
two sides interleaved, one cell per fresh process, and on a hybrid CPU only
on a fast core.

    .venv/Scripts/python tools/probe_isin_ratio.py
    .venv/Scripts/python tools/probe_isin_ratio.py --kind object

RUN IT ON THE QUIET BOX. These are timings.
"""

from __future__ import annotations

import argparse
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

# WALK test.size ITSELF, not a share of the total. The first cut of this
# probe moved a FRACTION, which conflates the two things that decide the
# answer and put its slowest cells where the answer was never in doubt: at
# combined 100,000 a 0.1% share is 100 test elements, far above any
# crossing, and stock's method there costs 100 vectorized passes over
# 99,900 objects, so the cell took minutes to confirm a win nobody
# questioned. Three completed rows had already located the crossing
# between 6 and 15 test elements at every size measured, which is the
# shape of a FLOOR ON test.size and not of a ratio at all:
#
#   combined    n_test=1   2      5      6      10     15     20
#   300         0.25x      0.37x  -      0.88x  -      1.91x  -
#   1,000       0.17x      -      0.73x  -      -      -      2.56x
#   10,000      -          -      -      -      1.53x  -      -
#
# So: the same small counts against every element count, which is exactly
# the grid that separates "few things to test against" from "small input".
TEST_SIZES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
ELEMENTS = (300, 1_000, 10_000, 100_000)

KINDS = ("object", "string")


def _words(n: int, rng) -> list[str]:
    """Distinct short strings, the shape a real membership test carries."""
    return [f"w{int(x):09d}" for x in rng.integers(0, 10 ** 9, size=n)]


def _build(kind: str, n_element: int, n_test: int):
    rng = np.random.default_rng(20260825)
    # Overlap matters: an all-miss test is a different call from a
    # half-hit one, and the hash route's cost is the same either way while
    # stock's is not. Half the elements are drawn FROM the test set.
    test_words = _words(max(1, n_test), rng)
    hits = rng.choice(test_words, size=n_element // 2) if n_test else []
    misses = _words(n_element - len(hits), rng)
    element_words = list(hits) + misses
    rng.shuffle(element_words)
    if kind == "object":
        element = np.array(element_words, dtype=object)
        test = np.array(test_words, dtype=object)
    else:
        element = np.array(element_words, dtype=np.dtypes.StringDType())
        test = np.array(test_words, dtype=np.dtypes.StringDType())
    return element, test


def _one(spec: str, fast_under: float | None) -> str:
    if fast_under is not None and cpuclass.probe_us() > fast_under:
        return "SLOW-CORE"
    kind, n_element, n_test = spec.split(",")
    element, test = _build(kind, int(n_element), int(n_test))
    pyoverdrive.enable()
    from pyoverdrive.dispatcher.gearbox import GEARBOX

    chosen, reason = GEARBOX.decide("numpy.isin", (element, test), {})
    if chosen is None or not chosen.startswith("isin_"):
        return f"NODISPATCH {reason}"
    ratio = _measure("numpy.isin", (element, test), {}, rounds=9)
    return "UNMEASURABLE" if ratio is None else f"RATIO {ratio:.4f} {chosen}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--fast-under", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--kind", choices=KINDS, action="append", default=[])
    ap.add_argument("--tests", help="explicit test_elements sizes, comma "
                                    "separated, instead of the default walk. "
                                    "The string path's crossing MOVES with the "
                                    "element count while the object path's does "
                                    "not, so its gate needs the columns above "
                                    "32 that the default walk stops at.")
    ap.add_argument("--elements", help="explicit element counts, comma separated")
    ap.add_argument("--retries", type=int, default=6)
    args = ap.parse_args(argv[1:])

    if args.one:
        print(_one(args.one, args.fast_under))
        return 0

    classes = cpuclass.classify()
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}")
    print(cpuclass.describe(classes))
    cutoff = cpuclass.fast_cutoff(classes)

    kinds = args.kind or list(KINDS)
    test_sizes = ([int(x) for x in args.tests.split(",") if x.strip()]
                  if args.tests else list(TEST_SIZES))
    elements = ([int(x) for x in args.elements.split(",") if x.strip()]
                if args.elements else list(ELEMENTS))
    losses = 0
    for kind in kinds:
        print(f"\n=== isin, {kind} operands "
              f"{'=' * 30}\n")
        print(f"{'elements':>10s} " +
              " ".join(f"{n:>8,d}" for n in test_sizes) +
              "   (test_elements, absolute)")
        for n_element in elements:
            row = []
            for n_test in test_sizes:
                cmd = [sys.executable, str(Path(__file__)), "--one",
                       f"{kind},{n_element},{n_test}"]
                if cutoff is not None:
                    cmd += ["--fast-under", f"{cutoff:.3f}"]
                for _ in range(max(1, args.retries)):
                    proc = subprocess.run(cmd, capture_output=True, text=True,
                                          cwd=str(REPO))
                    line = ((proc.stdout or "").strip().splitlines()[-1:]
                            or [""])[0]
                    if line.strip() != "SLOW-CORE":
                        break
                if line.startswith("RATIO"):
                    ratio = float(line.split()[1])
                    losses += ratio < 1.0
                    row.append(f"{ratio:>7.2f}x")
                elif line.startswith("NODISPATCH"):
                    row.append(f"{'-':>8s}")
                else:
                    row.append(f"{line.split()[0][:8]:>8s}")
            print(f"{n_element:>10,d} " + " ".join(row))

    print("\n'-' = the shipped predicate refuses that cell. Every column left "
          "of the crossing is a corner the shipped gate currently accepts and "
          "should not.")
    print(f"cells below 1.0x: {losses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
