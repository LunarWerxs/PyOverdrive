"""Derive the parallel_ufunc dispatch table from PYRALLEL-CAL evidence.

Reads benchmarks/results/PYRALLEL-CAL/<fingerprint>.json (default: this
machine's fingerprint) and prints, for each (op, dtype), the smallest
measured size from which the byte-scheduled thread count wins by at least
MIN_WIN on that size AND every larger measured size. Pairs with no such size
are reported as "no win: stays on stock".

The output is a proposal to paste into
src/pyoverdrive/fastpaths/parallel_ufunc.py by hand, with the numbers in
front of you. It never edits code. Rule of the house: hardware decides,
agents do not extrapolate.

Usage:
    python lab/cli/calibrate_pyrallel.py [--fingerprint 8f8198d9abab] [--min-win 1.3]
    python lab/cli/calibrate_pyrallel.py --suite PYRALLEL-BIN-CAL   # the binary family
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lab.dyno.fingerprint import machine_fingerprint  # noqa: E402

RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"

# Byte-based schedule: (minimum nbytes, threads). Keyed on bytes, not
# elements, so a float32 array (kernels ~2x faster per element on the
# calibrated machine) needs ~2x the elements to earn the same thread count.
BYTE_SCHEDULE = (
    (24 * 1024 * 1024, 16),
    (8 * 1024 * 1024, 8),
    (768 * 1024, 4),
)

ITEMSIZE = {"float64": 8, "float32": 4, "int64": 8, "int32": 4}


def threads_for_bytes(nbytes: int) -> int:
    for min_bytes, t in BYTE_SCHEDULE:
        if nbytes >= min_bytes:
            return t
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint", default=machine_fingerprint()["fingerprint"])
    ap.add_argument("--min-win", type=float, default=1.3)
    ap.add_argument("--suite", default="PYRALLEL-CAL", help="PYRALLEL-CAL (unary) or PYRALLEL-BIN-CAL (binary)")
    args = ap.parse_args()

    path = RESULTS_ROOT / args.suite / f"{args.fingerprint}.json"
    if not path.exists():
        print(f"no evidence at {path}; run the matching benchmarks/micro/bench_pyrallel*_calibration.py")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    cond = data.get("conditions", {})
    print(f"evidence: {path.relative_to(REPO_ROOT)}  recorded {data['recorded_at']}")
    print(
        f"conditions: busy before {cond.get('cpu_busy_before')}, after "
        f"{cond.get('cpu_busy_after')}, contended={cond.get('contended')}"
    )
    if cond.get("contended"):
        print("NOTE: contended run; thresholds derived from it are conservative.")

    table: dict[tuple[str, str], list[tuple[int, float, int, bool]]] = {}
    for case in data["cases"]:
        p = case["params"]
        n = p["n"]
        t = threads_for_bytes(n * ITEMSIZE[p["dtype"]])
        if t < 2:
            continue  # below the byte floor nothing dispatches; not a candidate size
        variant = case["variants"].get(f"pyrallel_{t}t")
        speedup = case["speedups"].get(f"pyrallel_{t}t")
        ok = bool(variant and variant.get("correct"))
        table.setdefault((p["op"], p["dtype"]), []).append((n, speedup or 0.0, t, ok))

    print(f"\nschedule by bytes: {BYTE_SCHEDULE}\nmin win: {args.min_win}x\n")
    proposal: dict[str, dict[str, int]] = {}
    for (op, dtype), rows in sorted(table.items()):
        rows.sort()
        if not all(ok for _, _, _, ok in rows):
            print(f"{op:6} {dtype:8}: CORRECTNESS FAILURE in battery; excluded")
            continue
        chosen = None
        for i, (n, sp, t, _) in enumerate(rows):
            if all(s >= args.min_win for _, s, _, _ in rows[i:]):
                chosen = n
                break
        detail = "  ".join(f"{n}:{sp:.2f}x@{t}t" for n, sp, t, _ in rows)
        if chosen is None:
            print(f"{op:6} {dtype:8}: no win >= {args.min_win}x at any size; stays on stock   [{detail}]")
        else:
            print(f"{op:6} {dtype:8}: dispatch from n >= {chosen:>10,}   [{detail}]")
            proposal.setdefault(op, {})[dtype] = chosen

    print("\nSUPPORTED proposal (paste by hand):")
    for op, d in proposal.items():
        inner = ", ".join(f"_{k.upper().replace('FLOAT', 'F').replace('INT', 'I')}: {v:_}" for k, v in d.items())
        print(f'    "{op}": {{{inner}}},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
