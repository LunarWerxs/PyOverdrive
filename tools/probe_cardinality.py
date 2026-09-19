"""Where does the sort-based set family stop paying as CARDINALITY rises?

The four-axis sweep caught `unique_sort`, `unique_values_sort` and
`intersect_sorted` losing 0.68-0.98x when their integer operands are drawn
from only 16 distinct values. All three gate on size and dtype, neither of
which can see that, and the shipped comments already half-knew: unique_sort
records "1.22-1.48x at n=1000 low cardinality" against 8.5-21.6x high, and
intersect_sorted records low-cardinality inputs as "marginal-to-losing".
Both numbers came from a calibration battery whose "low" was nowhere near
16 distinct values.

The mechanism, on stock's side, is a hash table: numpy 2.3 rewrote
np.unique to try one before sorting (gh-26018). A hash table over C
distinct values is a working set of about C entries, so while C is small it
lives in L1/L2 and beats an O(n log n) sort outright; as C grows the table
stops fitting and the sort catches up. That predicts a crossing in C - an
ABSOLUTE distinct count tied to cache size - rather than in C/n, and the
grid below is built to tell those two apart, because they imply completely
different gates.

    .venv/Scripts/python tools/probe_cardinality.py
    .venv/Scripts/python tools/probe_cardinality.py --op unique --dtype int64

RUN IT ON THE QUIET BOX. These are timings.

Use --json PATH --require-quiet for fingerprinted, load-qualified evidence.
--sizes and --cardinalities accept repeatable comma-separated selectors;
`u` retains the broad-range draw. --floor-neighbors selects the first input
length admitted by each shipped row and its immediate neighbors. For
intersect, n remains the FIRST operand length, with max(2, n//10) elements
in the second; the combined size is recorded. Defaults include the historical
grid plus those current floor neighbors. Fixed cardinality labels describe
the draw pool, not a guarantee that every value occurs in a finite sample.

Exit 0 means at least one cell was measured with every requested independent
repeat completed and no ratio below 1.0x. Losses, missing measurements,
worker failures, and failed quiet requirements return nonzero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402

import pyoverdrive  # noqa: E402
from pyoverdrive import _cpuclass as cpuclass  # noqa: E402
import verify_no_pessimization as sweep  # noqa: E402
from verify_no_pessimization import (  # noqa: E402
    _measure, _fingerprint, _foreign_load, _source_revision, _write_json,
)

# Distinct values. 16 is the sweep's own low end; the top of the walk is
# "all distinct", which is what every calibration battery used and is the
# regime the shipped thresholds were measured in.
CARDINALITIES = (16, 256, 4_096, 65_536)
# START AT THE ROW FLOORS, not at a comfortable size. The sweep's losing
# cells are ROW cells, and a row cell is measured at its row's own floor -
# 64 elements for unique_sort's 32/64-bit rows, 400 combined for
# intersect_sorted's. A first cut of this probe started at 100,000 and
# measured 2.11x for unique at 16 distinct values, i.e. a comfortable win
# where the sweep reports 0.95x, because it was not looking at the sizes
# the sweep was reporting on. The losing regime is small arrays.
SIZES = (64, 400, 1_000, 3_000, 10_000, 30_000, 100_000)
# THE UNSIGNED ROWS ARE NOT THE SIGNED ONES. That was assumed when the first
# floors were derived from int64/int32/int16 alone, and the verification run
# refuted it immediately: after the floors moved, every int32 and int64 cell
# came back clean while uint64 stayed red - unique_values_sort#uint64*30 at
# 0.75x, *10 at 0.96x, *100 at 0.98x, %d at 0.89x - at sizes where int64
# measures 1.24x. Same width, same sort, different answer, and the shipped
# tables key on the exact dtype, so each one is its own row and its own
# promise.
DTYPES = ("int64", "uint64", "int32", "uint32", "int16", "uint16")
OPS = ("unique", "unique_values", "intersect")
# Historical measured floors retained ONLY to replay withdrawn dtype rows.
# These do not restore dispatch: _one still records NODISPATCH from Gearbox.
_WITHDRAWN_FLOORS = {
    ("unique", "uint16"): 100_000, ("unique_values", "uint16"): 100_000,
    ("unique", "int16"): 10_000, ("unique_values", "int16"): 10_000,
    ("unique", "int64"): 1000, ("unique", "uint64"): 1000,
    ("unique_values", "int64"): 1000, ("unique_values", "uint64"): 1000,
    ("intersect", "int16"): 12_000, ("intersect", "uint16"): 12_000,
    ("intersect", "int64"): 10_000, ("intersect", "uint64"): 10_000,
    ("intersect", "uint8"): 12_000,
    ("intersect", "int8"): 12_000,
}


def _draw(n: int, card: int | None, dtype: str, seed: int) -> np.ndarray:
    """n values from `card` distinct ones (None = as distinct as possible)."""
    rng = np.random.default_rng(seed)
    info = np.iinfo(dtype)
    if card is None:
        span = min(info.max - info.min, 1 << 62)
        lo = max(info.min, -(span // 2))
        return rng.integers(lo, lo + span, size=n).astype(dtype)
    # The pool must itself be EXACTLY `card` distinct values, or the
    # effective cardinality is lower than the label and every number in the
    # table is against the wrong x-axis. For int16 a pool bigger than the
    # dtype's range cannot exist, which is why those cells are refused
    # rather than silently drawn smaller.
    span = int(info.max) - int(info.min) + 1
    if card > span:
        raise ValueError("cardinality exceeds the dtype's range")
    lo = int(info.min)
    # TWO CONSTRUCTIONS, and the dense one is not an optimisation - the
    # sparse one CANNOT terminate without it. Rejection-sampling distinct
    # values is the coupon collector: asking int16 for 65,536 distinct
    # values is asking for every value the dtype has, which needs about
    # 65,536*ln(65,536) ~ 730,000 draws, and a loop that tops up by
    # (missing*2+8) is drawing ten values into a space of 65,536 by the
    # end. The first run of this grid hung there.
    if card * 4 >= span:
        # dense: enumerate the range and take a random subset
        pool = rng.permutation(span)[:card].astype(np.int64) + lo
        pool.sort()
    else:
        # sparse: draw and dedup. choice(replace=False) is not an option -
        # it wants to materialise the whole range, and for int64 that
        # range is 1.8e19.
        hi = lo + min(span, max(card * 64, 1 << 20))
        pool = np.unique(rng.integers(lo, hi, size=card * 2, dtype=np.int64))
        while pool.size < card:
            more = rng.integers(lo, hi, size=(card - pool.size) * 4 + 64,
                                dtype=np.int64)
            pool = np.unique(np.concatenate([pool, more]))
        pool = pool[:card]
    return pool[rng.integers(0, card, size=n)].astype(dtype)


def _inputs(op: str, n: int, card: int | None, dtype: str):
    if op == "unique":
        return "numpy.unique", (_draw(n, card, dtype, 7),), {}
    if op == "unique_values":
        return "numpy.unique_values", (_draw(n, card, dtype, 8),), {}
    # intersect1d: second operand is a tenth of the first. Independent seeds
    # generate pools in the same value range; overlap is not guaranteed.
    a = _draw(n, card, dtype, 9)
    b = _draw(max(2, n // 10), card, dtype, 10)
    return "numpy.intersect1d", (a, b), {}


def _one(spec: str, fast_under: float | None, rounds: int = 7,
         warmup: bool = False, evidence=None) -> str:
    if fast_under is not None and cpuclass.probe_us() > fast_under:
        return "SLOW-CORE"
    op, n, card, dtype = spec.split(",")
    c = None if card == "u" else int(card)
    try:
        name, args, kwargs = _inputs(op, int(n), c, dtype)
    except ValueError as exc:
        return f"NOFIT {exc}"
    pyoverdrive.enable()
    from pyoverdrive.dispatcher.gearbox import GEARBOX

    chosen, reason = GEARBOX.decide(name, args, kwargs)
    if evidence is not None:
        evidence.update(op=name, path=chosen,
                        dispatch={"chosen": chosen, "reason": reason},
                        inputs=sweep._input_description(args),
                        combined_size=sum(value.size for value in args),
                        requested_pool_cardinality=c,
                        observed_cardinalities=[int(GEARBOX.stock_fn("numpy.unique")(value).size)
                                                for value in args])
    if chosen is None or chosen == "stock":
        return f"NODISPATCH {reason}"
    path = next(p for p in GEARBOX._paths[name] if p.name == chosen)
    mode = path.provenance.get("comparison_mode")
    if warmup:
        # A whole discarded measurement, not more rounds inside one: the
        # artifact is per-PROCESS first touch of a fresh allocation, and
        # _measure's internal warm-up already ran inside the cold one.
        discarded = {} if evidence is not None else None
        _measure(name, args, kwargs, rounds=max(3, rounds // 3),
                 evidence=discarded, comparison_mode=mode)
        if evidence is not None:
            evidence["warmup_measurement"] = discarded
    ratio = _measure(name, args, kwargs, rounds=rounds, evidence=evidence,
                     comparison_mode=mode)
    return "UNMEASURABLE" if ratio is None else f"RATIO {ratio:.4f} {chosen}"


def _parse_args(argv: list[str]):
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--fast-under", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--op", choices=OPS, action="append", default=[])
    ap.add_argument("--dtype", choices=DTYPES + ("int8", "uint8"), action="append", default=[])
    ap.add_argument("--sizes", action="append", default=[], help="repeatable comma-separated first-operand counts")
    ap.add_argument("--cardinalities", action="append", default=[], help="repeatable comma-separated pool sizes or u")
    ap.add_argument("--floor-neighbors", action="store_true", help="only each row's current floor and neighbors")
    ap.add_argument("--json", metavar="PATH", help="write raw fingerprinted evidence")
    ap.add_argument("--require-quiet", action="store_true", help="require known foreign CPU load <=20%% before/after")
    ap.add_argument("--evidence-cell", help=argparse.SUPPRESS)
    ap.add_argument("--rounds", type=int, default=7,
                    help="timed rounds per cell, interleaved")
    ap.add_argument("--warmup", action="store_true",
                    help="discard the FIRST measurement of each cell. Not "
                         "cosmetic: measured six times in a row on the idle "
                         "box, unique/int64 at n=100,000 and 16 distinct "
                         "values reads 0.91x once and then 1.73x five times "
                         "with almost no spread - and 0.84x then 1.25x twice "
                         "at 256 distinct. The cold reading is our own "
                         "allocation paying first-touch page faults on 800 KB "
                         "while stock's hash table for 16 distinct values is "
                         "a few hundred bytes and pays none. Without this, "
                         "--repeat's MINIMUM latches onto that artifact every "
                         "time and reports a path as losing when it wins.")
    ap.add_argument("--repeat", type=int, default=1,
                    help="measure every cell this many times in independent "
                         "processes and KEEP THE WORST. A threshold is a "
                         "green, so it has to reproduce exactly as a red "
                         "does - the first run of this grid put int64 unique "
                         "at 0.88x between neighbours of 1.55x and 3.22x, "
                         "which is not a mechanism, it is noise, and a floor "
                         "set from it would be a number nobody could trust.")
    ap.add_argument("--retries", type=int, default=6)
    args = ap.parse_args(argv[1:])
    try:
        args.sizes = list(dict.fromkeys(int(x) for group in args.sizes for x in group.split(",")))
        args.cardinalities = list(dict.fromkeys(None if x.strip() == "u" else int(x)
                                                for group in args.cardinalities for x in group.split(",")))
        if any(n <= 0 for n in args.sizes) or any(c is not None and c <= 0 for c in args.cardinalities):
            raise ValueError("sizes/cardinalities must be positive")
        if min(args.rounds, args.repeat, args.retries) < 1:
            raise ValueError("rounds/repeat/retries must be positive")
    except ValueError as exc:
        ap.error(str(exc))
    if args.sizes and args.floor_neighbors:
        ap.error("--sizes and --floor-neighbors are alternatives")
    args.cardinalities = args.cardinalities or [*CARDINALITIES, None]
    return args


def _floor(op, dtype):
    from pyoverdrive.fastpaths import intersect_sorted, unique_sort
    module = intersect_sorted if op == "intersect" else unique_sort
    live = module._THRESHOLDS.get(np.dtype(dtype))
    return live if live is not None else _WITHDRAWN_FLOORS[(op, np.dtype(dtype).name)]


def _sizes_for(op, dtype, args):
    if args.sizes:
        return args.sizes
    floor = _floor(op, dtype)
    if op == "intersect":
        first = max(1, floor * 10 // 11 - 2)
        while first + max(2, first // 10) < floor:
            first += 1
        floor = first
    neighbors = [max(1, floor - 1), floor, floor + 1]
    return neighbors if args.floor_neighbors else sorted(set(SIZES) | set(neighbors))


def _protocol_proc(proc):
    # Reuse the sweep's strict crash/protocol checks while preserving this
    # older probe's NOFIT/NODISPATCH words in its public table and sidecar.
    line = ((proc.stdout or "").strip().splitlines()[-1:] or [""])[0]
    translated = line
    if line.startswith(("NOFIT ", "NODISPATCH ")) or line == "UNMEASURABLE":
        translated = "SKIP " + line
    return subprocess.CompletedProcess(proc.args, proc.returncode, translated, proc.stderr)


def _child_result(proc, cell):
    line = ((proc.stdout or "").strip().splitlines()[-1:] or [""])[0]
    sweep._child_parts(_protocol_proc(proc), cell)
    return line


def _collect_evidence(args, sidecar, cell, repeat, attempt, proc):
    normalized = _protocol_proc(proc)
    if normalized.stdout.startswith("SKIP "):
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            record["original_result"] = record["result"]
            record["result"] = "SKIP " + record["result"]
            _write_json(sidecar, record)
        except (OSError, ValueError, KeyError, TypeError):
            pass  # the shared collector records and rejects malformed evidence
    sweep._collect_child_evidence(args, sidecar, cell, f"repeat-{repeat + 1}", attempt + 1, normalized)


def _measure_cell(cell, args, cutoff):
    cmd = [sys.executable, str(Path(__file__)), "--one", cell, "--rounds", str(args.rounds)]
    if args.warmup:
        cmd.append("--warmup")
    if cutoff is not None:
        cmd += ["--fast-under", f"{cutoff:.3f}"]
    ratios = []
    for repeat in range(args.repeat):
        for attempt in range(args.retries):
            child_cmd = list(cmd)
            sidecar = None
            if getattr(args, "_evidence", None) is not None:
                sidecar = Path(args._evidence_dir) / f"{len(args._evidence['cells'])}.json"
                child_cmd += ["--evidence-cell", str(sidecar)]
            proc = subprocess.run(child_cmd, capture_output=True, text=True, cwd=str(REPO))
            if sidecar is not None:
                _collect_evidence(args, sidecar, cell, repeat, attempt, proc)
            line = _child_result(proc, cell)
            if line != "SLOW-CORE":
                break
        if not line.startswith("RATIO "):
            if ratios:
                raise RuntimeError(f"incomplete repeats for {cell}: {line}")
            return None, line
        ratios.append(float(line.split()[1]))
    return min(ratios), line


def _run_grid(args):
    classes = cpuclass.classify()
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}")
    print(cpuclass.describe(classes))
    cutoff = cpuclass.fast_cutoff(classes)
    cards = [str(c) if c is not None else "u" for c in args.cardinalities]
    evidence = getattr(args, "_evidence", None)
    if evidence is not None:
        evidence.update(cpu_classes=classes, fast_core_cutoff_us=cutoff,
                        selected_cells=[f"{op},{n},{card},{dtype}"
                                        for op in (args.op or OPS)
                                        for dtype in (args.dtype or DTYPES)
                                        for n in _sizes_for(op, dtype, args)
                                        for card in cards])

    losses, judged, skipped = [], 0, 0
    for op in (args.op or list(OPS)):
        for dtype in (args.dtype or list(DTYPES)):
            print(f"\n=== {op}, {dtype} " + "=" * 34 + "\n")
            print(f"{'n':>12s} " +
                  " ".join(f"{c:>9s}" for c in cards) + "   (distinct values)")
            for n in _sizes_for(op, dtype, args):
                row = []
                for card in cards:
                    cell = f"{op},{n},{card},{dtype}"
                    ratio, line = _measure_cell(cell, args, cutoff)
                    if ratio is not None:
                        judged += 1
                        if ratio < 1.0:
                            losses.append({"cell": cell, "ratio": ratio})
                        row.append(f"{ratio:>8.4f}x")
                    else:
                        skipped += 1
                        label = "n/a" if line.startswith("NOFIT") else "-" if line.startswith("NODISPATCH") else line.split()[0][:9]
                        row.append(f"{label:>9s}")
                print(f"{n:>12,d} " + " ".join(row))

    print("\n'u' = drawn across the dtype's whole range (as distinct as it "
          "gets, and the regime every shipped threshold was measured in). "
          "'n/a' = that many distinct values do not fit the dtype. "
          "'-' = the shipped predicate refuses the cell.")
    print(f"judged {judged} cells; skipped {skipped}; cells below 1.0x: {len(losses)}")
    if evidence is not None:
        evidence["summary"] = {"judged": judged, "skipped": skipped, "losses": losses}
    if not judged:
        print("NOT verified: no cell was measured")
    return int(not judged or bool(losses))


def _source_manifest():
    manifest = sweep._source_manifest()
    manifest["files"]["tools/probe_cardinality.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    encoded = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def _run_evidence(args):
    if not args.json and not args.require_quiet:
        return _run_grid(args)
    evidence = {"schema_version": 1, "tool": "probe_cardinality",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "fingerprint": _fingerprint(), "source_revision": _source_revision(),
                "source": _source_manifest(),
                "selection": {key: getattr(args, key) for key in ("op", "dtype", "sizes", "cardinalities", "floor_neighbors")},
                "settings": {key: getattr(args, key) for key in ("rounds", "repeat", "retries", "warmup", "require_quiet")},
                "conditions": {"quiet_threshold": 0.20, "cpu_busy_before": _foreign_load()},
                "cells": [], "completed": False}
    evidence["settings"]["environment"] = {key: os.environ.get(key) for key in (
        "PYOVERDRIVE_THREADS", "PYOVERDRIVE_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    evidence["calibration"] = sweep._calibration_metadata()
    code = 1
    try:
        before = evidence["conditions"]["cpu_busy_before"]
        if args.require_quiet and (before is None or before > 0.20):
            evidence["error"] = f"quiet run refused: foreign CPU load is {before!r}"
            print("NOT verified: " + evidence["error"])
        else:
            with tempfile.TemporaryDirectory(prefix="pyoverdrive-cardinality-") as scratch:
                args._evidence = evidence if args.json else None
                args._evidence_dir = scratch
                code = _run_grid(args)
                evidence["completed"] = "summary" in evidence if args.json else True
    except BaseException as exc:
        evidence["error"] = repr(exc)
        print(f"NOT verified: {exc}", file=sys.stderr)
        if not isinstance(exc, Exception):
            raise
    finally:
        conditions = evidence["conditions"]
        conditions["cpu_busy_after"] = _foreign_load()
        loads = [conditions[key] for key in ("cpu_busy_before", "cpu_busy_after")]
        conditions["contended"] = any(value is not None and value > 0.20 for value in loads)
        conditions["load_known"] = all(value is not None for value in loads)
        if args.require_quiet and (conditions["contended"] or not conditions["load_known"]):
            code = 1
            print("NOT verified: quiet conditions were not met before and after the run")
        evidence["exit_code"] = code
        if args.json:
            _write_json(args.json, evidence)
    return code


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if not args.one:
        return _run_evidence(args)
    evidence = {"cell": args.one} if args.evidence_cell else None
    try:
        result = _one(args.one, args.fast_under, args.rounds, args.warmup, evidence=evidence)
        if evidence is not None:
            evidence["result"] = result
        print(result)
        return 0
    except BaseException as exc:
        if evidence is not None:
            evidence["error"] = repr(exc)
        raise
    finally:
        if evidence is not None:
            _write_json(args.evidence_cell, evidence)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
