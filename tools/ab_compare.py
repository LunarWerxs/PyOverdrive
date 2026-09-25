"""Judge A/B timing evidence, refusing comparisons that cannot mean anything.

WHY: verify_no_pessimization.py --json records every cell's raw interleaved
timings, but the only verdict ever drawn from them was a median ratio
against --min. That cannot say "no conclusion", and it will happily compare
runs from different machines, NumPy builds, thread settings or a contended
box. This tool reads the same evidence and returns one of the verdicts in
src/pyoverdrive/abverdict.py per cell - faster, slower, no-win or
inconclusive - with a Holm correction across the whole suite.

Two questions, one rule set:

    # Is each fast path's "measured faster" claim supported? (stock vs patched)
    .venv/Scripts/python tools/ab_compare.py RUN1.json RUN2.json RUN3.json --min-speedup 1.1

    # Did a change make the patched route slower? (old build vs new build)
    .venv/Scripts/python tools/ab_compare.py --baseline OLD1.json OLD2.json OLD3.json -- NEW1.json NEW2.json NEW3.json

An independent run is one child process's record: the initial reading and
any confirmation re-measure of a cell each count once, and so does each
evidence file. A cell with fewer than 3 runs per side is inconclusive, never
passed. Produce more runs by repeating the verify_no_pessimization --json
sweep with the same settings.

Refused outright (exit 2), never migrated or averaged over:

    invalid       a report whose schema_version is not the one this reads,
                  from another tool, incomplete, or with unusable samples
    incomparable  reports whose environment contract differs - machine
                  fingerprint, SIMD features, Python/NumPy/BLAS, rounds and
                  sample size, thread environment, saved calibration - or
                  that ran contended or with unknown load; and in claim mode
                  reports from different source builds (source sha256)

Claim mode (no --baseline) passes only when every cell is "faster":
exit 0 = every cell is conclusively faster than the bar; 1 = any cell is
not. Build mode (--baseline) asks only for no regression: exit 0 = no cell
is "slower"; 1 = at least one is (or, with --require-win, any cell is not
"faster"). Exit 2 = invalid/incomparable, including a cell measured on only
one side of a --baseline comparison.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pyoverdrive import abverdict  # noqa: E402

SCHEMA_VERSION = 1  # verify_no_pessimization evidence; a bump is refused
TOOL = "verify_no_pessimization"
_MACHINE_KEYS = ("fingerprint", "cpu", "machine", "system", "logical_cores",
                 "python", "numpy", "blas", "simd_features")
_SETTING_KEYS = ("rounds", "sample_seconds", "any_core", "environment")


class Refused(Exception):
    """A report set that must not be compared; the message says why."""

    def __init__(self, kind: str, why: str):
        super().__init__(f"{kind.upper()}: {why}")
        self.kind = kind


def load_report(path: Path) -> dict:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refused("invalid", f"{path}: unreadable ({exc})") from exc
    if not isinstance(report, dict):
        raise Refused("invalid", f"{path}: not an evidence object")
    if report.get("tool") != TOOL:
        raise Refused("invalid", f"{path}: not {TOOL} evidence")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise Refused("invalid", f"{path}: schema_version "
                      f"{report.get('schema_version')!r} is not {SCHEMA_VERSION}; "
                      f"re-measure instead of migrating")
    if report.get("completed") is not True:
        raise Refused("invalid", f"{path}: the sweep did not complete")
    if not isinstance(report.get("cells"), list):
        raise Refused("invalid", f"{path}: no cell records")
    conditions = report.get("conditions") or {}
    if conditions.get("contended") is not False or conditions.get("load_known") is not True:
        raise Refused("incomparable", f"{path}: ran contended or with unknown "
                      f"foreign load; its timings describe the load, not the build")
    return report


def contract(report: dict) -> dict:
    """Everything that must match for two timings to be comparable."""
    fp = report.get("fingerprint") or {}
    settings = report.get("settings") or {}
    return {
        "machine": {k: fp.get(k) for k in _MACHINE_KEYS},
        "settings": {k: settings.get(k) for k in _SETTING_KEYS},
        "calibration": (report.get("calibration") or {}).get("saved"),
    }


def source_sha(report: dict):
    return (report.get("source") or {}).get("sha256")


def require_same(reports: list[tuple[Path, dict]], *, same_source: bool) -> None:
    first_path, first = reports[0]
    want = contract(first)
    for path, report in reports[1:]:
        got = contract(report)
        diff = [f"{part}.{k}" for part in ("machine", "settings")
                for k in want[part] if want[part][k] != got[part][k]]
        if want["calibration"] != got["calibration"]:
            diff.append("calibration")
        if diff:
            raise Refused("incomparable", f"{path} vs {first_path}: environment "
                          f"contract differs ({', '.join(diff)})")
        if same_source and source_sha(report) != source_sha(first):
            raise Refused("incomparable", f"{path} vs {first_path}: different "
                          f"source builds; use --baseline to compare builds")


def runs_by_cell(reports: list[tuple[Path, dict]]) -> dict[str, list[tuple[float, float]]]:
    """(stock median, patched median) per usable child record, per cell."""
    out: dict[str, list[tuple[float, float]]] = {}
    for _, report in reports:
        for rec in report["cells"]:
            if not isinstance(rec, dict) or rec.get("returncode") != 0 or "error" in rec:
                continue
            stock, patched = rec.get("stock_seconds"), rec.get("patched_seconds")
            if not (isinstance(stock, list) and stock and isinstance(patched, list) and patched):
                continue
            try:
                pair = (statistics.median(stock), statistics.median(patched))
            except TypeError:
                pair = (float("nan"), float("nan"))  # abverdict marks it invalid
            out.setdefault(str(rec.get("cell")), []).append(pair)
    return out


def build_cells(candidate: list[tuple[Path, dict]],
                baseline: list[tuple[Path, dict]] | None) -> dict:
    new = runs_by_cell(candidate)
    if baseline is None:
        return {name: ([s for s, _ in runs], [p for _, p in runs])
                for name, runs in sorted(new.items())}
    old = runs_by_cell(baseline)
    # WHY: a cell measured on one side only would silently vanish, letting a
    # partial comparison pass as a full one; refuse and name the cells.
    one_sided = sorted(set(old) ^ set(new))
    if one_sided:
        raise Refused("invalid", "cells measured on only one side: "
                      + ", ".join(f"{n} ({'baseline' if n in old else 'candidate'} only)"
                                  for n in one_sided))
    return {name: ([p for _, p in old[name]], [p for _, p in new[name]])
            for name in sorted(old)}


def judge(candidate_paths: list[Path], baseline_paths: list[Path] | None,
          min_speedup: float) -> dict:
    candidate = [(p, load_report(p)) for p in candidate_paths]
    baseline = [(p, load_report(p)) for p in baseline_paths] if baseline_paths else None
    if baseline is None:
        require_same(candidate, same_source=True)
    else:
        require_same(baseline, same_source=True)
        require_same(candidate, same_source=True)
        require_same([baseline[0], candidate[0]], same_source=False)
    cells = build_cells(candidate, baseline)
    if not cells:
        raise Refused("invalid", "no cell was measured on both sides")
    return abverdict.compare_suite(cells, min_speedup=min_speedup)


def exit_code(verdicts: dict, require_win: bool) -> int:
    kinds = {v["verdict"] for v in verdicts.values()}
    if "invalid" in kinds:
        return 2
    if "slower" in kinds:
        return 1
    if require_win and kinds != {"faster"}:
        return 1
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reports", nargs="+", type=Path,
                    help="verify_no_pessimization --json evidence (the candidate side)")
    ap.add_argument("--baseline", nargs="+", type=Path,
                    help="evidence of the build to compare against; without it, "
                         "each report's stock timings are the baseline")
    ap.add_argument("--min-speedup", type=float, default=1.0,
                    help="bar a 'faster' verdict's whole interval must clear")
    ap.add_argument("--require-win", action="store_true",
                    help="fail unless every cell is conclusively faster "
                         "(always on without --baseline)")
    ap.add_argument("--json", action="store_true", help="print verdicts as JSON")
    args = ap.parse_args(argv[1:])

    try:
        verdicts = judge(args.reports, args.baseline, args.min_speedup)
    except Refused as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(verdicts, indent=2, default=str))
    else:
        for name, v in verdicts.items():
            ci = v.get("ci")
            span = f"{v['speedup']:.3f}x [{ci[0]:.3f}-{ci[1]:.3f}]" if ci else "-"
            print(f"  {v['verdict']:<12s} {name:40s} {span:24s} {v.get('reason', '')}")
        counts = {k: sum(v["verdict"] == k for v in verdicts.values())
                  for k in abverdict.VERDICTS}
        print("  " + ", ".join(f"{n} {k}" for k, n in counts.items() if n))
    # WHY: a speedup claim that exits 0 on "inconclusive" would let noise
    # pass as a win, so claim mode always requires every cell to be faster.
    return exit_code(verdicts, args.require_win or args.baseline is None)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
