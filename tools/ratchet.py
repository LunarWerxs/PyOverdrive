"""Keep-or-revert ratchet for unattended fast-path search.

WHY: fast paths are found by trying things, and trying things overnight
needs a judge that is not the agent doing the trying. This tool is that
judge for ONE candidate commit at a time. An agent (or a person) loops:

    1. edit src/pyoverdrive/fastpaths/<name>.py (and its selfcheck input,
       differential test, _MODULES entry) and commit on a search branch
    2. run  .venv/Scripts/python tools/ratchet.py --path <name>
    3. read the one-line verdict, go to 1

Each step the candidate must pass its differential test (the gate) and
then beat the incumbent on the measured speedup (the metric). A candidate
that fails either is discarded with `git reset --hard <incumbent>`, so the
branch tip is always the best version that passed, never a half-broken
tree. The idea is adapted from karpathy/autoresearch (program.md, MIT per
its README): fixed-budget attempts, one metric, keep or reset, and a
simplicity rule.

The metric is the WORST ratio (stock seconds / patched seconds) over every
cell tools/verify_no_pessimization.py judges for the path, each measured
in its own fresh process through the public API. The worst cell, not the
mean, because a path that wins one regime by losing another is exactly
what this project refuses to ship. A path needs a selfcheck input in
pyoverdrive/diagnostics.py to have any cell at all.

Rules, in order:
- gate fails, stage exceeds 2x --budget, or no cell measured: revert
- metric below 1.0 (slower than stock somewhere): revert
- metric >= incumbent * (1 + --min-gain): keep
- the change deletes more lines than it adds and the metric holds within
  --hold of the incumbent: keep (simpler code at equal speed is a win)
- anything else, including a gain too small to pay for added code: revert

The incumbent is the last kept commit in the ledger. With no ledger yet it
is --base (default HEAD~1) at 1.0x, i.e. stock NumPy: the right bar for a
brand-new path. To improve a path that already ships, run --init once on
the unchanged tip so the bar is the path's own measured speed.

The ledger (JSON lines) and every stage's log live under the worktree's
git directory, so a reset cannot erase them and `git status` never shows
them. A reverted commit stays reachable by the sha the ledger records.

Safety: refuses on a detached HEAD, on main/master, and on a dirty tree
(an uncommitted file would be measured but survive the reset).

A kept verdict is a search result, not shipping evidence: a path still
needs its Dyno benchmark, provenance record and the full
verify_no_pessimization sweep (docs/BUILD_SPEC.md section 10.2) before it
is listed as shipped. Run on a quiet machine; --require-quiet makes a
contended measurement inconclusive instead of a verdict.

Usage:
    .venv/Scripts/python tools/ratchet.py --path <name> [--test PATH]
        [--budget 600] [--min-gain 0.02] [--hold 0.01] [--base REV]
        [--sizes] [--shapes] [--rows] [--values] [--require-quiet]
        [--init] [--dry-run]

Exit 0 = kept (or --init recorded), 3 = reverted, 4 = inconclusive (quiet
run refused; tree untouched), 1 = refused or error (tree untouched).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED_BRANCHES = {"main", "master"}
KEEP, REVERT, INCONCLUSIVE, REFUSED = 0, 3, 4, 1


def decide(metric: float | None, incumbent: float, net_lines: int, *,
           min_gain: float, hold: float) -> tuple[str, str]:
    """The ratchet rule. Returns ("keep" | "revert", reason)."""
    if metric is None:
        return "revert", "no cell measured"
    if metric < 1.0:
        return "revert", f"slower than stock in some cell ({metric:.4f}x)"
    if metric >= incumbent * (1.0 + min_gain):
        return "keep", f"improved {incumbent:.4f}x -> {metric:.4f}x"
    if net_lines < 0 and metric >= incumbent * (1.0 - hold):
        return "keep", f"simpler ({net_lines} lines) and held {metric:.4f}x vs {incumbent:.4f}x"
    if metric > incumbent:
        return "revert", (f"gain {metric / incumbent - 1.0:+.2%} is under --min-gain "
                          f"for {net_lines:+d} lines")
    return "revert", f"no improvement ({metric:.4f}x vs {incumbent:.4f}x)"


def metric_from_evidence(evidence: dict, path: str) -> tuple[float | None, list[dict]]:
    """Worst valid ratio among the sweep's cells for exactly `path`.

    --only is a substring filter, so other paths whose names contain this
    one can appear in the evidence; they are ignored. Cells that errored,
    crashed or carry a non-finite ratio are not measurements.
    """
    judged = []
    for record in evidence.get("cells", []):
        ratio = record.get("ratio")
        if (record.get("path") != path or "error" in record or record.get("returncode")
                or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0):
            continue
        judged.append({"cell": record.get("cell"), "ratio": float(ratio)})
    if not judged:
        return None, []
    return min(c["ratio"] for c in judged), judged


def branch_refusal(branch: str | None) -> str | None:
    """Why the ratchet must not reset this branch, or None when it may."""
    if not branch:
        return "HEAD is detached; run on a search branch"
    if branch in PROTECTED_BRANCHES:
        return f"refusing to reset {branch}; run on a search branch"
    return None


def _git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    if check and proc.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _kill_tree(proc: subprocess.Popen) -> None:
    # The sweep spawns one child per cell; killing only the parent would
    # leave them measuring into the next attempt.
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def _run_bounded(cmd: list[str], log_path: Path, limit: float) -> int | None:
    """Run with output to a log; None when it exceeded `limit` and was killed."""
    extra = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **extra)
        try:
            return proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            return None


def _ledger_dir() -> Path:
    git_dir = Path(_git("rev-parse", "--git-dir"))
    if not git_dir.is_absolute():
        git_dir = ROOT / git_dir
    out = git_dir / "pyoverdrive-ratchet"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _incumbent(ledger: Path, base: str) -> tuple[str, float]:
    kept = None
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("verdict") in ("keep", "init"):
                kept = entry
    if kept is not None:
        return kept["head"], float(kept["metric"])
    return _git("rev-parse", "--verify", f"{base}^{{commit}}"), 1.0


def _net_lines(incumbent: str) -> int:
    net = 0
    for line in _git("diff", "--numstat", incumbent, "HEAD").splitlines():
        added, deleted = line.split("\t")[:2]
        if added != "-":  # binary files carry no line counts
            net += int(added) - int(deleted)
    return net


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/ratchet.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--path", required=True, help="Gearbox fast path name being searched")
    ap.add_argument("--test", action="append", default=[],
                    help="gate test file (repeatable); default "
                         "compatibility/differential/test_<path>_differential.py")
    ap.add_argument("--budget", type=float, default=600.0,
                    help="seconds per stage; a stage is killed at twice this")
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="fractional speedup a candidate that adds code must beat the incumbent by")
    ap.add_argument("--hold", type=float, default=0.01,
                    help="fractional slack for a candidate that removes code")
    ap.add_argument("--base", default="HEAD~1",
                    help="incumbent commit when the ledger has none (measured as stock, 1.0x)")
    for axis in ("sizes", "shapes", "rows", "values", "require-quiet"):
        ap.add_argument(f"--{axis}", action="store_true",
                        help=f"pass --{axis} to verify_no_pessimization.py")
    ap.add_argument("--init", action="store_true",
                    help="measure the current tip and record it as the incumbent")
    ap.add_argument("--dry-run", action="store_true", help="judge only: no reset, no ledger entry")
    args = ap.parse_args(argv)
    if args.budget <= 0 or args.min_gain < 0 or not 0 <= args.hold < 1:
        ap.error("--budget must be positive, --min-gain non-negative, --hold in [0, 1)")

    refusal = branch_refusal(_git("symbolic-ref", "--short", "-q", "HEAD", check=False) or None)
    if refusal is None and _git("status", "--porcelain"):
        refusal = "working tree is not clean; commit the candidate first"
    tests = args.test or [f"compatibility/differential/test_{args.path}_differential.py"]
    missing = [t for t in tests if not (ROOT / t).is_file()]
    if refusal is None and missing:
        refusal = f"gate test not found: {', '.join(missing)} (pass --test)"
    if refusal:
        print(f"REFUSED: {refusal}")
        return REFUSED

    head = _git("rev-parse", "HEAD")
    out = _ledger_dir()
    ledger = out / f"{args.path}.jsonl"
    try:
        incumbent, incumbent_metric = _incumbent(ledger, args.base)
    except RuntimeError as exc:
        print(f"REFUSED: no incumbent: {exc}")
        return REFUSED
    if not args.init and head == incumbent:
        print(f"REFUSED: HEAD {head[:9]} is the incumbent; commit a candidate first")
        return REFUSED
    descends = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                               incumbent, "HEAD"], capture_output=True).returncode == 0
    if not args.init and not descends:
        print(f"REFUSED: HEAD does not descend from the incumbent {incumbent[:9]}")
        return REFUSED

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = out / f"{stamp}-{args.path}-{head[:9]}"
    limit = 2 * args.budget
    entry = {"at": stamp, "path": args.path, "head": head, "incumbent": incumbent,
             "incumbent_metric": incumbent_metric, "metric": None, "cells": [],
             "net_lines": 0 if args.init else _net_lines(incumbent),
             "logs": {"gate": f"{prefix}-gate.log", "sweep": f"{prefix}-sweep.log"}}

    # Gate first: it is cheaper than the sweep and a wrong answer at any
    # speed is worthless (autoresearch's fast-fail on divergence).
    code = _run_bounded([sys.executable, "-m", "pytest", "-q", "-x", *tests],
                        Path(entry["logs"]["gate"]), limit)
    if code != 0:
        verdict, reason = "revert", ("gate exceeded 2x budget" if code is None
                                     else f"gate failed (exit {code})")
    else:
        evidence_path = Path(f"{prefix}-evidence.json")
        sweep = [sys.executable, "tools/verify_no_pessimization.py", "--only", args.path,
                 "--json", str(evidence_path)]
        sweep += [f"--{axis}" for axis in ("sizes", "shapes", "rows", "values")
                  if getattr(args, axis)]
        if args.require_quiet:
            sweep.append("--require-quiet")
        code = _run_bounded(sweep, Path(entry["logs"]["sweep"]), limit)
        entry["logs"]["evidence"] = str(evidence_path)
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            evidence = {}
        if args.require_quiet and "quiet run refused" in str(evidence.get("error", "")):
            print(f"INCONCLUSIVE: {evidence['error']}; tree untouched")
            return INCONCLUSIVE
        entry["metric"], entry["cells"] = metric_from_evidence(evidence, args.path)
        if code is None:
            verdict, reason = "revert", "sweep exceeded 2x budget"
        elif args.init:
            verdict, reason = ("init", "incumbent recorded") if entry["metric"] is not None \
                else ("revert", "no cell measured")
        else:
            verdict, reason = decide(entry["metric"], incumbent_metric, entry["net_lines"],
                                     min_gain=args.min_gain, hold=args.hold)

    entry.update(verdict=verdict, reason=reason)
    shown = "-" if entry["metric"] is None else f"{entry['metric']:.4f}x"
    if args.init and verdict != "init":
        print(f"REFUSED: cannot record {head[:9]} as incumbent: {reason}; logs in {out}")
        return REFUSED
    if not args.dry_run:
        if verdict == "revert":
            _git("reset", "--hard", incumbent)
        with open(ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    label = {"keep": "KEEP", "init": "INIT", "revert": "REVERT"}[verdict]
    after = "" if verdict != "revert" else (f"; reset to {incumbent[:9]}" if not args.dry_run
                                            else "; dry run, tree untouched")
    print(f"{label} {head[:9]} {args.path} {shown}: {reason}{after}")
    return REVERT if verdict == "revert" else KEEP


if __name__ == "__main__":
    raise SystemExit(main())
