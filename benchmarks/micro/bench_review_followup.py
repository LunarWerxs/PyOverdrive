"""Isolated public-API evidence for review gaps; no runtime thresholds change.

Run on an idle host:
  python benchmarks/micro/bench_review_followup.py --require-quiet --json result.json
Use --list, --family argmax|vectorize|out, or repeat --only SUBSTRING to select.
Use --cells-file FILE for exact newline-separated cell IDs, including IDs
from an earlier artifact's unverified_cells list. Selection bytes and names
are hashed in the evidence. Noisy cells and exhausted slow-core attempts
remain explicitly unverified while later cells run; unresolved coverage
returns nonzero. Correctness and execution errors still abort immediately.

Argmax explicitly enables its calibration-gated candidate in each child;
below-floor neighbors remain measured stock fallbacks. Vectorize constructs
both public classes outside timing and measures their instance calls. Output
cells compare stock and the public candidate with alias checks enabled and
disabled. The latter is a benchmark-local override, allowed only after the
real check proves the buffers safe, and restored even on failure. Each call
resets mutable inputs OUTSIDE its timed interval. Result consumption is timed.
Float64 add with exact in-place output intentionally measures the stock
dispatch refusal; its execution evidence names this withdrawn regime.

Fresh processes use pristine shipped calibration tables, bypassing saved
machine calibration only for this experiment. All inputs are deterministic.
Raw ratios are evidence, not a claim of a confirmed loss or cross-host win.
Exit zero requires complete, correct coverage and any requested quiet gate;
speed regressions are recorded for independent confirmation, not hidden.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pyoverdrive
from pyoverdrive import _cpuclass as cpuclass
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths import parallel_binary, parallel_ufunc
from pyoverdrive.parallel import pyrallel
from tools import verify_no_pessimization as sweep


@dataclass(frozen=True)
class Cell:
    family: str
    op: str
    dtype: str
    shape: tuple[int, ...]
    alias: str = "none"

    @property
    def name(self):
        return ":".join((self.family, self.op, self.dtype,
                         "x".join(map(str, self.shape)), self.alias))


def cells() -> dict[str, Cell]:
    specs = []
    shapes = ((2999, 3000), (3000, 2999), (3000, 3000),
              (4500, 3000), (3000, 4500), (10000, 1000))
    for dtype in ("float64", "float32", "int64"):
        specs.extend(Cell("argmax", "argmax", dtype, shape) for shape in shapes)
    for op in ("sin", "sqrt", "exp"):
        specs.extend(Cell("vectorize", op, "float64", (n,))
                     for n in (16, 10_000, 1_000_000))
    for mod, op in ((parallel_ufunc, "sin"), (parallel_binary, "add")):
        for dtype, floor in mod.SHIPPED[op].items():
            for factor in (1, 3):
                specs.extend(Cell("out", op, np.dtype(dtype).name,
                                  (int(floor) * factor,), alias)
                             for alias in ("disjoint", "inplace"))
    return {c.name: c for c in specs}


@contextmanager
def without_alias_guard(inputs, out):
    original = pyrallel.output_alias_safe
    if out is None or not original(inputs, out):
        raise ValueError("unsafe or unknown output overlap in guard benchmark")
    try:
        pyrallel.output_alias_safe = lambda inputs, out: True
        yield
    finally:
        pyrallel.output_alias_safe = original


@dataclass
class Workload:
    stock: Callable
    candidate: Callable
    reset: Callable
    inputs: tuple
    out: np.ndarray | None
    path: str
    decision: tuple


@contextmanager
def workload(cell: Cell):
    if pyoverdrive.enabled():
        raise RuntimeError("follow-up cells require initially unpatched NumPy")
    rng = np.random.default_rng(20260919)
    dtype = np.dtype(cell.dtype)
    op = "numpy.vectorize" if cell.family == "vectorize" else f"numpy.{cell.op}"
    path = {"argmax": "argmax_blocked_transpose", "vectorize": "vectorize_ufunc_direct",
            "out": f"pyrallel_{cell.op}"}[cell.family]
    paths = ([GEARBOX._class_paths[op]] if cell.family == "vectorize"
             else GEARBOX._paths[op])
    registered = next(p for p in paths if p.name == path)
    was_enabled = registered.enabled
    stock = getattr(np, op.split(".")[-1])
    out = None
    if cell.family == "argmax":
        a = (rng.integers(0, 100_000, size=cell.shape, dtype=dtype)
             if dtype.kind == "i" else rng.random(cell.shape, dtype=dtype))
        inputs, kwargs = (a,), {"axis": 0}
        reset = lambda: None
        stock_call = lambda: stock(*inputs, **kwargs)
    elif cell.family == "vectorize":
        inputs, kwargs = (np.linspace(0.1, 1.0, cell.shape[0]),), {}
        func = getattr(np, cell.op)
        stock_instance = stock(func)
        stock_call = lambda: stock_instance(*inputs)
        reset = lambda: None
    else:
        # Copying is explicit and untimed. Every side sees the same values,
        # including sin(x, out=x), whose repeated application changes x.
        template = np.linspace(0.1, 8.0, cell.shape[0], dtype=dtype)
        a = template.copy()
        inputs = (a,) if cell.op == "sin" else (a, np.ones(cell.shape, dtype=dtype))
        out = a if cell.alias == "inplace" else np.empty_like(a)
        kwargs = {"out": out}
        reset = lambda: np.copyto(a, template)
        stock_call = lambda: stock(*inputs, **kwargs)
    try:
        pyoverdrive.enable_path(path)  # explicit experiment, restored below
        pyoverdrive.enable([op])
        patched = getattr(np, op.split(".")[-1])
        if cell.family == "vectorize":
            candidate_instance = patched(func)
            if getattr(candidate_instance, "__pyoverdrive_direct__", None) is not func:
                raise RuntimeError(f"direct ufunc not installed: {cell.name}")
            candidate = lambda: candidate_instance(*inputs)
            decision = GEARBOX.decide(op, (func,), {})
        else:
            candidate = lambda: patched(*inputs, **kwargs)
            decision = GEARBOX.decide(op, inputs, kwargs)
        yield Workload(stock_call, candidate, reset, inputs, out, path, decision)
    finally:
        pyoverdrive.disable()
        GEARBOX.set_path_enabled(path, was_enabled)
        pyrallel.shutdown()


def _equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and np.array_equal(a.view(np.uint8), b.view(np.uint8)))


def _sample(call, reset):
    reset()
    start = time.perf_counter()
    sweep._consume(call())
    return time.perf_counter() - start


def _withdrawn_inplace_add(cell):
    return (cell.family == "out" and cell.op == "add"
            and cell.dtype == "float64" and cell.alias == "inplace")


def measure(cell: Cell, rounds: int) -> dict:
    with workload(cell) as w:
        calls = {"stock": w.stock, "candidate_guard_on": w.candidate}
        if cell.family == "out":
            calls["candidate_guard_off"] = w.candidate
        def context(label):
            return (without_alias_guard(w.inputs, w.out)
                    if label == "candidate_guard_off" else nullcontext())
        w.reset()
        expected = w.stock().copy()
        for label, call in calls.items():
            with context(label):
                w.reset()
                if not _equal(call(), expected):
                    raise RuntimeError(f"correctness failed: {cell.name} {label}")
        del expected
        execution = {"path": w.path}
        sweep._check_no_fallback(execution)
        # Only specifically documented stock regimes can qualify as refusals.
        neighbor = cell.family == "argmax" and (cell.shape[0] < 3000 or np.prod(cell.shape) < 9_000_000)
        intentional_stock = w.decision[0] == "stock" and _withdrawn_inplace_add(cell)
        if w.decision[0] != w.path and not (w.decision[0] == "stock" and neighbor) and not intentional_stock:
            raise RuntimeError(f"expected {w.path}, got {w.decision}: {cell.name}")
        for label, call in calls.items():
            with context(label):
                _sample(call, w.reset)  # warm every variant
        estimate = _sample(w.stock, w.reset)
        number = max(1, min(1000, int(0.01 / max(estimate, 1e-9))))
        samples = {label: [] for label in calls}
        labels = list(calls)
        for round_index in range(rounds):
            offset = round_index % len(labels)
            for label in labels[offset:] + labels[:offset]:
                with context(label):
                    elapsed = sum(_sample(calls[label], w.reset) for _ in range(number))
                samples[label].append(elapsed / number)
        sweep._check_no_fallback(execution)
        medians = {label: statistics.median(values) for label, values in samples.items()}
        result = {
            "cell": cell.name, "status": "measured", "family": cell.family,
            "operation": cell.op, "dtype": cell.dtype, "shape": list(cell.shape),
            "alias": cell.alias, "seed": 20260919, "dispatch": list(w.decision),
            "correctness": {"passed": True, "mode": "bit-identical"},
            "rounds": rounds, "calls_per_sample": number,
            "timings_seconds": samples, "median_seconds": medians,
            "ratios_stock_over_candidate": {
                label: medians["stock"] / elapsed for label, elapsed in medians.items()
                if label != "stock"
            },
            "input_reset": "outside each timed call", "result_consumed": True,
            "experimental_enable": w.path,
            "execution": {"path": w.path, "check": "Gearbox._warned_paths",
                          "fallback_observed": execution["dispatch"]["fallback_observed"]},
        }
        if cell.family == "vectorize":
            result["execution"]["direct_ufunc"] = cell.op
        if intentional_stock:
            result["execution"].update(intentional_stock=True, stock_reason="float64-add-exact-inplace")
        if "candidate_guard_off" in medians:
            result["guard_overhead_seconds"] = medians["candidate_guard_on"] - medians["candidate_guard_off"]
        return result


def _conditions(before, after=None):
    return {"quiet_threshold": 0.20, "cpu_busy_before": before, "cpu_busy_after": after,
            "contended": any(x is not None and x > 0.20 for x in (before, after)),
            "load_known": before is not None and after is not None}


def _quiet(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 0.20


def validate_measurement(record, expected_cell=None, *, require_execution=True):
    """Check a saved measurement without changing it or its source attribution.

    Legacy records can have internally consistent timing evidence while lacking
    the explicit execution checks added later. Opting into their inspection
    returns execution_verified=False; it never upgrades them to verified cells.
    """
    def check(condition, reason):
        if not condition:
            raise ValueError(reason)

    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    def agrees(value, expected):
        return (type(value) in (int, float) and math.isfinite(value)
                and math.isclose(value, expected, rel_tol=1e-12, abs_tol=0.0))

    check(isinstance(record, dict), "measurement must be an object")
    check(record.get("status") == "measured", "not a measured record")
    family, op = record.get("family"), record.get("operation")
    shape = record.get("shape")
    check(family in ("argmax", "vectorize", "out"), "unknown family")
    check(isinstance(shape, list) and shape and all(type(n) is int and n > 0 for n in shape), "invalid shape")
    check(isinstance(op, str) and isinstance(record.get("dtype"), str)
          and isinstance(record.get("alias"), str), "missing cell axes")
    cell = Cell(family, op, record["dtype"], tuple(shape), record["alias"])
    check(record.get("cell") == cell.name and (expected_cell is None or cell.name == expected_cell), "cell identity mismatch")
    path = {"argmax": "argmax_blocked_transpose", "vectorize": "vectorize_ufunc_direct",
            "out": f"pyrallel_{op}"}[family]
    check(record.get("experimental_enable") == path, "experimental path mismatch")
    decision = record.get("dispatch")
    neighbor = family == "argmax" and (shape[0] < 3000 or math.prod(shape) < 9_000_000)
    check(isinstance(decision, list) and len(decision) == 2 and all(isinstance(x, str) for x in decision), "invalid dispatch evidence")
    intentional_stock = decision[0] == "stock" and _withdrawn_inplace_add(cell)
    check(decision[0] == path or (neighbor and decision[0] == "stock") or intentional_stock,
          "unexpected dispatch path")
    correctness = record.get("correctness")
    check(isinstance(correctness, dict) and correctness.get("passed") is True
          and correctness.get("mode") == "bit-identical", "missing or failed correctness verification")
    fingerprint = record.get("fingerprint")
    check(isinstance(fingerprint, dict) and all(isinstance(fingerprint.get(k), str) and fingerprint[k]
          for k in ("fingerprint", "numpy", "python")), "missing machine fingerprint")
    conditions = record.get("conditions")
    check(isinstance(conditions, dict) and all(_quiet(conditions.get(k)) for k in ("cpu_busy_before", "cpu_busy_after"))
          and conditions.get("load_known") is True and conditions.get("contended") is False,
          "missing or nonquiet child load evidence")
    rounds, number = record.get("rounds"), record.get("calls_per_sample")
    check(type(rounds) is int and rounds > 0 and type(number) is int and number > 0, "invalid sample counts")
    labels = {"stock", "candidate_guard_on"}
    if family == "out":
        labels.add("candidate_guard_off")
    samples, medians = record.get("timings_seconds"), record.get("median_seconds")
    ratios = record.get("ratios_stock_over_candidate")
    check(isinstance(samples, dict) and set(samples) == labels, "missing raw timing samples")
    check(isinstance(medians, dict) and set(medians) == labels, "missing timing medians")
    check(isinstance(ratios, dict) and set(ratios) == labels - {"stock"}, "missing timing ratios")
    for label in labels:
        values = samples[label]
        check(isinstance(values, list) and len(values) == rounds and all(positive(v) for v in values), "invalid raw timing samples")
        check(agrees(medians[label], statistics.median(values)), "median does not match raw samples")
        sweep.validate_timing_stability(values, label)
    for label in labels - {"stock"}:
        check(agrees(ratios[label], medians["stock"] / medians[label]), "ratio does not match raw samples")
    if family == "out":
        check(agrees(record.get("guard_overhead_seconds"), medians["candidate_guard_on"] - medians["candidate_guard_off"]),
              "guard overhead does not match raw samples")
    stderr = record.get("stderr", "")
    check(isinstance(stderr, str), "invalid child stderr")
    check("falling back to stock" not in stderr.lower(), "child stderr reports fallback")
    execution = record.get("execution")
    if intentional_stock:
        check(isinstance(execution, dict) and execution.get("intentional_stock") is True
              and execution.get("stock_reason") == "float64-add-exact-inplace",
              "missing intentional stock execution verification")
    elif isinstance(execution, dict):
        check(execution.get("intentional_stock", False) is False,
              "intentional stock contradicts dispatch evidence")
    if execution is None and not require_execution:
        return {"timings_verified": True, "execution_verified": False,
                "stderr_fallback_observed": False}
    check(isinstance(execution, dict) and execution.get("path") == path
          and execution.get("check") == "Gearbox._warned_paths"
          and execution.get("fallback_observed") is False, "missing or failed execution verification")
    if family == "vectorize":
        check(execution.get("direct_ufunc") == op, "missing direct ufunc execution verification")
    return {"timings_verified": True, "execution_verified": True,
            "stderr_fallback_observed": False}


def _unqualify(record, reason_code, reason):
    record.update(status="unverified", qualified=False,
                  reason_code=reason_code, reason=reason)
    # Preserve raw samples for diagnosis without presenting their ratios as
    # usable speed evidence. The parent only prints qualified child ratios.
    for key in ("ratios_stock_over_candidate", "guard_overhead_seconds"):
        if key in record:
            record[f"unqualified_{key}"] = record.pop(key)


def _invalid_json_constant(value):
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _child(args):
    record = {"cell": args.one, "status": "error", "qualified": False,
              "fingerprint": sweep._fingerprint()}
    before = sweep._foreign_load()
    code = 1
    try:
        if args.require_quiet and not _quiet(before):
            _unqualify(record, "quiet-before", f"quiet child refused: load={before!r}")
        else:
            core_us = cpuclass.probe_us() if args.fast_under is not None else None
            record["core_probe_us"] = core_us
            if core_us is not None and core_us > args.fast_under:
                record["status"] = "slow-core"
            else:
                record.update(measure(cells()[args.one], args.rounds))
                for label, values in record["timings_seconds"].items():
                    sweep.validate_timing_stability(values, label)
            code = 0
    except Exception as exc:
        record.update(status="error", error=repr(exc))
    finally:
        after = sweep._foreign_load()
        record["conditions"] = _conditions(before, after)
        # Noise must not mask a correctness or execution exception. Even
        # without --require-quiet, noisy timings are not qualified evidence.
        if record["status"] != "error" and not _quiet(before):
            _unqualify(record, "quiet-before", f"unverified initial load={before!r}")
            code = 1
        elif record["status"] != "error" and not _quiet(after):
            _unqualify(record, "quiet-after", f"unverified final load={after!r}")
            code = 1
        record["qualified"] = record["status"] == "measured" and code == 0
        record["exit_code"] = code
        sweep._write_json(args.json, record)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append", choices=("argmax", "vectorize", "out"))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--cells-file", type=Path,
                        help="exact newline-separated cell IDs; cannot combine with --family/--only")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--require-quiet", action="store_true")
    parser.add_argument("--any-core", action="store_true")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--fast-under", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.retries < 1 or args.timeout <= 0:
        parser.error("rounds, retries and timeout must be positive")
    selected = [name for name, c in cells().items()
                if (not args.family or c.family in args.family)
                and (not args.only or any(s in name for s in args.only))]
    selection = {}
    if args.cells_file:
        if args.family or args.only:
            parser.error("--cells-file cannot combine with --family or --only")
        try:
            content = args.cells_file.read_bytes()
            selected = [line.strip() for line in content.decode("utf-8-sig").splitlines() if line.strip()]
        except (OSError, UnicodeError) as exc:
            parser.error(f"cannot read --cells-file: {exc}")
        if len(selected) != len(set(selected)):
            parser.error("--cells-file contains duplicate cell IDs")
        unknown = set(selected) - cells().keys()
        if unknown:
            parser.error(f"unknown exact cell IDs: {sorted(unknown)}")
        selection["cells_file"] = {"path": str(args.cells_file),
                                   "sha256": hashlib.sha256(content).hexdigest()}
    selection["requested_cells"] = selected
    selection["requested_cells_sha256"] = hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if args.list:
        print("\n".join(selected))
        return 0
    if args.json is None:
        parser.error("--json is required for measurements")
    if args.one:
        return _child(args)
    before = sweep._foreign_load()
    record = {
        "schema_version": 1, "tool": "bench_review_followup",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": sweep._fingerprint(), "source_revision": sweep._source_revision(),
        "source": sweep._source_manifest(),
        "benchmark_source": {
            "path": Path(__file__).resolve().relative_to(REPO).as_posix(),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "selected_cells": selected, "selection": selection,
        "cells": [], "completed": False, "verified_cells": [],
        "settings": {"rounds": args.rounds, "retries": args.retries,
                     "any_core": args.any_core, "require_quiet": args.require_quiet,
                     "calibration": "pristine shipped tables in fresh children",
                     "environment": {key: os.environ.get(key) for key in (
                         "PYOVERDRIVE_THREADS", "PYOVERDRIVE_DISABLE", "OMP_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONWARNINGS")}},
    }
    unresolved = {name: {"cell": name, "reason_code": "not-attempted",
                         "reason": "cell has not run"} for name in selected}
    attempted = set()
    code = 1
    try:
        if args.require_quiet and not _quiet(before):
            raise RuntimeError(f"quiet run refused: load={before!r}")
        if not selected:
            raise RuntimeError("no cells selected")
        classes = None if args.any_core else cpuclass.classify()
        cutoff = None if args.any_core else cpuclass.fast_cutoff(classes)
        record["cpu_classes"] = classes
        record["fast_under_us"] = cutoff
        with tempfile.TemporaryDirectory(prefix="pyoverdrive-followup-") as scratch:
            env = dict(os.environ, PYOVERDRIVE_CALIBRATION=str(Path(scratch) / "absent.json"))
            env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
            for name in selected:
                attempted.add(name)
                for attempt in range(1, args.retries + 1):
                    destination = Path(scratch) / "cell.json"
                    destination.unlink(missing_ok=True)
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", name,
                           "--json", str(destination), "--rounds", str(args.rounds)]
                    if args.require_quiet:
                        cmd.append("--require-quiet")
                    if cutoff is not None:
                        cmd += ["--fast-under", str(cutoff)]
                    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                                          text=True, timeout=args.timeout)
                    child = (json.loads(destination.read_text(), parse_constant=_invalid_json_constant)
                             if destination.exists() else {})
                    child.update(attempt=attempt, returncode=proc.returncode, stderr=proc.stderr)
                    record["cells"].append(child)
                    quiet_refusal = (child.get("status") == "unverified"
                                     and child.get("reason_code") in ("quiet-before", "quiet-after")
                                     and not child.get("error") and proc.returncode in (0, 1))
                    if child.get("cell") != name or (proc.returncode and not quiet_refusal) or child.get("error"):
                        raise RuntimeError(f"{name}: {child.get('error', 'child failed')} {proc.stderr}")
                    if quiet_refusal:
                        _unqualify(child, child["reason_code"], child.get("reason", "quiet conditions not met"))
                        break
                    if child.get("status") != "slow-core":
                        break
                if child.get("status") == "slow-core":
                    _unqualify(child, "slow-core-exhausted", "no fast-core draw within retry limit")
                elif child.get("status") == "measured":
                    if not child.get("correctness", {}).get("passed"):
                        raise RuntimeError(f"{name}: missing or failed correctness verification")
                    conditions = child.get("conditions", {})
                    if not all(key in conditions for key in ("cpu_busy_before", "cpu_busy_after")):
                        raise RuntimeError(f"{name}: missing child load evidence")
                    if not _quiet(conditions["cpu_busy_before"]) or not _quiet(conditions["cpu_busy_after"]):
                        _unqualify(child, "quiet-after", "child load evidence does not qualify")
                if child.get("status") == "unverified":
                    unresolved[name] = {"cell": name, "reason_code": child["reason_code"], "reason": child["reason"]}
                    print(f"UNVERIFIED {name}: {child['reason']}", flush=True)
                    continue
                if child.get("status") != "measured":
                    raise RuntimeError(f"{name}: unknown child status {child.get('status')!r}")
                child["qualified"] = False
                validate_measurement(child, name)
                child["qualified"] = True
                record["verified_cells"].append(name)
                unresolved.pop(name)
                print(f"{name}: {child['ratios_stock_over_candidate']}", flush=True)
        record["completed"] = not unresolved
        code = int(bool(unresolved))
    except Exception as exc:
        record["error"] = repr(exc)
        for name, pending in unresolved.items():
            if pending["reason_code"] == "not-attempted":
                pending.update(
                    reason_code="execution-error" if name in attempted else "not-attempted",
                    reason=f"run stopped: {exc}",
                )
        print(f"NOT verified: {exc}", file=sys.stderr)
    finally:
        after = sweep._foreign_load()
        record["conditions"] = _conditions(before, after)
        record["unverified_cells"] = list(unresolved.values())
        record["all_cells_attempted"] = len(attempted) == len(selected)
        if args.require_quiet and not _quiet(after):
            record["batch_unverified_reason"] = "final parent load does not meet quiet conditions; individually qualified cells are retained"
            code = 1
        record["exit_code"] = code
        sweep._write_json(args.json, record)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
