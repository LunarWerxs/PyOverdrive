"""Per-machine calibration: measured-on-THIS-box gates for fast paths whose
regime does not transfer across architectures.

Why this exists (the OPP-000034 lesson): a blocked-transpose argmax route
won 2.2-4x on Intel Alder Lake at every in-regime cell and was a
0.75-0.84x REGRESSION on AMD Zen 4 at the same sizes, because Zen 4's
stock strided argmax is ~2.3x faster than Alder Lake's. A size-only
predicate cannot ship that. The rule this module implements:

    A calibration-gated path is registered DISABLED. It turns on only
    when a probe suite has measured it winning on the machine it is
    running on, and the result is stored per machine fingerprint, so a
    calibration file copied to (or left behind for) different hardware
    is ignored rather than trusted.

Usage:

    python -m pyoverdrive --calibrate     # probe + persist + apply
    pyoverdrive.calibrate()               # the same, programmatically

The calibration file lives at ``~/.pyoverdrive/calibration.json``
(override with ``PYOVERDRIVE_CALIBRATION``). It is applied automatically
at import: paths it enables are enabled, floors it tightens are
tightened. With no file (or a file from a different machine/stack),
every calibration-gated path simply stays off - stock NumPy behavior,
zero risk.

Probe discipline: a probe measures the path's own regime EDGE cells (the
weakest cells its predicate admits) against stock, and enables only if
every probed cell clears MIN_WIN. Interior cells only get faster, so a
machine that wins the edges wins the regime. Probes run in-process on
unpatched stock functions and take a few seconds total.

TWO KINDS OF GATE LIVE HERE.

A NORMAL gate ships DISABLED and calibration is what turns it on
(`argmax_blocked_transpose`). Its probe runs in-process, which is sound
only because both sides are single-threaded: they run on whatever core
this process was given and the core's speed cancels out of the ratio.

An ALWAYS-ON gate ships ENABLED and calibration can only take rows AWAY
from it (`pyrallel`, the threaded ufunc families). That cancellation does
NOT hold for it - a threaded candidate spans cores while its baseline sits
on one, so on a hybrid CPU the ratio inherits a per-process coin flip: the
same single-threaded np.sin baseline came back 344 us in 15 of 25 fresh
processes and 497 us in the other 10, 1.44x apart. Its probe therefore
runs each cell in a FRESH SUBPROCESS and re-draws until it lands on a fast
core, and it requires two agreeing readings before removing anything.

So: DO NOT ADD A THREADED CANDIDATE TO AN IN-PROCESS PROBE. Model it on
the pyrallel gate instead. See docs/research/hybrid-cpu-baseline-coin-flip.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

MIN_WIN = 1.3  # same threshold the Dyno batteries hold shipped floors to
_SCHEMA_VERSION = 1


def _machine_identity() -> dict:
    """The same identity-bearing fields lab/dyno/fingerprint.py hashes, so
    calibration files and committed Dyno evidence share fingerprints."""
    import numpy as np

    blas = None
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {}).get("name")
    except Exception:
        pass
    info = {
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "logical_cores": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "blas": blas,
    }
    digest = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:12]
    info["fingerprint"] = digest
    return info


def calibration_path() -> Path:
    override = os.environ.get("PYOVERDRIVE_CALIBRATION", "")
    if override:
        return Path(override)
    return Path.home() / ".pyoverdrive" / "calibration.json"


_cache: dict | None = None


def load(refresh: bool = False) -> dict:
    """The stored per-path table for THIS machine, or {} (missing file,
    unreadable file, schema mismatch, or a fingerprint from other
    hardware/stack - stale calibration must never be trusted)."""
    global _cache
    if _cache is not None and not refresh:
        return _cache
    _cache = {}
    p = calibration_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _cache
    if raw.get("version") != _SCHEMA_VERSION:
        return _cache
    if raw.get("machine", {}).get("fingerprint") != _machine_identity()["fingerprint"]:
        return _cache
    paths = raw.get("paths")
    if isinstance(paths, dict):
        _cache = paths
    return _cache


def save(paths: dict, probes: dict | None = None) -> Path:
    p = calibration_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _SCHEMA_VERSION,
        "machine": _machine_identity(),
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paths": paths,
        "probes": probes or {},
    }
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    load(refresh=True)
    return p


def apply(gearbox) -> list[str]:
    """Apply the stored table to the live registry: enable the
    calibration-gated paths it vouches for and push any stored floors
    into their modules. Returns the names it enabled."""
    enabled = []
    table = load()
    for name, entry in table.items():
        if not isinstance(entry, dict):
            continue
        gate = _GATED.get(name)
        if gate is None:
            continue  # unknown path (older/newer install): ignore
        if gate.always_on:
            # Nothing to enable - these are on already. The stored entry can
            # only narrow the shipped table, so apply it and move on.
            gate.apply_floors(entry)
            continue
        if entry.get("enabled"):
            gate.apply_floors(entry)
            try:
                gearbox.set_path_enabled(name, True)
            except KeyError:
                continue
            enabled.append(name)
    return enabled


def _timeit(fn: Callable[[], Any], budget_s: float = 0.6, min_reps: int = 3) -> float:
    fn()  # warm
    best = float("inf")
    t_end = time.perf_counter() + budget_s
    reps = 0
    while reps < min_reps or time.perf_counter() < t_end:
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
        reps += 1
    return best


class _Gate:
    """One calibration-gated path: its probe cells and floor plumbing."""

    def __init__(self, name: str, probe: Callable[[], dict],
                 apply_floors: Callable[[dict], None],
                 always_on: bool = False):
        self.name = name
        self.probe = probe
        self.apply_floors = apply_floors
        # always_on gates ship ENABLED; calibration only narrows them. A
        # normal gate ships disabled and calibration is what turns it on.
        self.always_on = always_on


def _probe_argmax_blocked() -> dict:
    """Regime-edge cells for argmax_blocked_transpose: the (3000, 3000)
    floor corner and the thin-cols (10000, 1000) edge. Enable only if
    both clear MIN_WIN - Intel Alder Lake measured 2.2x/2.2x here, Zen 4
    measured 0.82x/0.75x."""
    import numpy as np

    from .fastpaths import argmax_blocked

    rng = np.random.default_rng(9182)
    cells = {}
    wins = []
    for rows, cols in ((3_000, 3_000), (10_000, 1_000)):
        a = rng.random(size=(rows, cols))
        t_stock = _timeit(lambda: np.argmax(a, axis=0))
        t_cand = _timeit(lambda: argmax_blocked._blocked_argmax_axis0(a, np.argmax))
        ratio = t_stock / t_cand if t_cand > 0 else 0.0
        cells[f"{rows}x{cols}"] = round(ratio, 3)
        wins.append(ratio >= MIN_WIN)
    return {
        "enabled": all(wins),
        "rows_min": argmax_blocked.ROWS_MIN,
        "size_min": argmax_blocked.SIZE_MIN,
        "cells": cells,
    }


def _apply_argmax_floors(entry: dict) -> None:
    from .fastpaths import argmax_blocked

    if isinstance(entry.get("rows_min"), int):
        argmax_blocked.ROWS_MIN = entry["rows_min"]
    if isinstance(entry.get("size_min"), int):
        argmax_blocked.SIZE_MIN = entry["size_min"]


# --------------------------------------------------------------------------
# PyRallel: verify the shipped threaded thresholds on THIS machine
# --------------------------------------------------------------------------
#
# The shipped PyRallel tables were derived on one machine (an idle Intel
# i7-12700K). They are the least transferable numbers in the project - the
# unary family's wins are compute scaling and the binary family's are memory
# bandwidth, and both depend on core count, cache and memory channels. A
# threshold that pays there can fail to pay on a laptop with two cores or a
# server whose single core already saturates its channels.
#
# So this probe re-measures each row AT ITS FLOOR - the weakest cell the
# predicate admits, and the only cell that matters, since the measured
# sweeps are monotone in size above it - and drops rows that do not pay
# here. It never lowers a floor: knowing where a row STARTS paying needs a
# full sweep (tools/calibrate_dispatch.py), not a probe.
#
# TWO THINGS MAKE THIS DIFFERENT FROM EVERY OTHER PROBE HERE.
#
# 1. It runs each cell in a FRESH SUBPROCESS and rejects the ones that drew
#    a slow core. Every other probe compares a single-threaded candidate
#    against a single-threaded baseline, so the core class cancels out of
#    the ratio; that cancellation is what makes an in-process probe sound
#    and it fails the moment one side is threaded. On the reference box the
#    same single-threaded np.sin baseline came back 344 us in 15 of 25 fresh
#    processes and 497 us in the other 10 - a 1.44x coin flip in the
#    denominator, which would enable or refuse a row at random.
# 2. Its bar is DELIBERATELY LOWER than the 1.3x the tables are derived at.
#    A shipped floor clears 1.3x by as little as 0.01x, and a single probe
#    cell carries roughly +/-0.1x of noise, so judging at 1.3x would drop
#    perfectly good rows about half the time - on the very machine they were
#    derived on. The probe's job is to catch a row that does not pay AT ALL
#    here, not to re-adjudicate a margin a careful two-sweep derivation
#    already established.
PYRALLEL_KEEP_MIN = 1.10


def _pyrallel_rows() -> list[tuple[str, str, str, int]]:
    """(family, op, dtype-name, floor) for every row as SHIPPED."""
    import numpy as np

    from .fastpaths import parallel_binary, parallel_ufunc

    rows = []
    for family, mod in (("unary", parallel_ufunc), ("binary", parallel_binary)):
        for op, table in mod.SHIPPED.items():
            for dtype, floor in table.items():
                rows.append((family, op, np.dtype(dtype).name, int(floor)))
    return sorted(rows)


def probe_cell(spec: str, fast_under: float | None = None) -> dict:
    """Measure ONE pyrallel row at its floor, end to end. Runs as its own
    process (``python -m pyoverdrive --probe-cell``); prints JSON."""
    import numpy as np

    from . import _cpuclass
    from . import diagnostics as D
    from .dispatcher.gearbox import GEARBOX

    family, op, dtype_name, n = spec.split(":")
    if fast_under is not None and _cpuclass.probe_us() > fast_under:
        return {"slow_core": True}

    want = {np.dtype(dtype_name): int(n)}
    make = (D._inputs_binary if family == "binary" else D._inputs_ufunc)(op, want)
    args, kwargs = make()

    import pyoverdrive as _pkg

    _pkg.enable()
    name = f"numpy.{op}"
    if GEARBOX.decide(name, args, kwargs)[0] != f"pyrallel_{op}":
        return {"no_dispatch": True}
    stock = GEARBOX.stock_fn(name)
    patched = getattr(np, op)

    def s_fn():
        float(np.asarray(stock(*args, **kwargs).sum()).real)

    def c_fn():
        float(np.asarray(patched(*args, **kwargs).sum()).real)

    for _ in range(5):
        s_fn()
        c_fn()
    st, ct = [], []
    for _ in range(7):  # INTERLEAVED, so drift cannot land on one side
        t0 = time.perf_counter()
        s_fn()
        st.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        c_fn()
        ct.append(time.perf_counter() - t0)
    st.sort()
    ct.sort()
    return {"ratio": round(st[len(st) // 2] / ct[len(ct) // 2], 4)}


def _probe_pyrallel(verbose: bool = True, out=None) -> dict:
    """Every pyrallel row at its floor, each in a re-drawn subprocess."""
    import subprocess

    from . import _cpuclass

    out = out or sys.stdout
    classes = _cpuclass.classify()
    cutoff = _cpuclass.fast_cutoff(classes)
    if verbose:
        print(f"  {_cpuclass.describe(classes)}", file=out)

    def one_reading(spec: str) -> dict:
        got: dict = {}
        for _ in range(6):  # re-draw until this process lands on a fast core
            cmd = [sys.executable, "-m", "pyoverdrive", "--probe-cell", spec]
            if cutoff is not None:
                cmd += ["--fast-under", f"{cutoff:.3f}"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            try:
                got = json.loads((proc.stdout or "").strip().splitlines()[-1])
            except Exception:  # noqa: BLE001
                got = {"error": (proc.stderr or "")[-120:]}
            if "slow_core" not in got:
                break
        return got

    cells: dict[str, list[float]] = {}
    drop: dict[str, list[str]] = {}
    unstable = 0
    for family, op, dtype_name, floor in _pyrallel_rows():
        spec = f"{family}:{op}:{dtype_name}:{floor}"
        # TWO readings, and a row is only dropped if BOTH fail. There is no
        # portable way to ask "is this machine busy?", and a user calibrating
        # while something else runs would otherwise disable paths that are
        # fine - so instead the readings have to AGREE. Disagreement is the
        # symptom of a machine too loaded to judge on, and the answer to it
        # is to keep the shipped row, never to guess.
        readings = [one_reading(spec), one_reading(spec)]
        got = [r["ratio"] for r in readings if r.get("ratio") is not None]
        if len(got) < 2:
            if verbose:
                why = next((k for r in readings for k in r if k != "ratio"), "?")
                print(f"        -  {op} {dtype_name} n={floor:,} "
                      f"(not measured: {why})", file=out)
            continue
        cells[spec] = [round(g, 4) for g in got]
        lo, hi = min(got), max(got)
        if hi / lo > 1.25:
            unstable += 1
            if verbose:
                print(f"  {lo:5.2f}/{hi:<5.2f}x {op} {dtype_name} n={floor:,}"
                      f"   UNSTABLE, keeping (machine too busy to judge)",
                      file=out)
            continue
        if hi < PYRALLEL_KEEP_MIN:  # both readings below the bar
            drop.setdefault(f"pyrallel_{op}", []).append(dtype_name)
        if verbose:
            verdict = "" if hi >= PYRALLEL_KEEP_MIN else "   DROPPED (no win here)"
            print(f"  {lo:5.2f}/{hi:<5.2f}x {op} {dtype_name} n={floor:,}"
                  f"{verdict}", file=out)
    if unstable and verbose:
        print(f"  NOTE: {unstable} row(s) gave inconsistent readings and were "
              f"kept as shipped.\n        Re-run on an idle machine for a "
              f"verdict on those.", file=out)
    return {"cells": cells, "drop": drop, "unstable": unstable}


def _apply_pyrallel(entry: dict) -> None:
    """Rewrite the live SUPPORTED tables from the pristine SHIPPED copy,
    minus whatever this machine measured as not paying. Rebuilt from SHIPPED
    every time so repeated calibrations are not cumulative."""
    import numpy as np

    from .fastpaths import parallel_binary, parallel_ufunc

    drop = entry.get("drop") or {}
    for mod in (parallel_ufunc, parallel_binary):
        for op, shipped in mod.SHIPPED.items():
            gone = set(drop.get(f"pyrallel_{op}", ()))
            live = mod.SUPPORTED.get(op)
            if live is None:
                continue
            live.clear()
            live.update({d: n for d, n in shipped.items()
                         if np.dtype(d).name not in gone})


_GATED: dict[str, _Gate] = {
    "argmax_blocked_transpose": _Gate(
        "argmax_blocked_transpose", _probe_argmax_blocked, _apply_argmax_floors
    ),
    # always_on: these paths ship enabled. Calibration can only take rows
    # AWAY from them on a machine where they do not pay; it never turns them
    # on, so a missing or foreign calibration file leaves the shipped table
    # exactly as it is.
    "pyrallel": _Gate("pyrallel", _probe_pyrallel, _apply_pyrallel,
                      always_on=True),
}


def calibrate(verbose: bool = True, file=None) -> dict:
    """Probe every calibration-gated path on THIS machine, persist the
    verdicts, and apply them to the live registry. Idempotent; safe to
    re-run after a numpy upgrade (the fingerprint changes, so the old
    file is already being ignored)."""
    from .dispatcher.gearbox import GEARBOX

    out = file or sys.stdout
    if GEARBOX.patched:
        raise RuntimeError("run calibrate() before enable(), on stock numpy")
    results = {}
    for name, gate in _GATED.items():
        if gate.always_on:
            # These ship enabled and print their own per-row detail; a
            # verdict here can only NARROW the shipped table.
            if verbose:
                out.write(f"  {name} (shipped on; re-measuring each row at "
                          f"its floor, >= {PYRALLEL_KEEP_MIN:g}x to keep)\n")
            verdict = gate.probe(verbose=verbose, out=out)
            results[name] = verdict
            dropped = sum(len(v) for v in verdict.get("drop", {}).values())
            if verbose:
                out.write(f"  -> {dropped} row(s) dropped on this machine\n")
            continue
        verdict = gate.probe()
        results[name] = verdict
        if verbose:
            cells = ", ".join(f"{k}: {v}x" for k, v in verdict.get("cells", {}).items())
            state = "ENABLED" if verdict["enabled"] else "stays off (no win here)"
            out.write(f"  {name:<28} {state}  [{cells}]\n")
    path = save(results)
    enabled = apply(GEARBOX)
    if verbose:
        out.write(f"calibration written to {path}\n")
        out.write(f"paths enabled on this machine: {enabled or 'none'}\n")
    return results
