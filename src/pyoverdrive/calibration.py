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

    def __init__(self, name: str, probe: Callable[[], dict], apply_floors: Callable[[dict], None]):
        self.name = name
        self.probe = probe
        self.apply_floors = apply_floors


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


_GATED: dict[str, _Gate] = {
    "argmax_blocked_transpose": _Gate(
        "argmax_blocked_transpose", _probe_argmax_blocked, _apply_argmax_floors
    ),
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
