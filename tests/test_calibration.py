"""The calibration layer must fail safe in every direction: no file means
gated paths stay off, a foreign machine's file is ignored, and only a
matching-fingerprint file with enabled=true turns a gated path on."""

from __future__ import annotations

import json

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive import calibration
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths import argmax_blocked

PATH = "argmax_blocked_transpose"


@pytest.fixture(autouse=True)
def _isolated_calibration(tmp_path, monkeypatch):
    monkeypatch.setenv("PYOVERDRIVE_CALIBRATION", str(tmp_path / "calibration.json"))
    calibration.load(refresh=True)
    saved_rows, saved_size = argmax_blocked.ROWS_MIN, argmax_blocked.SIZE_MIN
    # the REAL machine's calibration file may have enabled the gated path
    # at import time (it does on the Intel box); tests must start from the
    # registered default (disabled) and restore whatever was live after
    was_enabled = any(
        p.enabled for p in GEARBOX._paths["numpy.argmax"] if p.name == PATH
    )
    pyoverdrive.disable_path(PATH)
    yield
    argmax_blocked.ROWS_MIN, argmax_blocked.SIZE_MIN = saved_rows, saved_size
    GEARBOX.set_path_enabled(PATH, was_enabled)
    calibration.load(refresh=True)


def _dispatching_args():
    rng = np.random.default_rng(0)
    a = rng.random(size=(argmax_blocked.ROWS_MIN, 3_000))
    assert a.size >= argmax_blocked.SIZE_MIN
    return (a,), {"axis": 0}


def test_no_file_means_gated_path_stays_off():
    assert calibration.load(refresh=True) == {}
    assert calibration.apply(GEARBOX) == []
    args, kwargs = _dispatching_args()
    assert GEARBOX.decide("numpy.argmax", args, kwargs)[0] == "stock"


def test_registered_default_is_disabled():
    # the FastPath itself ships enabled=False; only calibration or an
    # explicit enable_path may flip it
    paths = [p for p in GEARBOX._paths["numpy.argmax"] if p.name == PATH]
    assert len(paths) == 1
    assert paths[0].provenance.get("calibration_gated") is True


def test_matching_file_enables_and_sets_floors():
    calibration.save({PATH: {"enabled": True, "rows_min": 4_000, "size_min": 12_000_000}})
    enabled = calibration.apply(GEARBOX)
    assert enabled == [PATH]
    assert argmax_blocked.ROWS_MIN == 4_000
    assert argmax_blocked.SIZE_MIN == 12_000_000
    rng = np.random.default_rng(1)
    a = rng.random(size=(4_000, 3_000))
    assert GEARBOX.decide("numpy.argmax", (a,), {"axis": 0})[0] == PATH
    small = rng.random(size=(3_500, 3_000))  # above old floor, below stored one
    assert GEARBOX.decide("numpy.argmax", (small,), {"axis": 0})[0] == "stock"


def test_foreign_fingerprint_file_is_ignored():
    p = calibration.save({PATH: {"enabled": True}})
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["machine"]["fingerprint"] = "deadbeef0000"
    p.write_text(json.dumps(raw), encoding="utf-8")
    assert calibration.load(refresh=True) == {}
    assert calibration.apply(GEARBOX) == []


def test_disabled_verdict_keeps_path_off():
    calibration.save({PATH: {"enabled": False, "cells": {"3000x3000": 0.82}}})
    assert calibration.apply(GEARBOX) == []
    args, kwargs = _dispatching_args()
    assert GEARBOX.decide("numpy.argmax", args, kwargs)[0] == "stock"


def test_corrupt_file_is_ignored():
    p = calibration.calibration_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert calibration.load(refresh=True) == {}


def test_unknown_path_entry_is_ignored():
    calibration.save({"some_future_path": {"enabled": True}})
    assert calibration.apply(GEARBOX) == []


def test_calibrate_end_to_end_with_stubbed_probe(monkeypatch):
    # force an enabling verdict without paying the real probe's runtime
    monkeypatch.setitem(
        calibration._GATED,
        PATH,
        calibration._Gate(
            PATH,
            lambda: {"enabled": True, "rows_min": 3_000, "size_min": 9_000_000,
                     "cells": {"stub": 9.99}},
            calibration._GATED[PATH].apply_floors,
        ),
    )
    import io

    buf = io.StringIO()
    results = pyoverdrive.calibrate(verbose=True, file=buf)
    assert results[PATH]["enabled"] is True
    assert "ENABLED" in buf.getvalue()
    args, kwargs = _dispatching_args()
    assert GEARBOX.decide("numpy.argmax", args, kwargs)[0] == PATH
    # and the verdict persisted: a fresh load sees it
    assert calibration.load(refresh=True)[PATH]["enabled"] is True


def test_calibrate_refuses_while_patched():
    pyoverdrive.enable(["numpy.roll"])
    try:
        with pytest.raises(RuntimeError):
            pyoverdrive.calibrate(verbose=False)
    finally:
        pyoverdrive.disable()


# --- pyrallel: an ALWAYS-ON gate that calibration can only narrow ----------
#
# The threaded tables ship enabled and were derived on one machine, so the
# calibration entry for them is the opposite of argmax's: it never turns a
# path on, it only removes rows that do not pay on the host. These check the
# directions it must fail in.

from pyoverdrive.fastpaths import parallel_binary, parallel_ufunc  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_pyrallel_tables():
    saved = {
        mod: {op: dict(row) for op, row in mod.SUPPORTED.items()}
        for mod in (parallel_ufunc, parallel_binary)
    }
    yield
    for mod, tables in saved.items():
        for op, row in tables.items():
            mod.SUPPORTED[op].clear()
            mod.SUPPORTED[op].update(row)


def test_shipped_snapshot_matches_the_live_table_before_any_calibration():
    # SHIPPED is the pristine copy the probe measures against; if it ever
    # drifts from SUPPORTED at import, a dropped row could never be re-probed
    for mod in (parallel_ufunc, parallel_binary):
        assert mod.SHIPPED == mod.SUPPORTED
        assert mod.SHIPPED is not mod.SUPPORTED
        for op in mod.SHIPPED:
            assert mod.SHIPPED[op] is not mod.SUPPORTED[op]


def test_drop_removes_only_the_named_dtype():
    f32 = np.dtype(np.float32)
    assert f32 in parallel_ufunc.SUPPORTED["sin"]
    calibration._apply_pyrallel({"drop": {"pyrallel_sin": ["float32"]}})
    assert f32 not in parallel_ufunc.SUPPORTED["sin"]
    assert np.dtype(np.float64) in parallel_ufunc.SUPPORTED["sin"]
    # untouched ops keep every row
    assert parallel_ufunc.SUPPORTED["tanh"] == parallel_ufunc.SHIPPED["tanh"]


def test_apply_is_rebuilt_from_shipped_not_cumulative():
    calibration._apply_pyrallel({"drop": {"pyrallel_sin": ["float32"]}})
    assert np.dtype(np.float32) not in parallel_ufunc.SUPPORTED["sin"]
    # a later calibration that drops nothing must RESTORE the row, not leave
    # the previous verdict in place
    calibration._apply_pyrallel({"drop": {}})
    assert parallel_ufunc.SUPPORTED["sin"] == parallel_ufunc.SHIPPED["sin"]


def test_dropped_row_stops_dispatching_but_the_other_dtype_still_does():
    pyoverdrive.enable(["numpy.sin"])
    try:
        n32 = parallel_ufunc.SHIPPED["sin"][np.dtype(np.float32)]
        n64 = parallel_ufunc.SHIPPED["sin"][np.dtype(np.float64)]
        x32 = np.linspace(0.0, 6.0, n32, dtype=np.float32)
        x64 = np.linspace(0.0, 6.0, n64, dtype=np.float64)
        assert GEARBOX.decide("numpy.sin", (x32,), {})[0] == "pyrallel_sin"
        calibration._apply_pyrallel({"drop": {"pyrallel_sin": ["float32"]}})
        assert GEARBOX.decide("numpy.sin", (x32,), {})[0] == "stock"
        assert GEARBOX.decide("numpy.sin", (x64,), {})[0] == "pyrallel_sin"
    finally:
        pyoverdrive.disable()


def test_apply_reads_the_stored_entry_for_an_always_on_gate(tmp_path):
    calibration.save({"pyrallel": {"drop": {"pyrallel_tanh": ["float64"]}}})
    calibration.apply(GEARBOX)
    assert np.dtype(np.float64) not in parallel_ufunc.SUPPORTED["tanh"]


def test_probe_cell_reports_a_ratio_for_a_shipped_row():
    dtype, floor = next(iter(parallel_ufunc.SHIPPED["sin"].items()))
    got = calibration.probe_cell(f"unary:sin:{np.dtype(dtype).name}:{floor}")
    assert "ratio" in got and got["ratio"] > 0
    pyoverdrive.disable()


def test_probe_cell_declines_a_slow_core_draw():
    # fast_under=0 makes every draw "slow", which is how the parent learns
    # to re-spawn rather than trusting a coin-flipped baseline
    got = calibration.probe_cell("unary:sin:float64:300000", fast_under=0.0)
    assert got == {"slow_core": True}


# --- the CPU classifier must not invent a core class ------------------------

from pyoverdrive import _cpuclass  # noqa: E402


def test_classifier_calls_a_uniform_machine_uniform():
    times = {c: 100.0 + (c % 3) for c in range(32)}  # ordinary jitter only
    got = _cpuclass.classify(times)
    assert got["hybrid"] is False
    assert got["outlier_cpus"] == []


def test_classifier_finds_a_real_hybrid_split():
    # the Intel reference box: 16 P-core threads at ~136us, 4 E-cores at ~325
    times = {c: (136.0 if c < 16 else 325.0) for c in range(20)}
    got = _cpuclass.classify(times)
    assert got["hybrid"] is True
    assert got["slow"] == [16, 17, 18, 19]
    assert got["class_ratio"] > 2.0


@pytest.mark.parametrize("n_busy", [1, 2])
def test_a_couple_of_busy_cpus_are_outliers_not_a_core_class(n_busy):
    # what the AMD box actually produced while another session had work
    # pinned to it: a real gap, but on far too few CPUs to be a design
    times = {c: (200.0 if c < n_busy else 136.0) for c in range(32)}
    got = _cpuclass.classify(times)
    assert got["hybrid"] is False, "contention must not be reported as hardware"
    assert got["outlier_cpus"] == list(range(n_busy))
    assert "contended" in _cpuclass.describe(got)


def test_fast_cutoff_is_none_when_there_is_no_split():
    assert _cpuclass.fast_cutoff({"hybrid": False, "fast": [0, 1]}) is None
