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
