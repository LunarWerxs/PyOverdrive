"""Unit tests for diagnostics: status/report/configure/selfcheck and the CLI."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive import diagnostics
from pyoverdrive.parallel import pyrallel

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_everything():
    yield
    pyrallel.set_max_threads(None)
    pyoverdrive.enable_path("unique_sort")
    pyoverdrive.set_debug(False)
    pyoverdrive.disable()
    pyrallel.shutdown()


def test_status_keys_and_fast_paths():
    s = pyoverdrive.status()
    assert set(s) == {
        "version",
        "numpy",
        "python",
        "activated",
        "patched_operations",
        "fast_paths",
        "pyrallel",
        "debug",
    }
    names = {p["name"] for p in s["fast_paths"]}
    assert {"unique_sort", "inner_tensordot", "intersect_sorted"} <= names
    assert any(n.startswith("pyrallel_") for n in names)


def test_activated_toggle():
    assert pyoverdrive.status()["activated"] is False
    pyoverdrive.enable(["numpy.unique"])
    s = pyoverdrive.status()
    assert s["activated"] is True
    assert s["patched_operations"] == ["numpy.unique"]
    pyoverdrive.disable()
    assert pyoverdrive.status()["activated"] is False


def test_report_writes_to_file_like_and_mentions_version_and_paths():
    buf = io.StringIO()
    pyoverdrive.report(buf)
    out = buf.getvalue()
    assert pyoverdrive.__version__ in out
    for p in pyoverdrive.status()["fast_paths"]:
        assert p["name"] in out


def test_configure_enable_disable_path():
    s = pyoverdrive.configure(disable=["unique_sort"])
    entry = next(p for p in s["fast_paths"] if p["name"] == "unique_sort")
    assert entry["enabled"] is False

    s = pyoverdrive.configure(enable=["unique_sort"])
    entry = next(p for p in s["fast_paths"] if p["name"] == "unique_sort")
    assert entry["enabled"] is True


def test_configure_threads():
    s = pyoverdrive.configure(threads=1)
    assert s["pyrallel"]["available"] is False
    assert s["pyrallel"]["max_threads"] == 1

    s2 = pyoverdrive.configure(threads=None)
    assert s2["pyrallel"]["max_threads"] == 1  # unchanged by threads=None

    pyrallel.set_max_threads(None)
    assert pyrallel.max_threads() >= 1
    assert pyrallel.available() == (pyrallel.max_threads() >= 2)


def test_configure_threads_zero_raises():
    with pytest.raises(ValueError):
        pyoverdrive.configure(threads=0)


def test_configure_debug():
    s = pyoverdrive.configure(debug=True)
    assert s["debug"] is True
    pyoverdrive.configure(debug=False)
    assert pyoverdrive.status()["debug"] is False


def test_set_max_threads_recreates_pool():
    pyrallel.parallel_unary(np.sin, np.linspace(0, 1, 300_000), 4)
    size_before = pyrallel.pool_size()
    assert size_before > 0

    pyrallel.set_max_threads(2)
    assert pyrallel.pool_size() == 0

    pyrallel.parallel_unary(np.sin, np.linspace(0, 1, 300_000), 4)
    assert pyrallel.pool_size() == 2


def test_selfcheck_all_pass_or_skip():
    results = pyoverdrive.selfcheck(verbose=False)
    for name, verdict in results.items():
        assert verdict == "PASS" or verdict.startswith("SKIP"), (name, verdict)
    assert results["unique_values_sort"] == "PASS"
    assert results["noop_positive"].startswith("SKIP")


def test_selfcheck_restores_activation_state_when_not_enabled():
    assert not pyoverdrive.enabled()
    pyoverdrive.selfcheck(verbose=False)
    assert not pyoverdrive.enabled()


def test_selfcheck_restores_activation_state_when_enabled():
    pyoverdrive.enable(["numpy.unique"])
    pyoverdrive.selfcheck(verbose=False)
    s = pyoverdrive.status()
    assert s["patched_operations"] == ["numpy.unique"]
    assert getattr(np.unique, "__pyoverdrive__", False)


def test_selfcheck_reports_fail_when_path_does_not_dispatch(monkeypatch):
    def _bad_inputs():
        # float64 is not in unique_sort's supported dtype set, so this input
        # will not dispatch to it.
        return {"unique_sort": lambda: ((np.array([1.0, 2.0], dtype=np.float64),), {})}

    monkeypatch.setattr(diagnostics, "_selfcheck_inputs", _bad_inputs)
    results = pyoverdrive.selfcheck(verbose=False)
    assert results["unique_sort"].startswith("FAIL: did not dispatch")
    for name, verdict in results.items():
        if name != "unique_sort":
            assert verdict.startswith("SKIP: no selfcheck input registered") or verdict.startswith("SKIP: disabled")


def test_cli_default_report():
    proc = subprocess.run(
        [sys.executable, "-m", "pyoverdrive"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "fast paths:" in proc.stdout


def test_cli_json():
    proc = subprocess.run(
        [sys.executable, "-m", "pyoverdrive", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "version" in data


def test_cli_selfcheck():
    proc = subprocess.run(
        [sys.executable, "-m", "pyoverdrive", "--selfcheck"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "0 failing" in proc.stdout
