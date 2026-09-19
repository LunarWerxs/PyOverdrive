"""Explicit sampling budgets propagate without running real benchmarks."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from tools import verify_no_pessimization as sweep


def test_sampling_defaults_remain_nine_rounds_and_twenty_milliseconds(monkeypatch):
    seen = []
    monkeypatch.setattr(sweep, "_run_with_evidence", lambda args: seen.append(args) or 0)
    assert sweep.main(["tool"]) == 0
    assert (seen[0].rounds, seen[0].sample_seconds) == (9, 0.02)


@pytest.mark.parametrize("flag,value", [("--rounds", "0"), ("--rounds", "-1"),
    ("--sample-seconds", "0"), ("--sample-seconds", "-0.1"),
    ("--sample-seconds", "nan"), ("--sample-seconds", "inf")])
def test_invalid_sampling_budget_refused_before_measurement(flag, value, monkeypatch):
    monkeypatch.setattr(sweep, "_run_with_evidence", lambda args: pytest.fail("invalid budget reached measurement"))
    with pytest.raises(SystemExit) as exc:
        sweep.main(["tool", flag, value])
    assert exc.value.code == 2


def test_sampling_budget_forwarded_to_every_child_retry(monkeypatch):
    commands = []
    def run(cmd, **kwargs):
        commands.append(cmd)
        result = "SLOW-CORE" if len(commands) == 1 else "RATIO 2.0000 numpy.sin"
        return subprocess.CompletedProcess(cmd, 0, result, "")
    monkeypatch.setattr(sweep.subprocess, "run", run)
    args = SimpleNamespace(rounds=31, sample_seconds=0.2, retries=2)
    sweep._measure_one_cell("sample", args, 100, phase="confirmation")
    assert len(commands) == 2
    for cmd in commands:
        assert cmd[cmd.index("--rounds") + 1] == "31"
        assert cmd[cmd.index("--sample-seconds") + 1] == "0.2"


def test_child_forwards_requested_sampling_budget(monkeypatch):
    received = []
    def measure(cell, cutoff, **kwargs):
        received.append(kwargs)
        return "SKIP no-input"
    monkeypatch.setattr(sweep, "_measure_one", measure)
    assert sweep.main(["tool", "--one", "sample", "--rounds", "31", "--sample-seconds", "0.2"]) == 0
    assert received[0]["rounds"] == 31
    assert received[0]["sample_seconds"] == 0.2


def test_sampling_settings_are_in_parent_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "_foreign_load", lambda: 0.01)
    monkeypatch.setattr(sweep, "_fingerprint", lambda: {})
    monkeypatch.setattr(sweep, "_source_manifest", lambda: {})
    monkeypatch.setattr(sweep, "_source_revision", lambda: None)
    monkeypatch.setattr(sweep, "_run_sweep", lambda args: 0)
    output = tmp_path / "record.json"
    assert sweep.main(["tool", "--json", str(output), "--rounds", "31", "--sample-seconds", "0.2"]) == 0
    settings = json.loads(output.read_text())["settings"]
    assert (settings["rounds"], settings["sample_seconds"]) == (31, 0.2)


@pytest.mark.parametrize("sample_seconds,expected_iterations", [(None, 1), (0.2, 5)])
def test_sampling_budget_controls_iterations_and_retains_raw_rounds(monkeypatch, sample_seconds, expected_iterations):
    stock = lambda: sweep.np.array([1.0])
    monkeypatch.setattr(sweep, "_resolve", lambda op: (SimpleNamespace(op=stock), "op"))
    monkeypatch.setattr(sweep.GEARBOX, "stock_fn", lambda op: stock)
    iterations = []
    def timeit(fn, number):
        iterations.append(number)
        return 0.04 * number
    monkeypatch.setattr(sweep.timeit, "timeit", timeit)
    evidence = {}
    kwargs = {} if sample_seconds is None else {"sample_seconds": sample_seconds}
    assert sweep._measure("numpy.op", (), {}, 31, evidence=evidence, **kwargs) == 1.0
    assert iterations == [1] + [expected_iterations] * 62
    assert evidence["rounds"] == 31
    assert evidence["sample_seconds"] == (0.02 if sample_seconds is None else sample_seconds)
    assert evidence["stock_seconds"] == [0.04] * 31
    assert evidence["patched_seconds"] == [0.04] * 31
