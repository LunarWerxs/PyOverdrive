"""Verification failures must not become successful skips (no real timings)."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_tool(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


across = _load_tool("verify_across_numpy")
sweep = _load_tool("verify_no_pessimization")


def _args(**overrides):
    values = dict(min=1.0, sizes=False, shapes=False, values=False, only=[],
                  retries=2, verbose=True)
    values.update(overrides)
    return SimpleNamespace(**values)


def _result(code=0, out="", err=""):
    return subprocess.CompletedProcess([], code, out, err)


def _completed_sweep(ratio=1.2, judged=1):
    verdict = ("PESSIMIZATION: 1 path(s) below 1x" if ratio < 1 else
               "no dispatching path is below 1x")
    return (f"  {ratio:.2f}x sample numpy.sin\n"
            f"judged {judged} dispatching paths in their own processes, "
            "skipped 2 (no canonical input, or it does not dispatch)\n"
            f"{verdict}\n")


@pytest.mark.parametrize("result", [
    _result(1, "", "worker crashed"),
    _result(1, "  1.20x sample numpy.sin\n", "worker crashed"),
    _result(0, "  1.20x sample numpy.sin\n"),
    _result(0, _completed_sweep(judged=2)),
    _result(2, _completed_sweep(), "late failure"),
    _result(1, _completed_sweep(), "late failure"),
    _result(1, _completed_sweep(0.9999)),  # rounded row says 1.00x, footer says loss
    _result(0, "judged 0 dispatching paths in their own processes, skipped 2 "
               "(no canonical input, or it does not dispatch)\n"
               "no dispatching path is below 1x\n"),
])
def test_incomplete_version_sweep_is_not_verified(monkeypatch, tmp_path, result):
    monkeypatch.setattr(across, "_venv_for", lambda *a: (Path("python"), "2.4.5"))
    monkeypatch.setattr(across, "_run", lambda *a, **k: result)
    info, cells, failure = across._run_one_version("2.4.5", tmp_path, _args())
    assert failure is not None
    if result.stderr:
        assert result.stderr in failure


@pytest.mark.parametrize("ratio,code", [(1.2, 0), (0.8, 1)])
def test_completed_sweep_distinguishes_losses_from_crashes(monkeypatch, tmp_path, ratio, code):
    monkeypatch.setattr(across, "_venv_for", lambda *a: (Path("python"), "2.4.5"))
    monkeypatch.setattr(across, "_run", lambda *a, **k: _result(code, _completed_sweep(ratio)))
    info, cells, failure = across._run_one_version("2.4.5", tmp_path, _args())
    assert info == "2.4.5"
    assert cells == {"sample": ratio}
    assert failure is None


def test_partial_version_cells_remain_visible_but_fail_verification(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(across, "_run_one_version",
                        lambda *a: ("2.4.5", {"sample": 1.2}, "child crashed"))
    ran, skipped = across._collect_all(["2.4.5"], tmp_path, _args())
    assert ran == {"2.4.5": {"sample": 1.2}}
    labels, every, losses, interesting = across._compare(ran, _args())
    assert across._report(labels, losses, interesting, ran, skipped, _args()) != 0
    assert "NOT verified" in capsys.readouterr().out


@pytest.mark.parametrize("ran,skipped", [
    ({"2.4.5": {"sample": 1.2}}, [("2.3.5", "no wheel")]),
    ({"2.4.5": {}}, []),
])
def test_missing_or_zero_cell_version_cannot_pass(ran, skipped):
    labels, every, losses, interesting = across._compare(ran, _args())
    assert across._report(labels, losses, interesting, ran, skipped, _args()) != 0


def test_cached_latest_environment_upgrades_numpy(monkeypatch, tmp_path):
    py = tmp_path / "nplatest" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.touch()
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        return _result(out="2.5.2\n")

    monkeypatch.setattr(across, "_run", run)
    assert across._venv_for("latest", tmp_path) == (py, "2.5.2")
    assert "--upgrade" in commands[0]


@pytest.mark.parametrize("output", ["", "RATIO 1.2000 numpy.sin\n", "SLOW-CORE\n"])
def test_crashed_cell_is_never_a_skip(monkeypatch, output):
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: _result(1, output, "worker died"))
    with pytest.raises(RuntimeError, match="worker died"):
        sweep._measure_one_cell("sample", _args(), None)


def test_crashed_confirmation_is_not_a_nonreproducing_loss(monkeypatch):
    results = iter([_result(out="RATIO 0.8000 numpy.sin\n"), _result(1, err="retry died")])
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: next(results))
    with pytest.raises(RuntimeError, match="retry died"):
        sweep._measure_all(["sample"], _args(), None)


@pytest.mark.parametrize("output", ["", "unexpected output", "RATIO nan numpy.sin"])
def test_invalid_successful_child_output_is_not_a_skip(monkeypatch, output):
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: _result(out=output))
    with pytest.raises(RuntimeError):
        sweep._measure_one_cell("sample", _args(), None)


@pytest.mark.parametrize("output", ["SKIP no-dispatch", "SKIP not-shaped", "SLOW-CORE"])
def test_legitimate_cell_skip_is_allowed(monkeypatch, output):
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: _result(out=output))
    losses, judged, skipped = sweep._measure_all(["sample"], _args(), None)
    assert (losses, judged, skipped) == ([], 0, 1)


def test_zero_judged_report_cannot_pass():
    assert sweep._report([], 0, 3, _args()) != 0


def test_nonreproducing_loss_still_emits_a_complete_measured_row(monkeypatch, capsys):
    results = iter([_result(out="RATIO 0.8000 numpy.sin\n"),
                    _result(out="RATIO 1.2000 numpy.sin\n")])
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: next(results))
    losses, judged, skipped = sweep._measure_all(["sample"], _args(), None)
    assert (losses, judged, skipped) == ([], 1, 0)
    measured = [across.CELL.match(line) for line in capsys.readouterr().out.splitlines()]
    assert any(match and float(match.group(1)) == 1.2 for match in measured)


def test_measure_records_samples_and_correctness_without_changing_ratio(monkeypatch):
    stock = lambda: sweep.np.array([1.0, 2.0])
    holder = SimpleNamespace(op=stock)
    monkeypatch.setattr(sweep, "_resolve", lambda op: (holder, "op"))
    monkeypatch.setattr(sweep.GEARBOX, "stock_fn", lambda op: stock)
    times = iter([0.03, 0.058, 0.029, 0.06, 0.03, 0.062, 0.031])

    def timeit(fn, number):
        fn()
        return next(times)

    monkeypatch.setattr(sweep.timeit, "timeit", timeit)
    evidence = {}
    assert sweep._measure("numpy.op", (), {}, 3, evidence=evidence,
                          comparison_mode="bit-identical") == 2.0
    assert evidence["stock_seconds"] == [0.058, 0.06, 0.062]
    assert evidence["patched_seconds"] == [0.029, 0.03, 0.031]
    assert evidence["correctness"]["passed"] is True
    assert evidence["iterations_per_sample"] == 1
    assert evidence["ratio"] == 2.0


def test_measure_refuses_to_time_wrong_answers(monkeypatch):
    monkeypatch.setattr(sweep, "_resolve", lambda op: (SimpleNamespace(op=lambda: 2), "op"))
    monkeypatch.setattr(sweep.GEARBOX, "stock_fn", lambda op: lambda: 1)
    monkeypatch.setattr(sweep.timeit, "timeit", lambda *a, **k: pytest.fail("timed wrong answer"))
    evidence = {}
    with pytest.raises(RuntimeError, match="correctness"):
        sweep._measure("numpy.op", (), {}, 3, evidence=evidence)
    assert evidence["correctness"]["passed"] is False


@pytest.mark.parametrize("loads,expected_calls,expected_code", [
    ([0.25, 0.10], 0, 1),
    ([0.10, 0.30], 1, 1),
    ([0.10, 0.15], 1, 0),
])
def test_json_quiet_contract_records_conditions_and_filters(
        monkeypatch, tmp_path, loads, expected_calls, expected_code):
    readings = iter(loads)
    monkeypatch.setattr(sweep, "_foreign_load", lambda: next(readings), raising=False)
    monkeypatch.setattr(sweep, "_fingerprint", lambda: {"fingerprint": "test", "numpy": "2.4.5"}, raising=False)
    monkeypatch.setattr(sweep, "_source_revision", lambda: "test-revision", raising=False)
    calls = []

    def run(args):
        calls.append(True)
        args._evidence["summary"] = {"judged": 1, "skipped": 0, "losses": []}
        return 0

    monkeypatch.setattr(sweep, "_run_sweep", run, raising=False)
    output = tmp_path / "evidence.json"
    code = sweep.main(["tool", "--json", str(output), "--require-quiet",
                       "--rows", "--values", "--only", "unique"])
    assert code == expected_code
    assert len(calls) == expected_calls
    evidence = json.loads(output.read_text())
    assert evidence["fingerprint"]["fingerprint"] == "test"
    assert evidence["conditions"]["cpu_busy_before"] == loads[0]
    assert evidence["conditions"]["cpu_busy_after"] == loads[1]
    assert evidence["conditions"]["contended"] == (max(loads) > 0.20)
    assert evidence["selection"]["only"] == ["unique"]
    assert evidence["selection"]["rows"] is True
    assert evidence["selection"]["values"] is True
    assert evidence["exit_code"] == code


def test_child_sidecar_preserves_stdout_protocol_and_dispatch_evidence(monkeypatch, tmp_path, capsys):
    def measure(cell, cutoff, evidence=None, **sampling):
        evidence.update(dispatch={"chosen": "sample", "reason": "predicate-accepted"},
                        stock_seconds=[0.04], patched_seconds=[0.02], ratio=2.0)
        return "RATIO 2.0000 numpy.sin"

    monkeypatch.setattr(sweep, "_measure_one", measure)
    output = tmp_path / "child.json"
    assert sweep.main(["tool", "--one", "sample", "--evidence-cell", str(output)]) == 0
    assert capsys.readouterr().out.strip() == "RATIO 2.0000 numpy.sin"
    evidence = json.loads(output.read_text())
    assert evidence["cell"] == "sample"
    assert evidence["dispatch"]["chosen"] == "sample"
    assert evidence["result"] == "RATIO 2.0000 numpy.sin"


def test_parent_collects_child_sidecars_for_core_retries(monkeypatch, tmp_path):
    evidence = {"cells": []}
    args = _args(_evidence=evidence, _evidence_dir=tmp_path)
    results = iter(["SLOW-CORE", "RATIO 2.0000 numpy.sin"])

    def run(cmd, **kwargs):
        result = next(results)
        path = Path(cmd[cmd.index("--evidence-cell") + 1])
        record = _valid_ratio_record() if result.startswith("RATIO") else {"cell": "sample"}
        record["result"] = result
        path.write_text(json.dumps(record))
        return _result(out=result)

    monkeypatch.setattr(sweep.subprocess, "run", run)
    _, parts = sweep._measure_one_cell("sample", args, 100)
    assert parts == ["RATIO", "2.0000", "numpy.sin"]
    assert [cell["result"] for cell in evidence["cells"]] == ["SLOW-CORE", "RATIO 2.0000 numpy.sin"]


def test_failed_sweep_still_writes_partial_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "_foreign_load", lambda: 0.10)
    monkeypatch.setattr(sweep, "_fingerprint", lambda: {"fingerprint": "test"})
    monkeypatch.setattr(sweep, "_source_revision", lambda: "test-revision")

    def run(args):
        args._evidence["cells"].append({"cell": "sample", "error": "crashed"})
        raise RuntimeError("child crashed")

    monkeypatch.setattr(sweep, "_run_sweep", run)
    output = tmp_path / "partial.json"
    assert sweep.main(["tool", "--json", str(output)]) == 1
    evidence = json.loads(output.read_text())
    assert evidence["completed"] is False
    assert evidence["cells"][0]["error"] == "crashed"
    assert "child crashed" in evidence["error"]
    assert evidence["conditions"]["cpu_busy_after"] == 0.10


def test_missing_sidecar_cannot_be_judged(monkeypatch, tmp_path):
    args = _args(_evidence={"cells": []}, _evidence_dir=tmp_path)
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: _result(out="RATIO 2.0000 numpy.sin"))
    with pytest.raises(RuntimeError, match="missing/invalid child evidence"):
        sweep._measure_one_cell("sample", args, None)
    assert "error" in args._evidence["cells"][0]


def test_unknown_load_is_not_assumed_quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "_foreign_load", lambda: None)
    monkeypatch.setattr(sweep, "_fingerprint", lambda: {})
    monkeypatch.setattr(sweep, "_source_revision", lambda: None)
    monkeypatch.setattr(sweep, "_run_sweep", lambda args: pytest.fail("timed under unknown load"))
    output = tmp_path / "unknown.json"
    assert sweep.main(["tool", "--json", str(output), "--require-quiet"]) == 1
    assert json.loads(output.read_text())["conditions"]["load_known"] is False


def test_default_cli_does_not_sample_load_or_collect_evidence(monkeypatch):
    monkeypatch.setattr(sweep, "_foreign_load", lambda: pytest.fail("changed default measurement"))
    monkeypatch.setattr(sweep, "_fingerprint", lambda: pytest.fail("changed default metadata"))
    monkeypatch.setattr(sweep, "_run_sweep", lambda args: 0)
    assert sweep.main(["tool"]) == 0


def test_child_correctness_failure_is_saved_without_ratio(monkeypatch, tmp_path):
    def measure(cell, cutoff, evidence=None, **sampling):
        evidence["correctness"] = {"passed": False}
        raise RuntimeError("correctness check failed")

    monkeypatch.setattr(sweep, "_measure_one", measure)
    output = tmp_path / "incorrect.json"
    with pytest.raises(RuntimeError, match="correctness check"):
        sweep.main(["tool", "--one", "sample", "--evidence-cell", str(output)])
    evidence = json.loads(output.read_text())
    assert evidence["correctness"]["passed"] is False
    assert "result" not in evidence
    assert "correctness check failed" in evidence["error"]


def test_source_manifest_is_order_independent_and_tracks_dirty_content(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    files = {"src/pyoverdrive/a.py": "a = 1\n", "tools/verify_no_pessimization.py": "# tool\n",
             "lab/dyno/load.py": "# load\n", "pyproject.toml": "# metadata\n"}
    for root, entries in ((first, files.items()), (second, reversed(list(files.items())))):
        for name, content in entries:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    before = sweep._source_manifest(first)
    assert before == sweep._source_manifest(second)
    assert set(before["files"]) == set(files)
    (second / "src/pyoverdrive/a.py").write_text("a = 2\n")
    assert before["sha256"] != sweep._source_manifest(second)["sha256"]


def test_missing_git_is_optional_metadata(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("no git executable")

    monkeypatch.setattr(sweep.subprocess, "run", unavailable)
    assert sweep._source_revision() is None


def test_correctness_snapshots_shared_output_before_patched_call(monkeypatch):
    out = sweep.np.empty(2)

    def stock(*, out):
        out[:] = [1.0, 2.0]
        return (out,)

    def incorrect(*, out):
        out[:] = [99.0, 99.0]
        return (out,)

    monkeypatch.setattr(sweep, "_resolve", lambda op: (SimpleNamespace(op=incorrect), "op"))
    monkeypatch.setattr(sweep.GEARBOX, "stock_fn", lambda op: stock)
    monkeypatch.setattr(sweep.timeit, "timeit", lambda *a, **k: pytest.fail("timed wrong shared output"))
    evidence = {}
    with pytest.raises(RuntimeError, match="correctness"):
        sweep._measure("numpy.op", (), {"out": out}, 1, evidence=evidence,
                       comparison_mode="bit-identical")
    assert evidence["correctness"]["passed"] is False


def _valid_ratio_record():
    return {"cell": "sample", "result": "RATIO 2.0000 numpy.sin", "op": "numpy.sin",
            "path": "sample", "dispatch": {"chosen": "sample", "reason": "predicate-accepted", "fallback_observed": False},
            "correctness": {"passed": True}, "stock_seconds": [0.058, 0.06, 0.062],
            "patched_seconds": [0.029, 0.03, 0.031], "rounds": 3,
            "iterations_per_sample": 1, "stock_median_seconds": 0.06,
            "patched_median_seconds": 0.03, "ratio": 2.0}


@pytest.mark.parametrize("change", [
    {"result": "SKIP no-dispatch"},
    {"correctness": {"passed": False}},
    {"stock_seconds": []},
    {"patched_seconds": [0.02]},
    {"stock_seconds": [float("nan"), 0.06, 0.08]},
    {"rounds": True},
    {"iterations_per_sample": 0},
    {"ratio": 1.2},
    {"stock_median_seconds": 0.04},
    {"dispatch": {"chosen": "stock"}},
    {"dispatch": {"chosen": "sample", "reason": "predicate-accepted"}},
])
def test_malformed_ratio_sidecar_cannot_be_judged(monkeypatch, tmp_path, change):
    args = _args(_evidence={"cells": []}, _evidence_dir=tmp_path)

    def run(cmd, **kwargs):
        record = _valid_ratio_record()
        record.update(change)
        path = Path(cmd[cmd.index("--evidence-cell") + 1])
        path.write_text(json.dumps(record), encoding="utf-8")
        return _result(out="RATIO 2.0000 numpy.sin")

    monkeypatch.setattr(sweep.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="evidence"):
        sweep._measure_one_cell("sample", args, None)
    assert "error" in args._evidence["cells"][0]


def test_ratio_sidecar_requires_raw_samples(monkeypatch, tmp_path):
    args = _args(_evidence={"cells": []}, _evidence_dir=tmp_path)

    def run(cmd, **kwargs):
        path = Path(cmd[cmd.index("--evidence-cell") + 1])
        path.write_text(json.dumps({"cell": "sample", "result": "RATIO 2.0000 numpy.sin"}))
        return _result(out="RATIO 2.0000 numpy.sin")

    monkeypatch.setattr(sweep.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="evidence"):
        sweep._measure_one_cell("sample", args, None)


def test_calibration_metadata_freezes_loaded_records_without_exposing_override_path(monkeypatch, tmp_path):
    from pyoverdrive import calibration

    saved = {"pyrallel": {"drop": {"pyrallel_sin": ["float32"]}}}
    monkeypatch.setattr(calibration, "load", lambda: saved)
    private_path = str(tmp_path / "private-calibration.json")
    monkeypatch.setenv("PYOVERDRIVE_CALIBRATION", private_path)
    metadata = sweep._calibration_metadata()
    saved["pyrallel"]["drop"].clear()
    assert metadata == {"saved": {"pyrallel": {"drop": {"pyrallel_sin": ["float32"]}}},
                        "override_configured": True}
    assert private_path not in json.dumps(metadata)


def test_evidence_includes_loaded_calibration(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "_foreign_load", lambda: 0.10)
    monkeypatch.setattr(sweep, "_fingerprint", lambda: {})
    monkeypatch.setattr(sweep, "_source_revision", lambda: None)
    metadata = {"saved": {"argmax_blocked_transpose": {"enabled": True}},
                "override_configured": False}
    monkeypatch.setattr(sweep, "_calibration_metadata", lambda: metadata, raising=False)
    monkeypatch.setattr(sweep, "_run_sweep", lambda args: 1)
    output = tmp_path / "evidence.json"
    sweep.main(["tool", "--json", str(output)])
    assert json.loads(output.read_text())["calibration"] == metadata


@pytest.mark.parametrize("fail_on_call", [1, 5])
def test_fastpath_fallback_cannot_be_verified_as_execution(monkeypatch, fail_on_call):
    from pyoverdrive.dispatcher.gearbox import FastPath, Gearbox

    box = Gearbox()
    calls = 0
    stock = lambda: sweep.np.array([1.0])

    def candidate():
        nonlocal calls
        calls += 1
        if calls >= fail_on_call:
            raise ValueError("candidate broke")
        return stock()

    box.register(FastPath("sample", "numpy.op", lambda a, k: True, candidate))
    wrapped = box._make_wrapper("numpy.op", stock)
    monkeypatch.setattr(sweep, "GEARBOX", box)
    monkeypatch.setattr(box, "stock_fn", lambda op: stock)
    monkeypatch.setattr(sweep, "_resolve", lambda op: (SimpleNamespace(op=wrapped), "op"))

    def timeit(fn, number):
        for _ in range(number):
            fn()
        return 0.03

    monkeypatch.setattr(sweep.timeit, "timeit", timeit)
    evidence = {"path": "sample", "dispatch": {"chosen": "sample"}}
    with pytest.warns(RuntimeWarning, match="fast path"):
        with pytest.raises(RuntimeError, match="fell back"):
            sweep._measure("numpy.op", (), {}, 1, evidence=evidence,
                           comparison_mode="bit-identical")
    assert evidence["dispatch"]["fallback_observed"] is True
