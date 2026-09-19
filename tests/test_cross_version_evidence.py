"""Cross-version evidence must keep every requested version and its verdict."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "tools" / "verify_across_numpy.py"
_SPEC = importlib.util.spec_from_file_location("cross_version_evidence_tool", _PATH)
across = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(across)


def _completed_output():
    return ("  1.2000x sample numpy.sin\n"
            "judged 1 dispatching paths in their own processes, skipped 2 "
            "(no canonical input, or it does not dispatch)\n"
            "no dispatching path is below 1x\n")


@pytest.mark.parametrize("quiet", [False, True])
def test_each_requested_version_gets_distinct_absolute_evidence(monkeypatch, tmp_path, quiet):
    monkeypatch.chdir(tmp_path)
    options = ["tool", "--json-dir", "evidence"]
    if quiet:
        options.append("--require-quiet")
    args = across._parse_args(options, ("2.5.2", "latest"), "test")
    monkeypatch.setattr(across, "_venv_for", lambda *a: (Path("python"), "2.5.2"))
    commands = []

    def child(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, _completed_output(), "")

    monkeypatch.setattr(across, "_run", child)
    ran, unverified = across._collect_all(["2.5.2", "latest"], tmp_path, args)
    assert not unverified
    paths = [Path(cmd[cmd.index("--json") + 1]) for cmd in commands]
    assert len(set(paths)) == 2
    assert all(path.is_absolute() and path.parent == tmp_path / "evidence" for path in paths)
    assert paths[0].parent.is_dir()
    assert all(("--require-quiet" in cmd) is quiet for cmd in commands)
    assert len(ran) == 2  # latest and a pin must not overwrite each other


def test_options_are_opt_in(monkeypatch, tmp_path):
    args = across._parse_args(["tool"], ("2.3.0",), "test")
    monkeypatch.setattr(across, "_venv_for", lambda *a: (Path("python"), "2.3.0"))
    commands = []

    def child(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, _completed_output(), "")

    monkeypatch.setattr(across, "_run", child)
    across._run_one_version("2.3.0", tmp_path, args)
    assert "--json" not in commands[0]
    assert "--require-quiet" not in commands[0]


@pytest.mark.parametrize("stage", ["before", "after"])
def test_contended_version_reason_and_incomplete_coverage_are_reported(monkeypatch, tmp_path, capsys, stage):
    args = across._parse_args(["tool", "--require-quiet"], ("2.3.0", "latest"), "test")
    monkeypatch.setattr(across, "_venv_for", lambda version, cache: (Path("python"), version))
    results = iter([
        subprocess.CompletedProcess([], 0, _completed_output(), ""),
        subprocess.CompletedProcess([], 1, (_completed_output() if stage == "after" else "") +
                                    "NOT verified: foreign CPU load is contended\n", ""),
    ])
    monkeypatch.setattr(across, "_run", lambda *a, **k: next(results))
    ran, unverified = across._collect_all(["2.3.0", "latest"], tmp_path, args)
    assert len(unverified) == 1
    assert "contended" in unverified[0][1]
    labels, _, losses, interesting = across._compare(ran, args)
    assert across._report(labels, losses, interesting, ran, unverified, args) == 1
    output = capsys.readouterr().out
    assert "1/2" in output
    assert "skipped" in output


def test_unavailable_version_is_in_coverage_count(monkeypatch, tmp_path, capsys):
    args = across._parse_args(["tool"], ("2.3.0", "latest"), "test")
    monkeypatch.setattr(across, "_run_one_version", lambda version, *a:
                        (None, {}, "no wheel") if version == "2.3.0" else
                        ("2.5.2", {"sample": 1.2}, None))
    ran, unverified = across._collect_all(["2.3.0", "latest"], tmp_path, args)
    labels, _, losses, interesting = across._compare(ran, args)
    assert across._report(labels, losses, interesting, ran, unverified, args) == 1
    assert "1/2" in capsys.readouterr().out


@pytest.mark.parametrize("workflow", [None, "jobs: {}"])
def test_missing_workflow_fallback_uses_declared_supported_floor(monkeypatch, tmp_path, workflow):
    if workflow is not None:
        path = tmp_path / ".github" / "workflows" / "ci.yml"
        path.parent.mkdir(parents=True)
        path.write_text(workflow, encoding="utf-8")
    monkeypatch.setattr(across, "REPO", tmp_path)
    versions, source = across.supported_versions()
    assert source.startswith("FALLBACK")
    assert versions == ("2.3.0", "2.4.5", "latest")
