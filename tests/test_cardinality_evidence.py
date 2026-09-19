"""Cardinality evidence orchestration, using fake workers and no timings."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "tools" / "probe_cardinality.py"
spec = importlib.util.spec_from_file_location("probe_cardinality", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def restore_activation():
    yield
    probe.pyoverdrive.disable()


def test_target_selectors_keep_middle_cardinality_and_floor_neighbors():
    args = probe._parse_args(["tool", "--op", "unique", "--dtype", "int64",
                              "--sizes", "999,1000", "--sizes", "10000",
                              "--cardinalities", "16,256", "--cardinalities", "u"])
    assert probe._sizes_for("unique", "int64", args) == [999, 1000, 10000]
    assert args.cardinalities == [16, 256, None]
    args = probe._parse_args(["tool", "--floor-neighbors"])
    floor = probe._floor("unique", "int64")
    assert probe._sizes_for("unique", "int64", args) == [floor - 1, floor, floor + 1]
    sizes = probe._sizes_for("intersect", "int16", args)
    floor = probe._floor("intersect", "int16")
    assert sizes[0] + max(2, sizes[0] // 10) < floor
    assert sizes[1] + max(2, sizes[1] // 10) >= floor


@pytest.mark.parametrize("code,out", [(1, ""), (1, "RATIO 2.0000 unique_sort"), (0, "")])
def test_failed_workers_cannot_be_skipped(monkeypatch, code, out):
    args = probe._parse_args(["tool"])
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], code, out, "crash"))
    with pytest.raises(RuntimeError):
        probe._measure_cell("unique,1000,256,int64", args, None)


def test_incomplete_repeats_cannot_reuse_first_ratio(monkeypatch):
    args = probe._parse_args(["tool", "--repeat", "2"])
    outputs = iter(["RATIO 2.0000 unique_sort", "NODISPATCH refused"])
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, next(outputs), ""))
    with pytest.raises(RuntimeError, match="incomplete"):
        probe._measure_cell("unique,1000,256,int64", args, None)


@pytest.mark.parametrize("output", ["NODISPATCH refused", "NOFIT cardinality exceeds range"])
def test_expected_skips_remain_allowed(monkeypatch, output):
    args = probe._parse_args(["tool"])
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, output, ""))
    ratio, result = probe._measure_cell("unique,1000,256,int64", args, None)
    assert ratio is None
    assert result == output


def test_json_refuses_busy_host_without_starting_grid(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "_foreign_load", lambda: 0.30, raising=False)
    monkeypatch.setattr(probe, "_fingerprint", lambda: {"fingerprint": "test"}, raising=False)
    monkeypatch.setattr(probe, "_source_revision", lambda: None, raising=False)
    monkeypatch.setattr(probe, "_run_grid", lambda args: pytest.fail("busy timing"), raising=False)
    output = tmp_path / "busy.json"
    assert probe.main(["tool", "--json", str(output), "--require-quiet", "--cardinalities", "256"]) == 1
    evidence = json.loads(output.read_text())
    assert evidence["selection"]["cardinalities"] == [256]
    assert evidence["conditions"]["contended"] is True
    assert "tools/probe_cardinality.py" in evidence["source"]["files"]


def test_all_skipped_grid_is_not_verified(monkeypatch):
    args = probe._parse_args(["tool", "--op", "unique", "--dtype", "int64",
                              "--sizes", "1", "--cardinalities", "256"])
    monkeypatch.setattr(probe.cpuclass, "classify", lambda: {"hybrid": False, "fast": []})
    monkeypatch.setattr(probe, "_measure_cell", lambda *a: (None, "NODISPATCH refused"))
    assert probe._run_grid(args) == 1


def test_child_json_keeps_raw_samples_and_original_stdout(monkeypatch, tmp_path, capsys):
    def one(spec, cutoff, rounds, warmup, evidence=None):
        evidence.update(dispatch={"chosen": "unique_sort"}, correctness={"passed": True},
                        stock_seconds=[0.02], patched_seconds=[0.01])
        return "RATIO 2.0000 unique_sort"

    monkeypatch.setattr(probe, "_one", one)
    output = tmp_path / "child.json"
    assert probe.main(["tool", "--one", "unique,1000,256,int64", "--evidence-cell", str(output)]) == 0
    assert capsys.readouterr().out.strip() == "RATIO 2.0000 unique_sort"
    evidence = json.loads(output.read_text())
    assert evidence["correctness"]["passed"] is True
    assert evidence["stock_seconds"] == [0.02]


@pytest.mark.parametrize("op", probe.OPS)
@pytest.mark.parametrize("dtype", probe.DTYPES)
def test_each_shipped_row_has_neighbors_on_both_sides(op, dtype):
    args = probe._parse_args(["tool", "--floor-neighbors"])
    sizes = probe._sizes_for(op, dtype, args)
    totals = [n + max(2, n // 10) if op == "intersect" else n for n in sizes]
    assert totals[0] < probe._floor(op, dtype) <= totals[1] < totals[2]


@pytest.mark.parametrize("dtype", probe.DTYPES)
def test_draw_is_replayable_and_does_not_exceed_cardinality(dtype):
    a = probe._draw(1000, 256, dtype, 9)
    assert a.dtype == probe.np.dtype(dtype)
    assert probe.np.array_equal(a, probe._draw(1000, 256, dtype, 9))
    assert 1 <= probe.np.unique(a).size <= 256


def test_dense_int16_pool_fits_and_excess_cardinality_refuses():
    assert probe._draw(1000, 65536, "int16", 9).size == 1000
    with pytest.raises(ValueError, match="exceeds"):
        probe._draw(1000, 65537, "int16", 9)


def test_child_evidence_uses_shared_timing_and_validates_with_shared_protocol(monkeypatch, tmp_path):
    calls = []

    def measure(op, inputs, kwargs, rounds, evidence=None, comparison_mode=None):
        calls.append((op, comparison_mode))
        if evidence is not None:
            evidence.update(stock_seconds=[0.02] * rounds, patched_seconds=[0.01] * rounds,
                            stock_median_seconds=0.02, patched_median_seconds=0.01,
                            ratio=2.0, rounds=rounds, iterations_per_sample=1,
                            correctness={"passed": True, "mode": comparison_mode})
            evidence["dispatch"]["fallback_observed"] = False
        return 2.0

    monkeypatch.setattr(probe, "_measure", measure)
    cell = "unique,10000,256,int32"
    evidence = {"cell": cell}
    result = probe._one(cell, None, rounds=3, evidence=evidence)
    evidence["result"] = result
    assert evidence["dispatch"]["chosen"] == "unique_sort"
    assert evidence["requested_pool_cardinality"] == 256
    assert evidence["combined_size"] == 10000
    assert 1 <= evidence["observed_cardinalities"][0] <= 256
    assert calls[0][1] is not None
    sidecar = tmp_path / "child.json"
    sidecar.write_text(json.dumps(evidence))
    args = probe._parse_args(["tool"])
    args._evidence = {"cells": []}
    proc = subprocess.CompletedProcess([], 0, result, "")
    probe.sweep._collect_child_evidence(args, sidecar, cell, "repeat-1", 1, proc)
    assert args._evidence["cells"][0]["ratio"] == 2.0


def test_repeated_measurement_keeps_worst_independent_ratio(monkeypatch):
    args = probe._parse_args(["tool", "--repeat", "2"])
    outputs = iter(["SLOW-CORE", "RATIO 1.2000 unique_sort", "RATIO 0.9000 unique_sort"])
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, next(outputs), ""))
    ratio, result = probe._measure_cell("unique,1000,256,int64", args, 100)
    assert ratio == 0.9


def test_final_contention_invalidates_completed_grid(monkeypatch, tmp_path):
    loads = iter([0.10, 0.30])
    monkeypatch.setattr(probe, "_foreign_load", lambda: next(loads))
    monkeypatch.setattr(probe, "_fingerprint", lambda: {})
    monkeypatch.setattr(probe, "_source_revision", lambda: None)

    def run(args):
        args._evidence["summary"] = {"judged": 1, "skipped": 0, "losses": []}
        return 0

    monkeypatch.setattr(probe, "_run_grid", run)
    output = tmp_path / "contended.json"
    assert probe.main(["tool", "--json", str(output), "--require-quiet"]) == 1
    evidence = json.loads(output.read_text())
    assert evidence["completed"] is True
    assert evidence["conditions"]["contended"] is True


def test_crashed_grid_preserves_partial_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "_foreign_load", lambda: 0.10)
    monkeypatch.setattr(probe, "_fingerprint", lambda: {})
    monkeypatch.setattr(probe, "_source_revision", lambda: None)

    def run(args):
        args._evidence["cells"].append({"cell": "sample", "error": "worker crashed"})
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(probe, "_run_grid", run)
    output = tmp_path / "partial.json"
    assert probe.main(["tool", "--json", str(output)]) == 1
    evidence = json.loads(output.read_text())
    assert evidence["completed"] is False
    assert evidence["cells"][0]["error"] == "worker crashed"


@pytest.mark.parametrize("output", ["NODISPATCH refused", "NOFIT too many values", "UNMEASURABLE"])
def test_legacy_skip_sidecars_pass_shared_validation(monkeypatch, tmp_path, output):
    cell = "unique,1,256,int64"
    args = probe._parse_args(["tool"])
    args._evidence, args._evidence_dir = {"cells": []}, tmp_path

    def run(cmd, **kwargs):
        path = Path(cmd[cmd.index("--evidence-cell") + 1])
        path.write_text(json.dumps({"cell": cell, "result": output}))
        return subprocess.CompletedProcess(cmd, 0, output, "")

    monkeypatch.setattr(probe.subprocess, "run", run)
    assert probe._measure_cell(cell, args, None) == (None, output)
    assert args._evidence["cells"][0]["original_result"] == output


@pytest.mark.parametrize("op,dtype", [
    ("unique", "uint16"), ("unique_values", "uint16"), ("intersect", "int8"),
    ("unique", "int16"), ("unique_values", "int16"),
    ("unique", "int64"), ("unique", "uint64"),
    ("unique_values", "int64"), ("unique_values", "uint64"),
    ("intersect", "int16"), ("intersect", "uint16"),
    ("intersect", "int64"), ("intersect", "uint64"), ("intersect", "uint8"),
])
def test_withdrawn_rows_replay_as_nodispatch_without_timing(monkeypatch, op, dtype):
    monkeypatch.setattr(probe, "_measure", lambda *a, **k: pytest.fail("withdrawn dtype was timed"))
    args = probe._parse_args(["tool", "--op", op, "--dtype", dtype, "--floor-neighbors"])
    sizes = probe._sizes_for(op, dtype, args)
    for n in sizes:
        evidence = {}
        result = probe._one(f"{op},{n},16,{dtype}", None, evidence=evidence)
        assert result.startswith("NODISPATCH")
        assert evidence["dispatch"]["chosen"] == "stock"
