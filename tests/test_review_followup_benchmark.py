"""Small correctness/control checks; these never run benchmark measurements."""

import json
import subprocess

import numpy as np
import pytest


def _measured_record(name):
    from benchmarks.micro import bench_review_followup as b

    cell = b.cells()[name]
    path = {"argmax": "argmax_blocked_transpose", "vectorize": "vectorize_ufunc_direct",
            "out": f"pyrallel_{cell.op}"}[cell.family]
    candidates = ["candidate_guard_on"] + (["candidate_guard_off"] if cell.family == "out" else [])
    record = {
        "cell": name, "status": "measured", "family": cell.family, "operation": cell.op,
        "dtype": cell.dtype, "shape": list(cell.shape), "alias": cell.alias,
        "fingerprint": {"fingerprint": "test", "python": "test", "numpy": "test"},
        "correctness": {"passed": True, "mode": "bit-identical"},
        "conditions": b._conditions(0.01, 0.01), "dispatch": [path, "predicate-accepted"],
        "experimental_enable": path, "rounds": 1, "calls_per_sample": 1,
        "timings_seconds": {"stock": [0.04], **{label: [0.02] for label in candidates}},
        "median_seconds": {"stock": 0.04, **{label: 0.02 for label in candidates}},
        "ratios_stock_over_candidate": {label: 2.0 for label in candidates},
        "execution": {"path": path, "fallback_observed": False, "check": "Gearbox._warned_paths"},
    }
    if cell.family == "vectorize":
        record["execution"]["direct_ufunc"] = cell.op
    if cell.family == "out":
        record["guard_overhead_seconds"] = 0.0
    return record


def test_followup_cells_cover_requested_axes_and_bound_argmax_memory():
    from benchmarks.micro import bench_review_followup as b

    cells = list(b.cells().values())
    argmax = [c for c in cells if c.family == "argmax"]
    assert {c.dtype for c in argmax} == {"float64", "float32", "int64"}
    assert {(3000, 3000), (2999, 3000), (3000, 2999), (4500, 3000),
            (3000, 4500), (10000, 1000)} <= {c.shape for c in argmax}
    assert max(np.prod(c.shape) * np.dtype(c.dtype).itemsize for c in argmax) < 150_000_000
    vectorize = [c for c in cells if c.family == "vectorize"]
    assert {(c.op, c.shape) for c in vectorize} == {
        (op, (n,)) for op in ("sin", "sqrt", "exp") for n in (16, 10_000, 1_000_000)
    }
    outputs = [c for c in cells if c.family == "out"]
    assert {c.alias for c in outputs} == {"disjoint", "inplace"}
    assert {c.op for c in outputs} == {"sin", "add"}


@pytest.mark.parametrize("op", ["sin", "add"])
@pytest.mark.parametrize("alias", ["disjoint", "inplace"])
def test_followup_resets_mutating_inputs_and_restores_guard(op, alias):
    from benchmarks.micro import bench_review_followup as b
    from pyoverdrive.parallel import pyrallel

    original = pyrallel.output_alias_safe
    spec = b.Cell("out", op, "float64", (16,), alias)
    with b.workload(spec) as w:
        w.reset()
        expected = w.stock().copy()
        for _ in range(3):
            w.reset()
            np.testing.assert_array_equal(w.candidate(), expected)
            with b.without_alias_guard(w.inputs, w.out):
                w.reset()
                np.testing.assert_array_equal(w.candidate(), expected)
        assert pyrallel.output_alias_safe is original
    assert not b.pyoverdrive.enabled()


def test_followup_guard_override_rejects_overlap_and_restores_on_exception():
    from benchmarks.micro import bench_review_followup as b
    from pyoverdrive.parallel import pyrallel

    a = np.arange(17.0)
    original = pyrallel.output_alias_safe
    with pytest.raises(ValueError, match="overlap"):
        with b.without_alias_guard((a[:-1],), a[1:]):
            pytest.fail("unsafe alias was accepted")
    with pytest.raises(RuntimeError, match="intentional"):
        with b.without_alias_guard((a,), a):
            raise RuntimeError("intentional")
    assert pyrallel.output_alias_safe is original


def test_followup_list_does_not_measure(monkeypatch, capsys):
    from benchmarks.micro import bench_review_followup as b

    def forbidden():
        pytest.fail("listing cells must not sample load or classify CPUs")
    monkeypatch.setattr(b.sweep, "_foreign_load", forbidden)
    monkeypatch.setattr(b.cpuclass, "classify", forbidden)
    assert b.main(["--list", "--family", "vectorize"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 9


@pytest.mark.parametrize("load", [None, 0.21])
def test_followup_quiet_refusal_persists_evidence_without_children(load, tmp_path, monkeypatch):
    from benchmarks.micro import bench_review_followup as b

    monkeypatch.setattr(b.sweep, "_foreign_load", lambda: load)
    monkeypatch.setattr(b.sweep, "_fingerprint", lambda: {"numpy": "test", "python": "test"})
    monkeypatch.setattr(b.sweep, "_source_revision", lambda: "test")
    def forbidden():
        pytest.fail("quiet refusal must precede CPU classification")
    monkeypatch.setattr(b.cpuclass, "classify", forbidden)
    path = tmp_path / "result.json"
    assert b.main(["--require-quiet", "--json", str(path)]) == 1
    record = json.loads(path.read_text())
    assert not record["completed"]
    assert record["cells"] == []
    assert record["conditions"]["cpu_busy_before"] == load
    import hashlib
    from pathlib import Path
    script = Path(b.__file__)
    assert record["source"]["files"]["src/pyoverdrive/parallel/pyrallel.py"]
    assert record["benchmark_source"] == {
        "path": "benchmarks/micro/bench_review_followup.py",
        "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }


def test_followup_child_failure_is_not_a_successful_skip(tmp_path, monkeypatch):
    from benchmarks.micro import bench_review_followup as b

    monkeypatch.setattr(b.sweep, "_foreign_load", lambda: 0.01)
    monkeypatch.setattr(b.sweep, "_fingerprint", lambda: {"numpy": "test", "python": "test"})
    monkeypatch.setattr(b.sweep, "_source_revision", lambda: "test")
    monkeypatch.setattr(b.cpuclass, "classify", lambda: {"hybrid": False})
    monkeypatch.setattr(b.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "child crashed"))
    path = tmp_path / "result.json"
    assert b.main(["--family", "vectorize", "--json", str(path)]) == 1
    record = json.loads(path.read_text())
    assert not record["completed"]
    assert "child crashed" in record["error"]


def test_followup_measure_records_raw_samples_and_real_correctness_without_timing(monkeypatch):
    from benchmarks.micro import bench_review_followup as b

    def fixed_clock(call, reset):
        reset()
        b.sweep._consume(call())
        return 0.02
    monkeypatch.setattr(b, "_sample", fixed_clock)
    result = b.measure(b.Cell("vectorize", "sqrt", "float64", (16,)), 3)
    assert result["correctness"] == {"passed": True, "mode": "bit-identical"}
    assert result["dispatch"][0] == "vectorize_ufunc_direct"
    assert result["timings_seconds"] == {
        "stock": [0.02, 0.02, 0.02], "candidate_guard_on": [0.02, 0.02, 0.02],
    }
    assert result["calls_per_sample"] == 1
    assert not b.pyoverdrive.enabled()


def test_followup_reset_is_outside_clock_and_consumption_inside(monkeypatch):
    from benchmarks.micro import bench_review_followup as b

    order = []
    times = iter((10.0, 10.5))
    def clock():
        order.append("clock")
        return next(times)
    monkeypatch.setattr(b.time, "perf_counter", clock)
    monkeypatch.setattr(b.sweep, "_consume", lambda result: order.append("consume"))
    elapsed = b._sample(lambda: order.append("call"), lambda: order.append("reset"))
    assert elapsed == 0.5
    assert order == ["reset", "clock", "call", "consume", "clock"]


@pytest.fixture
def followup_parent(tmp_path, monkeypatch):
    from benchmarks.micro import bench_review_followup as b

    monkeypatch.setattr(b.sweep, "_foreign_load", lambda: 0.01)
    monkeypatch.setattr(b.sweep, "_fingerprint", lambda: {"numpy": "test", "python": "test"})
    monkeypatch.setattr(b.sweep, "_source_revision", lambda: None)
    monkeypatch.setattr(b.cpuclass, "classify", lambda: {"hybrid": False})
    names = [name for name, cell in b.cells().items() if cell.family == "vectorize"][:2]
    selection = tmp_path / "cells.txt"
    selection.write_text("\n".join(names) + "\n", encoding="utf-8")
    return b, names, selection


@pytest.mark.parametrize("refusal", ["quiet-before", "quiet-after", "slow-core"])
def test_followup_continues_unverified_cells_and_keeps_quiet_successes(followup_parent, tmp_path, monkeypatch, refusal):
    from pathlib import Path
    import hashlib

    b, names, selection = followup_parent
    visited = []
    def child(cmd, **kwargs):
        name = cmd[cmd.index("--one") + 1]
        visited.append(name)
        record = _measured_record(name)
        code = 0
        if name == names[0]:
            record["status"] = "slow-core" if refusal == "slow-core" else "unverified"
            record["reason_code"] = refusal
            record["reason"] = "test refusal"
            if refusal != "slow-core":
                code = 1
                record["conditions"] = b._conditions(0.3, 0.01) if refusal == "quiet-before" else b._conditions(0.01, 0.3)
        Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps(record))
        return subprocess.CompletedProcess(cmd, code, "", "")
    monkeypatch.setattr(b.subprocess, "run", child)
    path = tmp_path / "result.json"
    assert b.main(["--cells-file", str(selection), "--require-quiet", "--retries", "2", "--json", str(path)]) == 1
    record = json.loads(path.read_text())
    assert visited[-1] == names[1]
    assert visited.count(names[0]) == (2 if refusal == "slow-core" else 1)
    assert record["verified_cells"] == [names[1]]
    assert [r["cell"] for r in record["unverified_cells"]] == [names[0]]
    assert not record["completed"]
    assert record["all_cells_attempted"]
    assert record["selection"]["requested_cells"] == names
    assert record["selection"]["cells_file"]["sha256"] == hashlib.sha256(selection.read_bytes()).hexdigest()


def test_followup_exact_selection_does_not_expand_prefixes(followup_parent, monkeypatch, capsys):
    b, names, selection = followup_parent
    selection.write_text(names[0] + "\n")
    assert b.main(["--cells-file", str(selection), "--list"]) == 0
    assert capsys.readouterr().out.strip() == names[0]
    selection.write_text("vectorize:sin\n")
    with pytest.raises(SystemExit) as exc:
        b.main(["--cells-file", str(selection), "--list"])
    assert exc.value.code == 2


@pytest.mark.parametrize("measurement_error", [False, True])
def test_followup_child_noise_does_not_qualify_ratios_or_mask_errors(tmp_path, monkeypatch, measurement_error):
    from benchmarks.micro import bench_review_followup as b
    from types import SimpleNamespace

    samples = iter((0.01, 0.3))
    monkeypatch.setattr(b.sweep, "_foreign_load", lambda: next(samples))
    monkeypatch.setattr(b.sweep, "_fingerprint", lambda: {})
    def fake_measure(*args):
        if measurement_error:
            raise RuntimeError("correctness failed deliberately")
        return {"status": "measured", "correctness": {"passed": True},
                "ratios_stock_over_candidate": {"candidate_guard_on": 9.0},
                "timings_seconds": {"stock": [0.1], "candidate_guard_on": [0.01]}}
    monkeypatch.setattr(b, "measure", fake_measure)
    path = tmp_path / "child.json"
    args = SimpleNamespace(one=next(iter(b.cells())), require_quiet=True,
                           fast_under=None, rounds=1, json=path)
    assert b._child(args) == 1
    record = json.loads(path.read_text())
    assert "ratios_stock_over_candidate" not in record
    assert not record["qualified"]
    if measurement_error:
        assert record["status"] == "error"
        assert "correctness failed deliberately" in record["error"]
    else:
        assert record["status"] == "unverified"
        assert record["reason_code"] == "quiet-after"
        assert record["timings_seconds"]["stock"] == [0.1]


def test_followup_parent_final_noise_retains_individually_qualified_cells(followup_parent, tmp_path, monkeypatch):
    from pathlib import Path

    b, names, selection = followup_parent
    loads = iter((0.01, 0.3))
    monkeypatch.setattr(b.sweep, "_foreign_load", lambda: next(loads))
    def child(cmd, **kwargs):
        name = cmd[cmd.index("--one") + 1]
        record = _measured_record(name)
        Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps(record))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(b.subprocess, "run", child)
    path = tmp_path / "result.json"
    assert b.main(["--cells-file", str(selection), "--require-quiet", "--json", str(path)]) == 1
    record = json.loads(path.read_text())
    assert record["verified_cells"] == names
    assert record["unverified_cells"] == []
    assert record["batch_unverified_reason"]


def test_followup_correctness_error_aborts_instead_of_skipping(followup_parent, tmp_path, monkeypatch):
    from pathlib import Path

    b, names, selection = followup_parent
    visited = []
    def child(cmd, **kwargs):
        name = cmd[cmd.index("--one") + 1]
        visited.append(name)
        record = {"cell": name, "status": "error", "error": "correctness failed",
                  "conditions": b._conditions(0.01, 0.3)}
        Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps(record))
        return subprocess.CompletedProcess(cmd, 1, "", "")
    monkeypatch.setattr(b.subprocess, "run", child)
    path = tmp_path / "result.json"
    assert b.main(["--cells-file", str(selection), "--json", str(path)]) == 1
    record = json.loads(path.read_text())
    assert visited == names[:1]
    assert record["verified_cells"] == []
    assert record["unverified_cells"][0]["reason_code"] == "execution-error"
    assert record["unverified_cells"][1]["reason_code"] == "not-attempted"
    assert "correctness failed" in record["error"]


@pytest.mark.parametrize("field,value", [
    ("timings_seconds", {}), ("median_seconds", {}),
    ("ratios_stock_over_candidate", {"candidate_guard_on": 99.0}),
    ("correctness", {"passed": "true", "mode": "bit-identical"}),
    ("dispatch", ["stock", "no-applicable-path"]), ("fingerprint", {}),
    ("execution", {}),
    ("timings_seconds", {"stock": [float("nan")], "candidate_guard_on": [0.02]}),
])
def test_followup_parent_rejects_incomplete_or_contradictory_measurement(
        followup_parent, tmp_path, monkeypatch, field, value):
    from pathlib import Path

    b, names, selection = followup_parent
    selection.write_text(names[0] + "\n")

    def child(cmd, **kwargs):
        record = _measured_record(names[0])
        record[field] = value
        Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps(record))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(b.subprocess, "run", child)
    output = tmp_path / "rejected.json"
    assert b.main(["--cells-file", str(selection), "--json", str(output)]) == 1
    record = json.loads(output.read_text())
    assert not record["completed"]
    assert record["verified_cells"] == []


@pytest.mark.parametrize("fail_on_call", [1, 5])
def test_followup_rejects_fallback_during_correctness_or_timing(monkeypatch, fail_on_call):
    from contextlib import contextmanager
    from benchmarks.micro import bench_review_followup as b
    from pyoverdrive.dispatcher.gearbox import FastPath, Gearbox

    box = Gearbox()
    calls = 0
    stock = lambda: np.array([1.0])

    def run():
        nonlocal calls
        calls += 1
        if calls >= fail_on_call:
            raise RuntimeError("candidate failed")
        return stock()

    path = "argmax_blocked_transpose"
    box.register(FastPath(path, "numpy.argmax", lambda a, k: True, run))
    wrapped = box._make_wrapper("numpy.argmax", stock)

    @contextmanager
    def tiny_workload(cell):
        yield b.Workload(stock, wrapped, lambda: None, (), None, path, (path, "predicate-accepted"))

    monkeypatch.setattr(b, "workload", tiny_workload)
    monkeypatch.setattr(b.sweep, "GEARBOX", box)
    monkeypatch.setattr(b, "_sample", lambda call, reset: (call(), 0.02)[1])
    with pytest.warns(RuntimeWarning, match="fast path"):
        with pytest.raises(RuntimeError, match="fell back"):
            b.measure(b.Cell("argmax", "argmax", "float64", (3000, 3000)), 4)


def test_legacy_record_validation_does_not_invent_execution_proof(followup_parent):
    import copy

    b, names, _ = followup_parent
    record = _measured_record(names[0])
    del record["execution"]
    record["stderr"] = ""
    before = copy.deepcopy(record)
    assert b.validate_measurement(record, require_execution=False)["execution_verified"] is False
    assert record == before
    with pytest.raises(ValueError, match="execution"):
        b.validate_measurement(record)


def test_followup_rejects_silent_vectorize_constructor_fallback(monkeypatch):
    from benchmarks.micro import bench_review_followup as b
    from pyoverdrive.fastpaths import vectorize_ufunc

    monkeypatch.setattr(vectorize_ufunc, "_instance_ok", lambda instance: None)
    with pytest.raises(RuntimeError, match="direct ufunc"):
        with b.workload(b.Cell("vectorize", "sin", "float64", (16,))):
            pytest.fail("silently stock vectorize instance was accepted")
    assert not b.pyoverdrive.enabled()


@pytest.mark.parametrize("change", [
    lambda r: r["timings_seconds"]["stock"].__setitem__(0, float("nan")),
    lambda r: r["timings_seconds"]["stock"].__setitem__(0, True),
    lambda r: r.update(rounds=True),
    lambda r: r["conditions"].update(cpu_busy_before=-0.1),
    lambda r: r["conditions"].update(cpu_busy_before=float("-inf")),
    lambda r: r["execution"].update(fallback_observed=True),
    lambda r: r["execution"].update(direct_ufunc="other"),
    lambda r: r.update(stderr="RuntimeWarning: PyOverdrive fast path raised; falling back to stock NumPy"),
])
def test_saved_measurement_rejects_invalid_numeric_and_execution_evidence(followup_parent, change):
    b, names, _ = followup_parent
    record = _measured_record(names[0])
    change(record)
    with pytest.raises(ValueError):
        b.validate_measurement(record)


def test_saved_measurement_uses_the_same_even_sample_median(followup_parent):
    b, names, _ = followup_parent
    record = _measured_record(names[0])
    record.update(rounds=2, timings_seconds={"stock": [0.038, 0.042], "candidate_guard_on": [0.019, 0.021]})
    assert b.validate_measurement(record)["execution_verified"]


def test_saved_measurement_allows_intentional_stock_neighbor():
    from benchmarks.micro import bench_review_followup as b

    name = "argmax:argmax:float64:2999x3000:none"
    record = _measured_record(name)
    record["dispatch"] = ["stock", "no-applicable-path"]
    assert b.validate_measurement(record)["execution_verified"]


def test_followup_records_withdrawn_inplace_add_as_intentional_stock(monkeypatch):
    from benchmarks.micro import bench_review_followup as b

    path = next(p for p in b.GEARBOX._paths["numpy.add"] if p.name == "pyrallel_add")
    predicate = b.parallel_binary._make_applicable({np.dtype("float64"): 1}, op_name="add")
    monkeypatch.setattr(path, "applicable", predicate)
    monkeypatch.setattr(b.parallel_binary._common, "core_ready", lambda: True)
    monkeypatch.setattr(b, "_sample", lambda call, reset: (reset(), call(), 0.02)[2])
    record = b.measure(b.Cell("out", "add", "float64", (16,), "inplace"), 2)
    assert record["dispatch"][0] == "stock"
    assert record["execution"]["intentional_stock"] is True
    assert record["execution"]["stock_reason"] == "float64-add-exact-inplace"
    record.update(fingerprint={"fingerprint": "test", "python": "test", "numpy": "test"},
                  conditions=b._conditions(0.01, 0.01))
    assert b.validate_measurement(record)["execution_verified"]


@pytest.mark.parametrize("dtype,alias,proof,accepted", [
    ("float64", "inplace", True, True), ("float64", "inplace", False, False),
    ("int64", "inplace", True, False), ("float64", "disjoint", True, False),
])
def test_only_explicit_withdrawn_add_stock_evidence_is_accepted(dtype, alias, proof, accepted):
    from benchmarks.micro import bench_review_followup as b

    name = next(n for n, c in b.cells().items()
                if c.family == "out" and c.op == "add" and c.dtype == dtype and c.alias == alias)
    record = _measured_record(name)
    # Earlier actual-candidate records retain their original valid meaning.
    assert b.validate_measurement(record)["execution_verified"]
    record["dispatch"] = ["stock", "no-applicable-path"]
    if proof:
        record["execution"].update(intentional_stock=True, stock_reason="float64-add-exact-inplace")
    if accepted:
        assert b.validate_measurement(record)["execution_verified"]
    else:
        with pytest.raises(ValueError):
            b.validate_measurement(record)
