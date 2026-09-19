"""Robust timing qualification, using synthetic samples rather than clocks."""

import copy
import json
import statistics
import subprocess

import pytest

from benchmarks.micro import bench_review_followup as followup
from tools import verify_no_pessimization as sweep


def _records(ratio, noisy):
    samples = {"stock": [ratio] * 5, "patched": [1.0] * 5}
    base = samples[noisy][0]
    samples[noisy] = [base * x for x in (0.5, 0.5, 1.0, 1.5, 1.5)]
    medians = {k: statistics.median(v) for k, v in samples.items()}
    result = f"RATIO {ratio:.4f} numpy.sin"
    main = {"cell": "sample", "result": result, "op": "numpy.sin", "path": "sample",
            "dispatch": {"chosen": "sample", "fallback_observed": False},
            "correctness": {"passed": True}, "rounds": 5, "iterations_per_sample": 1,
            "ratio": ratio}
    for label in samples:
        main[f"{label}_seconds"] = samples[label]
        main[f"{label}_median_seconds"] = medians[label]
    record = {
        "cell": "vectorize:sin:float64:16:none", "status": "measured", "family": "vectorize",
        "operation": "sin", "dtype": "float64", "shape": [16], "alias": "none",
        "experimental_enable": "vectorize_ufunc_direct",
        "dispatch": ["vectorize_ufunc_direct", "predicate-accepted"],
        "execution": {"path": "vectorize_ufunc_direct", "check": "Gearbox._warned_paths",
                      "fallback_observed": False, "direct_ufunc": "sin"},
        "correctness": {"passed": True, "mode": "bit-identical"},
        "fingerprint": {"fingerprint": "test", "numpy": "test", "python": "test"},
        "conditions": followup._conditions(0.01, 0.01), "rounds": 5, "calls_per_sample": 1,
        "timings_seconds": {"stock": samples["stock"], "candidate_guard_on": samples["patched"]},
        "median_seconds": {"stock": medians["stock"], "candidate_guard_on": medians["patched"]},
        "ratios_stock_over_candidate": {"candidate_guard_on": ratio},
    }
    return main, record


@pytest.mark.parametrize("ratio", [0.5, 2.0])
@pytest.mark.parametrize("side", ["stock", "patched"])
def test_unstable_winners_and_losers_fail_both_record_validators_without_mutation(ratio, side):
    main, record = _records(ratio, side)
    before = copy.deepcopy((main, record))
    proc = subprocess.CompletedProcess([], 0, main["result"], "")
    with pytest.raises(RuntimeError, match="unstable timing"):
        sweep._validate_child_record(main, proc, "sample")
    with pytest.raises(RuntimeError, match="unstable timing"):
        followup.validate_measurement(record)
    assert (main, record) == before


def test_isolated_outlier_is_accepted_by_robust_dispersion():
    result = sweep.validate_timing_stability([1.0, 1.01, 1.02, 1.03, 99.0], "stock")
    assert result["relative_mad"] == pytest.approx(0.01 / 1.02)
    assert result["threshold"] == 0.20


def test_dispersion_boundary_is_inclusive_and_above_it_is_unverified():
    sweep.validate_timing_stability([0.8, 1.0, 1.2], "stock")
    with pytest.raises(RuntimeError, match="patched.*MAD/median"):
        sweep.validate_timing_stability([0.79, 1.0, 1.21], "patched")


def test_guard_off_samples_are_qualified_too():
    _, record = _records(2.0, "patched")
    record.update(cell="out:add:int64:16:inplace", family="out", operation="add", dtype="int64", alias="inplace",
                  experimental_enable="pyrallel_add", dispatch=["pyrallel_add", "predicate-accepted"],
                  execution={"path": "pyrallel_add", "check": "Gearbox._warned_paths", "fallback_observed": False},
                  guard_overhead_seconds=0.0)
    record["timings_seconds"]["candidate_guard_off"] = record["timings_seconds"]["candidate_guard_on"]
    record["timings_seconds"]["candidate_guard_on"] = [1.0] * 5
    record["median_seconds"]["candidate_guard_off"] = 1.0
    record["ratios_stock_over_candidate"]["candidate_guard_off"] = 2.0
    with pytest.raises(RuntimeError, match="candidate_guard_off.*unstable timing"):
        followup.validate_measurement(record)


@pytest.mark.parametrize("save_evidence", [False, True])
def test_live_measurement_rejects_noise_even_without_json(monkeypatch, save_evidence):
    from types import SimpleNamespace

    stock = lambda: sweep.np.array([1.0])
    monkeypatch.setattr(sweep, "_resolve", lambda op: (SimpleNamespace(op=stock), "op"))
    monkeypatch.setattr(sweep.GEARBOX, "stock_fn", lambda op: stock)
    times = iter([0.03, 0.05, 0.025, 0.1, 0.025, 0.15, 0.025])
    monkeypatch.setattr(sweep.timeit, "timeit", lambda *a, **k: next(times))
    evidence = {} if save_evidence else None
    with pytest.raises(RuntimeError, match="stock.*unstable timing"):
        sweep._measure("numpy.op", (), {}, 3, evidence=evidence)
    if save_evidence:
        assert evidence["stock_seconds"] == [0.05, 0.1, 0.15]


def test_parent_retains_unstable_raw_evidence_for_diagnosis(tmp_path):
    from types import SimpleNamespace

    record, _ = _records(2.0, "patched")
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(record))
    args = SimpleNamespace(_evidence={"cells": []})
    proc = subprocess.CompletedProcess([], 0, record["result"], "")
    with pytest.raises(RuntimeError, match="unstable timing"):
        sweep._collect_child_evidence(args, path, "sample", "initial", 1, proc)
    saved = args._evidence["cells"][0]
    assert saved["patched_seconds"] == record["patched_seconds"]
    assert "unstable timing" in saved["error"]


def test_followup_child_never_qualifies_unstable_raw_record(tmp_path, monkeypatch):
    from types import SimpleNamespace

    _, record = _records(2.0, "patched")
    monkeypatch.setattr(followup, "measure", lambda *a: record)
    monkeypatch.setattr(followup.sweep, "_fingerprint", lambda: record["fingerprint"])
    monkeypatch.setattr(followup.sweep, "_foreign_load", lambda: 0.01)
    output = tmp_path / "child.json"
    args = SimpleNamespace(one=record["cell"], require_quiet=True, fast_under=None, rounds=5, json=output)
    assert followup._child(args) == 1
    saved = json.loads(output.read_text())
    assert saved["status"] == "error"
    assert saved["qualified"] is False
    assert saved["timings_seconds"] == record["timings_seconds"]
    assert "unstable timing" in saved["error"]
