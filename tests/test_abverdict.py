"""A/B verdicts: noise must come out "inconclusive", never as a win.

Synthetic timings only, so every expectation here is exact arithmetic and
no clock is involved.
"""

from __future__ import annotations

import json

import pytest

from pyoverdrive import abverdict
from tools import ab_compare

FAST = [5.0, 5.05, 4.95, 5.02, 4.98]
SLOW = [10.0, 10.1, 9.9, 10.05, 9.95]


def test_t_distribution_matches_the_published_table():
    # A wrong incomplete beta would silently widen or narrow every interval.
    assert abverdict.t_sf(0.0, 5) == pytest.approx(0.5)
    assert abverdict.t_ppf(0.975, 10) == pytest.approx(2.2281, abs=1e-3)
    assert abverdict.t_ppf(0.975, 2) == pytest.approx(4.3027, abs=1e-3)


def test_holm_step_down_is_monotone_and_scaled_by_rank():
    assert abverdict.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_separated_runs_give_a_real_difference_in_both_directions():
    assert abverdict.compare(SLOW, FAST, min_speedup=1.3)["verdict"] == "faster"
    assert abverdict.compare(FAST, SLOW)["verdict"] == "slower"
    # A real 2x speedup is still not a 3x one: conclusive, but below the bar.
    assert abverdict.compare(SLOW, FAST, min_speedup=3.0)["verdict"] == "no-win"


def test_fewer_than_three_runs_is_inconclusive_however_large_the_delta():
    got = abverdict.compare([10.0, 10.0], [1.0, 1.0])
    assert got["verdict"] == "inconclusive"
    assert "fewer than 3" in got["reason"]


def test_overlapping_per_run_ranges_are_inconclusive_despite_a_big_median_gap():
    # Medians say 2.2x; one baseline run is faster than one candidate run.
    got = abverdict.compare([10.0, 20.0, 30.0, 22.0, 25.0], [8.0, 9.0, 11.0, 7.0, 10.0],
                            min_speedup=1.3)
    assert got["verdict"] == "inconclusive"


def test_unusable_samples_are_invalid():
    assert abverdict.compare([1.0, float("nan"), 1.0], FAST)["verdict"] == "invalid"


def _report(tmp_path, name, *, numpy="2.3.0", schema=1, stock=SLOW, patched=FAST):
    report = {
        "schema_version": schema, "tool": "verify_no_pessimization", "completed": True,
        "fingerprint": {"fingerprint": "abc", "numpy": numpy, "python": "3.12.0"},
        "settings": {"rounds": 5, "sample_seconds": 0.02, "any_core": False,
                     "environment": {}},
        "calibration": {"saved": {}},
        "conditions": {"contended": False, "load_known": True},
        "source": {"sha256": "s1"},
        "cells": [{"cell": "roll", "returncode": 0, "stock_seconds": stock,
                   "patched_seconds": patched}],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


def test_cli_supports_a_claim_backed_by_three_matching_runs(tmp_path):
    runs = [_report(tmp_path, f"r{i}", stock=[s * (1 + i / 100) for s in SLOW])
            for i in range(3)]
    assert ab_compare.main(["ab_compare", *runs, "--min-speedup", "1.3",
                            "--require-win"]) == 0


def test_cli_refuses_a_schema_bump_and_a_foreign_environment(tmp_path, capsys):
    good = [_report(tmp_path, f"r{i}") for i in range(2)]
    assert ab_compare.main(["ab_compare", *good,
                            _report(tmp_path, "bumped", schema=2)]) == 2
    assert "INVALID" in capsys.readouterr().err
    assert ab_compare.main(["ab_compare", *good,
                            _report(tmp_path, "other", numpy="2.4.0")]) == 2
    assert "machine.numpy" in capsys.readouterr().err
