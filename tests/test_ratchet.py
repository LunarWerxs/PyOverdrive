"""The keep-or-revert rule of tools/ratchet.py, on synthetic numbers.

WHY: the ratchet resets a branch unattended, so its judgement is the one
thing that must not drift: a candidate slower than stock anywhere, or a
tiny gain bought with added code, must never survive, and another path's
cell must never stand in for the candidate's.
"""

import pytest

from tools import ratchet


def _decide(metric, incumbent=1.0, net_lines=10):
    return ratchet.decide(metric, incumbent, net_lines, min_gain=0.02, hold=0.01)[0]


def test_keeps_a_clear_gain_over_the_incumbent():
    assert _decide(1.30, incumbent=1.20) == "keep"


def test_new_path_must_beat_stock_by_the_margin():
    assert _decide(1.01) == "revert"
    assert _decide(1.05) == "keep"


def test_any_cell_slower_than_stock_reverts_even_when_simpler():
    assert _decide(0.99, incumbent=0.99, net_lines=-40) == "revert"


def test_tiny_gain_with_added_code_reverts():
    assert _decide(1.21, incumbent=1.20, net_lines=30) == "revert"


def test_deletion_that_holds_the_metric_is_kept():
    assert _decide(1.195, incumbent=1.20, net_lines=-12) == "keep"
    assert _decide(1.10, incumbent=1.20, net_lines=-12) == "revert"


def test_unmeasured_candidate_reverts():
    assert _decide(None) == "revert"


def _sweep(*cells, completed=True):
    return {"cells": list(cells), "completed": completed}


def test_metric_is_worst_cell_of_exactly_this_path():
    evidence = _sweep(
        {"cell": "unique_sort", "path": "unique_sort", "ratio": 1.8, "returncode": 0},
        {"cell": "unique_sort*10", "path": "unique_sort", "ratio": 1.3, "returncode": 0},
        # substring match from --only: a different path, must not count
        {"cell": "unique_sort_wide", "path": "unique_sort_wide", "ratio": 0.5, "returncode": 0},
    )
    metric, cells = ratchet.metric_from_evidence(evidence, "unique_sort")
    assert metric == pytest.approx(1.3)
    assert len(cells) == 2
    assert ratchet.sweep_failure(evidence, "unique_sort", 0, metric) is None


def test_confirmation_reading_overrules_a_noisy_first_reading():
    # The sweep passes a cell whose confirmation run reads 1.0x or more.
    evidence = _sweep(
        {"cell": "unique_sort", "ratio": 0.95, "returncode": 0, "phase": "initial"},
        {"cell": "unique_sort", "ratio": 1.25, "returncode": 0, "phase": "confirmation"},
    )
    metric, _ = ratchet.metric_from_evidence(evidence, "unique_sort")
    assert metric == pytest.approx(1.25)


@pytest.mark.parametrize("failed", [
    # wrong result in one cell: the child fails the sweep's correctness check
    {"cell": "unique_sort%d", "returncode": 1,
     "error": "RuntimeError('correctness check failed for unique')"},
    # crashed child: its record carries no "path" key and no ratio
    {"cell": "unique_sort/3", "error": "missing/invalid child evidence: gone"},
    # unstable reading: returncode 0 but the record carries an error
    {"cell": "unique_sort@float32", "path": "unique_sort", "ratio": 1.9, "returncode": 0,
     "error": "unverified child evidence: unstable timing"},
])
def test_an_errored_or_crashed_cell_of_the_path_reverts(failed):
    evidence = _sweep(
        {"cell": "unique_sort", "path": "unique_sort", "ratio": 1.8, "returncode": 0},
        failed, completed=False)
    metric, _ = ratchet.metric_from_evidence(evidence, "unique_sort")
    reason = ratchet.sweep_failure(evidence, "unique_sort", 1, metric)
    assert reason is not None and failed["cell"] in reason


def test_an_incomplete_or_failed_sweep_reverts():
    good = {"cell": "unique_sort", "path": "unique_sort", "ratio": 1.8, "returncode": 0}
    assert "incomplete" in ratchet.sweep_failure(
        {"cells": [good], "completed": False, "error": "KeyboardInterrupt()"},
        "unique_sort", 1, 1.8)
    assert "incomplete" in ratchet.sweep_failure({}, "unique_sort", 1, None)
    assert "exit 1" in ratchet.sweep_failure(_sweep(good), "unique_sort", 1, 1.8)
    # a measured loss is the one nonzero exit left to decide(), which names it
    loss = {"cell": "unique_sort", "path": "unique_sort", "ratio": 0.9, "returncode": 0}
    assert ratchet.sweep_failure(_sweep(loss), "unique_sort", 1, 0.9) is None


def test_contention_after_the_run_is_inconclusive():
    assert ratchet.quiet_refusal({"conditions": {"contended": True, "load_known": True}})
    assert ratchet.quiet_refusal({"error": "quiet run refused: foreign CPU load is 0.5"})
    assert ratchet.quiet_refusal({"conditions": {"contended": False, "load_known": True}}) is None


def test_no_cells_for_the_path_is_no_metric():
    assert ratchet.metric_from_evidence({"cells": []}, "unique_sort") == (None, [])


@pytest.mark.parametrize("branch", [None, "main", "master"])
def test_refuses_to_reset_detached_or_primary_branches(branch):
    assert ratchet.branch_refusal(branch) is not None


def test_allows_a_search_branch():
    assert ratchet.branch_refusal("search/unique-sort") is None
