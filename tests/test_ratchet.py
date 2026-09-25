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


def test_metric_is_worst_valid_cell_of_exactly_this_path():
    evidence = {"cells": [
        {"cell": "unique_sort", "path": "unique_sort", "ratio": 1.8, "returncode": 0},
        {"cell": "unique_sort@x10", "path": "unique_sort", "ratio": 1.3, "returncode": 0},
        # substring match from --only: a different path, must not count
        {"cell": "unique_sort_wide", "path": "unique_sort_wide", "ratio": 0.5, "returncode": 0},
        # errored and crashed cells are not measurements
        {"cell": "unique_sort@x30", "path": "unique_sort", "ratio": 0.4, "error": "unstable"},
        {"cell": "unique_sort/3", "path": "unique_sort", "ratio": 0.3, "returncode": 1},
    ]}
    metric, cells = ratchet.metric_from_evidence(evidence, "unique_sort")
    assert metric == pytest.approx(1.3)
    assert len(cells) == 2


def test_no_cells_for_the_path_is_no_metric():
    assert ratchet.metric_from_evidence({"cells": []}, "unique_sort") == (None, [])


@pytest.mark.parametrize("branch", [None, "main", "master"])
def test_refuses_to_reset_detached_or_primary_branches(branch):
    assert ratchet.branch_refusal(branch) is not None


def test_allows_a_search_branch():
    assert ratchet.branch_refusal("search/unique-sort") is None
