"""Smoke runs of the batch13 calibration battery must not write results.

Checked from the source tree: importing the battery would run its measurements.
"""

import ast
from pathlib import Path

BATTERY = Path(__file__).resolve().parents[1] / "benchmarks" / "micro" / "bench_batch13_calibration.py"


def _save_calls(tree):
    """Yield (call, guarded) for each `suite.save()` at module level, guarded by `if not SMOKE:`."""
    def is_save(node):
        return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "save")

    def is_not_smoke(test):
        return (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Name) and test.operand.id == "SMOKE")

    for node in tree.body:
        if is_save(node):
            yield node, False
        elif isinstance(node, ast.If) and is_not_smoke(node.test):
            for inner in node.body:
                if is_save(inner):
                    yield inner, True


def test_batch13_save_is_skipped_in_smoke_mode():
    tree = ast.parse(BATTERY.read_text(encoding="utf-8"))
    saves = list(_save_calls(tree))
    assert saves, "battery must still save its results on a full run"
    assert all(guarded for _, guarded in saves), "suite.save() must sit under `if not SMOKE:`"
