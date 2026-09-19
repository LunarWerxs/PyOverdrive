"""Sweep limits must reject oversized inputs before allocating them."""

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1] / "tools" / "verify_no_pessimization.py"
_SPEC = importlib.util.spec_from_file_location("allocation_limits_sweep", _PATH)
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)


def test_oversized_scale_aware_cell_does_not_invoke_maker(monkeypatch):
    monkeypatch.setattr(sweep, "MAX_ELEMENTS", 11)

    def make(scale=1):
        pytest.fail("oversized maker was invoked")

    assert sweep._cell_variants("sample*3", (np.arange(2), np.arange(2)), make) == (
        None, "SKIP too-big")


def test_scaled_checks_combined_size_before_allocating_first_operand(monkeypatch):
    monkeypatch.setattr(sweep, "MAX_ELEMENTS", 20)
    monkeypatch.setattr(sweep, "_resample_grown", lambda *a: pytest.fail("resample allocated"))
    assert sweep._scaled((np.arange(6), np.arange(6)), 2) is None


def test_downscale_checks_target_size_before_copy(monkeypatch):
    arrays = (np.arange(18).reshape(9, 2), np.arange(18).reshape(9, 2))
    monkeypatch.setattr(sweep, "MAX_ELEMENTS", 11)
    monkeypatch.setattr(sweep.np, "ascontiguousarray", lambda *a: pytest.fail("slice copied"))
    assert sweep._scaled(arrays, 1, 3) is None


@pytest.mark.parametrize("factor", [np.int64(2**62), 10**100])
def test_huge_multiplier_is_rejected_without_overflow_or_allocation(monkeypatch, factor):
    monkeypatch.setattr(sweep, "_resample_grown", lambda *a: pytest.fail("resample allocated"))
    assert sweep._scaled((np.arange(4),), factor) is None


def test_zero_element_shape_still_checks_dimension_range(monkeypatch):
    monkeypatch.setattr(sweep, "_resample_grown", lambda *a: pytest.fail("resample allocated"))
    assert sweep._scaled((np.empty((2, 0)),), 10**100) is None


def test_scale_cap_boundary_preserves_builder_result(monkeypatch):
    monkeypatch.setattr(sweep, "MAX_ELEMENTS", 6)
    rebuilt = (np.arange(6),)
    calls = []

    def make(scale=1):
        calls.append(scale)
        return rebuilt, {}

    variants, skip = sweep._cell_variants("sample*3", (np.arange(2),), make)
    assert skip is None
    assert calls == [3]
    assert variants[0] is rebuilt


def test_resample_cap_boundary_preserves_untouched_operands_and_layout(monkeypatch):
    source = np.asfortranarray(np.arange(12).reshape(4, 3))
    parameter = np.arange(2)
    monkeypatch.setattr(sweep, "MAX_ELEMENTS", 24)
    expected = sweep._resample_grown(source, 2)
    got = sweep._scaled((source, parameter, "unchanged"), 2)
    np.testing.assert_array_equal(got[0], expected)
    assert got[0].flags.f_contiguous
    assert got[1] is parameter
    assert got[2] == "unchanged"


def test_downscale_preserves_existing_slice_semantics_at_cap(monkeypatch):
    source = np.arange(24).reshape(6, 4)
    monkeypatch.setattr(sweep, "MAX_ELEMENTS", 8)
    got = sweep._scaled((source,), 1, 3)
    np.testing.assert_array_equal(got[0], [[0, 1, 2, 3], [4, 5, 6, 7]])
    assert got[0].flags.c_contiguous


def test_current_scale_aware_makers_have_linear_element_counts(monkeypatch):
    from pyoverdrive.fastpaths import intersect_sorted, unique_sort

    monkeypatch.setattr(unique_sort, "_THRESHOLDS", {np.dtype("int64"): 4})
    monkeypatch.setattr(intersect_sorted, "_THRESHOLDS", {np.dtype("int64"): 5})
    axes = {**sweep._axes_unique_sort(), **sweep._axes_intersect()}
    for rows in axes.values():
        for _, make in rows:
            base, _ = make()
            grown, _ = make(scale=3)
            assert sum(a.size for a in grown) == 3 * sum(a.size for a in base)


@pytest.mark.parametrize("args,factor,cap", [
    ((np.arange(4), np.arange(4)), 4, 16),
    ((np.arange(6).reshape(2, 3),), 4, 11),
    ((np.arange(6).reshape(2, 3), np.arange(6).reshape(3, 2)), 4, 17),
])
def test_oversized_shape_cell_is_skipped_before_any_growth(monkeypatch, args, factor, cap):
    monkeypatch.setattr(sweep, "MAX_ELEMENTS", cap)
    monkeypatch.setattr(sweep, "_grow_axis", lambda *a: pytest.fail("shape growth allocated"))
    monkeypatch.setattr(sweep.np, "concatenate", lambda *a, **k: pytest.fail("concatenate allocated"))
    assert sweep._cell_variants(f"sample>{factor}", args, lambda: None) == (None, "SKIP too-big")


def test_shape_growth_crops_first_to_avoid_oversized_intermediate(monkeypatch):
    source = np.asfortranarray(np.arange(16).reshape(8, 2))
    monkeypatch.setattr(sweep, "MAX_ELEMENTS", 16)
    original_grow = sweep._grow_axis

    def grow(a, factor, axis):
        assert a.shape == (2, 2), "uncropped intermediate would exceed cap"
        return original_grow(a, factor, axis)

    monkeypatch.setattr(sweep, "_grow_axis", grow)
    variants = sweep._reshaped((source,), 4, True)
    assert len(variants) == 1
    np.testing.assert_array_equal(variants[0][0], np.tile(source[:2], (1, 4)))
    assert variants[0][0].flags.c_contiguous


def test_chain_shape_preserves_values_with_cropping_before_growth(monkeypatch):
    a, b = np.arange(24).reshape(8, 3), np.arange(24).reshape(3, 8)
    monkeypatch.setattr(sweep, "MAX_ELEMENTS", 48)
    na, nb = sweep._reshaped_chain((a, b), 4, True)
    np.testing.assert_array_equal(na, np.tile(a[:2], (1, 4)))
    np.testing.assert_array_equal(nb, np.tile(b[:, :2], (4, 1)))


def test_chain_shape_preserves_singleton_strides_from_fortran_source():
    a = np.asfortranarray(np.arange(3, dtype=np.int64).reshape(1, 3))
    b = np.asfortranarray(np.arange(12, dtype=np.int64).reshape(3, 4))
    _, nb = sweep._reshaped_chain((a, b), 4, True)
    assert nb.shape == (12, 1)
    assert nb.strides == (8, 96)


def _selection_args(path, **changes):
    values = dict(sizes=True, shapes=False, rows=False, values=False, only=[], cells_file=path)
    values.update(changes)
    return SimpleNamespace(**values)


def _stub_cells(monkeypatch):
    monkeypatch.setattr(sweep, "cells", lambda *a, **k: ["sample", "sample*3", "other@float32"])
    monkeypatch.setattr(sweep.pyoverdrive, "disable", lambda: None)


def test_exact_cells_file_selects_canonical_names_without_substring_matching(monkeypatch, tmp_path):
    _stub_cells(monkeypatch)
    path = tmp_path / "remaining.txt"
    path.write_text("other@float32\nsample*3\n", encoding="utf-8")
    names, code = sweep._select_cells(_selection_args(path), {"sample", "other"})
    assert code is None
    assert names == ["sample*3", "other@float32"]


@pytest.mark.parametrize("content", ["missing\n", "\n \n"])
def test_exact_cells_file_rejects_unknown_or_empty_selection(monkeypatch, tmp_path, content):
    _stub_cells(monkeypatch)
    path = tmp_path / "remaining.txt"
    path.write_text(content, encoding="utf-8")
    names, code = sweep._select_cells(_selection_args(path), {"sample", "other"})
    assert code == 1
    assert names == []


def test_cells_file_evidence_records_raw_hash_and_requested_names(monkeypatch, tmp_path):
    _stub_cells(monkeypatch)
    monkeypatch.setattr(sweep, "_foreign_load", lambda: 0.10)
    monkeypatch.setattr(sweep, "_fingerprint", lambda: {})
    monkeypatch.setattr(sweep, "_source_revision", lambda: None)
    path = tmp_path / "remaining.txt"
    content = b"sample*3\nother@float32\n"
    path.write_bytes(content)

    def run(args):
        names, code = sweep._select_cells(args, {"sample", "other"})
        assert code is None
        args._evidence["selected_cells"] = names
        return 1

    monkeypatch.setattr(sweep, "_run_sweep", run)
    output = tmp_path / "evidence.json"
    assert sweep.main(["tool", "--sizes", "--cells-file", str(path), "--json", str(output)]) == 1
    evidence = json.loads(output.read_text())
    assert evidence["selection"]["cells_file"] == {
        "sha256": hashlib.sha256(content).hexdigest(),
        "requested": ["sample*3", "other@float32"],
    }
    assert evidence["selected_cells"] == ["sample*3", "other@float32"]
