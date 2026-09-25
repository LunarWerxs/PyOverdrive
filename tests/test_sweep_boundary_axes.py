"""Boundary rows describe real inputs without running timed kernels."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyoverdrive.fastpaths import hist2d_uniform, matmul_split_complex


_SPEC = importlib.util.spec_from_file_location(
    "boundary_sweep", Path(__file__).resolve().parents[1] / "tools/verify_no_pessimization.py")
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_histogram_rows_bracket_sample_floor(offset):
    rows = dict(sweep._axes_hist2d()["hist2d_uniform"])
    args, kwargs = rows[f"samples{offset:+d}"]()
    assert [a.shape for a in args] == [(6666 + offset,)] * 2
    assert hist2d_uniform._applicable(args, kwargs) == (offset >= 0)


@pytest.mark.parametrize("axis,offset,admitted", [
    ("m", -1, True), ("m", 1, False),
    ("n", -1, False), ("n", 1, True),
    ("q", -1, False), ("q", 1, True),
    ("m", 0, True),
])
def test_split_rows_have_coherent_boundary_shapes(monkeypatch, axis, offset, admitted):
    for name, value in (("M_MAX", 3), ("N_MIN", 4), ("Q_MIN", 5)):
        monkeypatch.setattr(matmul_split_complex, name, value)
    rows = dict(sweep._axes_split_complex()["matmul_split_complex"])
    label = "complex128" if offset == 0 else f"complex128-{axis}{offset:+d}"
    args, kwargs = rows[label]()
    dims = {"m": 3, "n": 4, "q": 5}
    dims[axis] += offset
    assert [a.shape for a in args] == [(dims["m"], dims["n"]), (dims["n"], dims["q"])]
    assert [a.dtype for a in args] == [np.dtype("complex128"), np.dtype("float64")]
    assert matmul_split_complex._applicable(args, kwargs) == admitted


def test_complex64_pair_is_withdrawn_even_inside_shape_gate(monkeypatch):
    monkeypatch.setattr(matmul_split_complex, "M_MAX", 3)
    monkeypatch.setattr(matmul_split_complex, "N_MIN", 4)
    monkeypatch.setattr(matmul_split_complex, "Q_MIN", 5)
    args = (np.ones((3, 4), dtype=np.complex64), np.ones((4, 5), dtype=np.float32))
    assert not matmul_split_complex._applicable(args, {})
    assert "complex64" not in dict(sweep._axes_split_complex()["matmul_split_complex"])


@pytest.mark.parametrize("m", [1, 16, 32, 63, 64, 65, 128, 192, 255, 256])
def test_fixed_matmul_row_grid_preserves_requested_m(monkeypatch, m):
    monkeypatch.setattr(matmul_split_complex, "N_MIN", 4)
    monkeypatch.setattr(matmul_split_complex, "Q_MIN", 5)
    rows = dict(sweep._axes_split_complex()["matmul_split_complex"])
    args, kwargs = rows[f"complex128-rows-m{m}"]()
    assert [a.shape for a in args] == [(m, 4), (4, 5)]
    assert matmul_split_complex._applicable(args, kwargs) == (m <= matmul_split_complex.M_MAX)


@pytest.mark.parametrize("axis,offset", [("n", -1), ("n", 1), ("q", -1), ("q", 1)])
def test_fixed_m64_grid_brackets_inner_and_output_floors(monkeypatch, axis, offset):
    monkeypatch.setattr(matmul_split_complex, "N_MIN", 4)
    monkeypatch.setattr(matmul_split_complex, "Q_MIN", 5)
    rows = dict(sweep._axes_split_complex()["matmul_split_complex"])
    args, kwargs = rows[f"complex128-rows-m64-{axis}{offset:+d}"]()
    n = 4 + (offset if axis == "n" else 0)
    q = 5 + (offset if axis == "q" else 0)
    assert [a.shape for a in args] == [(64, n), (n, q)]
    assert matmul_split_complex._applicable(args, kwargs) == (offset > 0)
