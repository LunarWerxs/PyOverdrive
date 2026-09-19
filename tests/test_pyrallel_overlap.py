"""Overlapping output views must retain stock NumPy's buffering semantics."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pyoverdrive.fastpaths import parallel_binary, parallel_ufunc
from pyoverdrive.parallel import pyrallel


@pytest.mark.parametrize("family", ["unary", "binary"])
@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("direction", [1, -1])
def test_predicate_refuses_shifted_overlap(family, operand, direction, monkeypatch):
    monkeypatch.setattr(pyrallel, "max_threads", lambda: 4)
    base = np.arange(20, dtype=np.float64)
    x, out = (base[:16], base[4:]) if direction == 1 else (base[4:], base[:16])
    inputs = [x] if family == "unary" else [np.ones(16), np.ones(16)]
    inputs[0 if family == "unary" else operand] = x
    module = parallel_ufunc if family == "unary" else parallel_binary
    applicable = module._make_applicable({np.dtype("float64"): 1})
    assert not applicable(tuple(inputs), {"out": out})


@pytest.mark.parametrize("family", ["unary", "binary"])
@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("output_kind", ["same", "same_view", "disjoint_view"])
def test_predicate_accepts_safe_output_aliases(family, operand, output_kind, monkeypatch):
    monkeypatch.setattr(pyrallel, "max_threads", lambda: 4)
    base = np.arange(32, dtype=np.float64)
    x = base[:16]
    outputs = {"same": x, "same_view": x.view(), "disjoint_view": base[16:]}
    inputs = [x] if family == "unary" else [np.ones(16), np.ones(16)]
    inputs[0 if family == "unary" else operand] = x
    module = parallel_ufunc if family == "unary" else parallel_binary
    applicable = module._make_applicable({np.dtype("float64"): 1})
    assert applicable(tuple(inputs), {"out": outputs[output_kind]})


@pytest.mark.parametrize("operand", [None, 0, 1])
def test_core_shifted_overlap_matches_stock(operand, monkeypatch):
    monkeypatch.setattr(pyrallel, "max_threads", lambda: 4)
    base = np.arange(20, dtype=np.float64)
    x, out = base[:16], base[4:]
    inputs = [x] if operand is None else [np.ones(16), np.ones(16)]
    inputs[0 if operand is None else operand] = x
    ufunc = np.sin if operand is None else np.add
    expected = ufunc(*(a.copy() for a in inputs))
    # One worker is a valid executor schedule: it makes the cross-chunk
    # corruption deterministic instead of relying on a race to reproduce.
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pyrallel, "_pool", lambda: executor)
        got = pyrallel.parallel_elementwise(ufunc, tuple(inputs), 4, out=out)
    assert got is out
    np.testing.assert_array_equal(got, expected)
