"""Operation-specific output gates use tiny buffers, never benchmark arrays."""

import numpy as np
import pytest

from pyoverdrive.dispatcher.gearbox import Gearbox
from pyoverdrive.fastpaths import parallel_binary as binary


@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("view", [False, True])
def test_float64_add_exact_alias_uses_stock(operand, view, monkeypatch):
    monkeypatch.setattr(binary, "SUPPORTED", {"add": {np.dtype("float64"): 1}})
    monkeypatch.setattr(binary._common, "core_ready", lambda: True)
    box = Gearbox()
    binary.register(box)
    a, b = np.arange(16.0), np.ones(16)
    inputs = (a, b)
    out = inputs[operand].view() if view else inputs[operand]
    expected = np.add(a, b)
    assert box.decide("numpy.add", inputs, {"out": out})[0] == "stock"
    got = box._make_wrapper("numpy.add", np.add)(*inputs, out=out)
    assert got is out
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("op,dtype,alias", [
    ("add", "float64", None), ("add", "float64", "disjoint"),
    ("add", "int64", "inplace"), ("maximum", "float64", "inplace"),
])
def test_unaffected_binary_regimes_keep_dispatch(op, dtype, alias, monkeypatch):
    monkeypatch.setattr(binary, "SUPPORTED", {op: {np.dtype(dtype): 1}})
    monkeypatch.setattr(binary._common, "core_ready", lambda: True)
    box = Gearbox()
    binary.register(box)
    base = np.arange(32, dtype=dtype)
    a, b = base[:16], np.ones(16, dtype=dtype)
    kwargs = {} if alias is None else {"out": a if alias == "inplace" else base[16:]}
    assert box.decide(f"numpy.{op}", (a, b), kwargs)[0] == f"pyrallel_{op}"
