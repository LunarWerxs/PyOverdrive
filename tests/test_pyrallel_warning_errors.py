"""Warnings promoted to exceptions must never replay an in-place operation."""

import warnings

import numpy as np
import pytest

from pyoverdrive.dispatcher.gearbox import FastPath, Gearbox
from pyoverdrive.fastpaths import _pyrallel_common, parallel_binary, parallel_ufunc
from pyoverdrive.parallel import pyrallel


@pytest.mark.parametrize("op", ["sin", "add"])
@pytest.mark.parametrize("filter_kind", ["all", "message"])
def test_warning_error_preserves_stock_exception_and_inplace_result(op, filter_kind, monkeypatch):
    monkeypatch.setattr(pyrallel, "max_threads", lambda: 4)
    monkeypatch.setattr(_pyrallel_common, "threads_for", lambda n: 4)
    monkeypatch.setattr(parallel_ufunc, "threads_for", lambda n: 4)
    module = parallel_ufunc if op == "sin" else parallel_binary
    stock = getattr(np, op)
    gearbox = Gearbox()
    gearbox.register(FastPath(
        name=f"pyrallel_{op}", op=f"numpy.{op}",
        applicable=module._make_applicable({np.dtype("float64"): 1}),
        run=module._make_run(gearbox, f"numpy.{op}"),
    ))
    wrapped = gearbox._make_wrapper(f"numpy.{op}", stock)
    x = np.ones(16)
    x[0] = np.inf
    inputs = [x]
    if op == "add":
        y = np.ones(16)
        y[0] = -np.inf
        inputs.append(y)
    expected_inputs = [a.copy() for a in inputs]
    message = f"invalid value encountered in {op}"
    try:
        with np.errstate(invalid="warn"), warnings.catch_warnings():
            if filter_kind == "all":
                warnings.simplefilter("error")
            else:
                warnings.filterwarnings("ignore", message="PyOverdrive fast path")
                warnings.filterwarnings("error", message=message)
            with pytest.raises(RuntimeWarning, match=f"^{message}$"):
                stock(*expected_inputs, out=expected_inputs[0])
            with pytest.raises(RuntimeWarning, match=f"^{message}$"):
                wrapped(*inputs, out=x)
        np.testing.assert_array_equal(x, expected_inputs[0])
    finally:
        pyrallel.shutdown()
