"""The demo must never lie or crash: every case it shows dispatches through
a real fast path, and both sides of its comparison run the same call."""

from __future__ import annotations

import io

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.demo import _Stock, _cases, _fmt_t
from pyoverdrive.dispatcher.gearbox import GEARBOX


def test_every_demo_case_runs_both_sides_and_dispatches():
    pyoverdrive.enable()
    try:
        stock = _Stock()
        for label, call in _cases(quick=True):
            r_stock = call(stock)
            r_over = call(np)
            rs, ro = np.asarray(r_stock), np.asarray(r_over)
            assert ro.shape == rs.shape and ro.dtype == rs.dtype, label
    finally:
        pyoverdrive.disable()


def test_demo_cases_actually_dispatch():
    # a demo row that silently runs stock-vs-stock would show 1.0x and lie
    # by omission; every case must route to a fast path at quick sizes too
    op_of = {
        "convolve": "numpy.convolve",
        "unique": "numpy.unique",
        "nanquantile": "numpy.nanquantile",
        "intersect1d": "numpy.intersect1d",
        "einsum": "numpy.einsum",
        "sort": "numpy.sort",
        "mean": "numpy.mean",
        "linalg.eigvalsh": "numpy.linalg.eigvalsh",
        "linalg.inv": "numpy.linalg.inv",
        "isin": "numpy.isin",
        "matmul": "numpy.matmul",
        "nanmean": "numpy.nanmean",
        "sin": "numpy.sin",
    }

    class Recorder:
        def __init__(self, calls=None, prefix=""):
            self.calls = calls if calls is not None else []
            self._prefix = prefix

        def __getattr__(self, name):
            dotted = f"{self._prefix}{name}"
            if dotted not in op_of:
                return Recorder(self.calls, f"{dotted}.")  # submodule access

            def f(*args, **kwargs):
                self.calls.append((op_of[dotted], args, kwargs))
                return GEARBOX.stock_fn(op_of[dotted])(*args, **kwargs)

            return f

    rec = Recorder()
    for label, call in _cases(quick=True):
        call(rec)
    assert len(rec.calls) == len(_cases(quick=True))
    for op, args, kwargs in rec.calls:
        chosen, reason = GEARBOX.decide(op, args, kwargs)
        assert chosen != "stock", (op, reason)


def test_fmt_t_ranges():
    assert _fmt_t(2.5).endswith("s ")
    assert _fmt_t(0.002).endswith("ms")
    assert _fmt_t(3e-6).endswith("us")


def test_main_demo_flag_wiring(monkeypatch):
    from pyoverdrive import __main__ as main_mod

    called = {}

    def fake_demo(quick=False, file=None):
        called["quick"] = quick
        return 0

    monkeypatch.setattr("pyoverdrive.demo.demo", fake_demo)
    assert main_mod.main(["--demo", "--quick"]) == 0
    assert called["quick"] is True
