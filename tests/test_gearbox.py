"""Unit tests for the Gearbox dispatcher (no numpy patching side effects leak)."""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import FastPath, Gearbox


@pytest.fixture(autouse=True)
def _always_unpatched():
    yield
    pyoverdrive.disable()


def make_box():
    box = Gearbox()
    calls = []
    box.register(
        FastPath(
            name="double_small",
            op="numpy.positive",
            applicable=lambda a, k: len(a) == 1 and not k and a[0].size <= 4,
            run=lambda x: calls.append("fast") or x * 2,  # wrong on purpose: observable
        )
    )
    return box, calls


def test_dispatch_prefers_applicable_path():
    box, calls = make_box()
    out = box.dispatch("numpy.positive", np.positive, (np.arange(3),), {})
    assert calls == ["fast"]
    assert np.array_equal(out, np.arange(3) * 2)


def test_dispatch_falls_back_when_predicate_rejects():
    box, calls = make_box()
    big = np.arange(100)
    out = box.dispatch("numpy.positive", np.positive, (big,), {})
    assert calls == []
    assert np.array_equal(out, big)


def test_dispatch_falls_back_when_predicate_raises():
    box = Gearbox()
    box.register(
        FastPath(
            name="broken_pred",
            op="numpy.positive",
            applicable=lambda a, k: 1 / 0,
            run=lambda x: x,
        )
    )
    x = np.arange(3)
    assert np.array_equal(
        box.dispatch("numpy.positive", np.positive, (x,), {}), x
    )


def test_dispatch_falls_back_and_warns_when_run_raises():
    box = Gearbox()
    box.register(
        FastPath(
            name="explodes",
            op="numpy.positive",
            applicable=lambda a, k: True,
            run=lambda x: (_ for _ in ()).throw(ValueError("boom")),
        )
    )
    x = np.arange(3)
    with pytest.warns(RuntimeWarning, match="explodes"):
        out = box.dispatch("numpy.positive", np.positive, (x,), {})
    assert np.array_equal(out, x)
    # warning fires once per path, not per call
    out2 = box.dispatch("numpy.positive", np.positive, (x,), {})
    assert np.array_equal(out2, x)


def test_kill_switch():
    box, calls = make_box()
    box.set_path_enabled("double_small", False)
    out = box.dispatch("numpy.positive", np.positive, (np.arange(3),), {})
    assert calls == []
    assert np.array_equal(out, np.arange(3))
    assert box.decide("numpy.positive", (np.arange(3),), {}) == (
        "stock",
        "no-applicable-path",
    )


def test_decide_reason_codes():
    box, _ = make_box()
    assert box.decide("numpy.positive", (np.arange(2),), {}) == (
        "double_small",
        "predicate-accepted",
    )
    assert box.decide("numpy.unique", (np.arange(2),), {}) == (
        "stock",
        "operation-not-supported",
    )


def test_duplicate_name_rejected():
    box, _ = make_box()
    with pytest.raises(ValueError, match="duplicate"):
        box.register(
            FastPath(
                name="double_small",
                op="numpy.positive",
                applicable=lambda a, k: False,
                run=lambda x: x,
            )
        )


def test_patch_unpatch_restores_exactly():
    stock = np.positive
    pyoverdrive.enable(["numpy.positive"])
    assert np.positive is not stock
    assert getattr(np.positive, "__pyoverdrive__", False)
    assert pyoverdrive.enabled()
    pyoverdrive.enable(["numpy.positive"])  # idempotent
    pyoverdrive.disable()
    assert np.positive is stock
    assert not pyoverdrive.enabled()


def test_noop_path_does_not_recurse_and_matches_stock():
    pyoverdrive.enable(["numpy.positive"])
    pyoverdrive.enable_path("noop_positive")
    try:
        x = np.array([1.5, -2.5, 0.0])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any fallback warning = failure
            out = np.positive(x)
        assert np.array_equal(out, x)
        assert pyoverdrive.explain("numpy.positive", x) == (
            "noop_positive",
            "predicate-accepted",
        )
    finally:
        pyoverdrive.disable_path("noop_positive")


def test_kill_switch_reaches_the_patched_wrapper():
    """set_path_enabled must rebuild the per-op active list the wrapper
    closes over, so a disabled path stops dispatching without re-patching."""
    import numpy as np
    from pyoverdrive.dispatcher.gearbox import Gearbox, FastPath

    g = Gearbox()
    calls = []
    g.register(FastPath(
        name="spy",
        op="numpy.positive",
        applicable=lambda args, kwargs: True,
        run=lambda x: (calls.append("spy"), g.stock_fn("numpy.positive")(x))[1],
    ))
    g.patch(["numpy.positive"])
    try:
        x = np.arange(3)
        np.positive(x)
        assert calls == ["spy"]
        g.set_path_enabled("spy", False)
        np.positive(x)
        assert calls == ["spy"], "disabled path still dispatched"
        assert g.decide("numpy.positive", (x,), {}) == ("stock", "no-applicable-path")
        g.set_path_enabled("spy", True)
        np.positive(x)
        assert calls == ["spy", "spy"]
    finally:
        g.unpatch()
    assert isinstance(np.positive, np.ufunc)
