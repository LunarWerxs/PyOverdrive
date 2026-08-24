"""Unit tests for the PyRallel threaded-ufunc mechanism (parallel/pyrallel.py).

These exercise ``parallel_unary`` and the pool lifecycle directly, plus the
``parallel_ufunc`` fast-path predicate through ``pyoverdrive.explain``. Most
tests need no patching; only the Gearbox-mirroring test enables PyOverdrive.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.fastpaths import parallel_ufunc
from pyoverdrive.parallel import pyrallel

STOCK_SIN = np.sin

SUPPORTED_CASES = [
    (op, dtype, threshold)
    for op, table in parallel_ufunc.SUPPORTED.items()
    for dtype, threshold in table.items()
]

RNG = np.random.default_rng(20260823)


@pytest.fixture(autouse=True)
def _pool_isolation():
    pyrallel.shutdown()
    yield
    pyrallel.shutdown()


def _assert_bit_identical(got, expected):
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    assert np.array_equal(got, expected, equal_nan=True)


# -- parallel_unary correctness ----------------------------------------------

@pytest.mark.parametrize("threads", [2, 3])
def test_parallel_unary_bit_identical_odd_size(threads):
    x = np.linspace(0, 10, 100_003, dtype=np.float64)
    expected = np.sin(x)
    got = pyrallel.parallel_unary(np.sin, x, threads)
    _assert_bit_identical(got, expected)


@pytest.mark.parametrize("threads", [2, 3])
def test_parallel_unary_out_form(threads):
    x = np.linspace(0, 10, 100_003, dtype=np.float64)
    expected = np.sin(x)
    out = np.empty_like(x)
    got = pyrallel.parallel_unary(np.sin, x, threads, out=out)
    assert got is out
    _assert_bit_identical(got, expected)


@pytest.mark.parametrize("threads", [2, 3])
def test_parallel_unary_in_place(threads):
    x = np.linspace(0, 10, 100_003, dtype=np.float64)
    expected = np.sin(x.copy())
    got = pyrallel.parallel_unary(np.sin, x, threads, out=x)
    assert got is x
    _assert_bit_identical(got, expected)


# -- exception propagation ----------------------------------------------------

def test_parallel_unary_exception_propagates_and_pool_recovers():
    x = np.linspace(0, 1, 100_003, dtype=np.float64)
    lock = threading.Lock()
    raised = {"done": False}

    def flaky(chunk, out):
        with lock:
            should_raise = not raised["done"]
            raised["done"] = True
        if should_raise:
            raise ValueError("boom")
        out[:] = np.sin(chunk)
        return out

    with pytest.raises(ValueError):
        pyrallel.parallel_unary(flaky, x, 4)

    result = pyrallel.parallel_unary(np.sin, x, 4)
    _assert_bit_identical(result, np.sin(x))


# -- PYOVERDRIVE_THREADS env handling -----------------------------------------

def test_threads_env_one_disables_parallel(monkeypatch):
    monkeypatch.setenv("PYOVERDRIVE_THREADS", "1")
    assert pyrallel.available() is False
    op, dtype, threshold = SUPPORTED_CASES[0]
    x = np.ones(threshold, dtype=dtype)
    assert pyoverdrive.explain(f"numpy.{op}", x)[0] == "stock"


def test_threads_env_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PYOVERDRIVE_THREADS", "garbage")
    cores = os.cpu_count() or 1
    assert pyrallel.max_threads() == min(cores, 16)


def test_threads_env_huge_value_caps_at_cpu_count(monkeypatch):
    monkeypatch.setenv("PYOVERDRIVE_THREADS", "99999999")
    assert pyrallel.max_threads() == (os.cpu_count() or 1)


# -- lazy pool lifecycle -------------------------------------------------------

def test_lazy_pool_lifecycle():
    assert pyrallel.pool_size() == 0
    x = np.linspace(0, 1, 100_003, dtype=np.float64)
    pyrallel.parallel_unary(np.sin, x, 3)
    assert pyrallel.pool_size() > 0
    assert pyrallel.pool_size() == pyrallel.max_threads()


# -- threads_for schedule -------------------------------------------------------

def test_threads_for_monotonic_non_decreasing():
    ns = [0, 1, 99_999, 100_000, 999_999, 1_000_000, 9_999_999, 10_000_000, 50_000_000]
    prev = 0
    for n in ns:
        t = parallel_ufunc.threads_for(n)
        assert t >= prev
        prev = t


def test_threads_for_returns_one_below_smallest_row():
    smallest_min = min(m for m, _ in parallel_ufunc.THREAD_SCHEDULE)
    assert parallel_ufunc.threads_for(0) == 1
    assert parallel_ufunc.threads_for(smallest_min - 1) == 1


# -- predicate refusals through explain() --------------------------------------

def _different_dtype(dtype):
    alt = np.dtype(np.float32)
    return alt if np.dtype(dtype) != alt else np.dtype(np.float64)


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_predicate_refuses_ndarray_subclass(op, dtype, threshold):
    class Sub(np.ndarray):
        pass

    x = np.ones(threshold, dtype=dtype).view(Sub)
    assert pyoverdrive.explain(f"numpy.{op}", x)[0] == "stock"


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_predicate_refuses_non_contiguous(op, dtype, threshold):
    x = np.ones(threshold * 2, dtype=dtype)[::2]
    assert pyoverdrive.explain(f"numpy.{op}", x)[0] == "stock"


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_predicate_refuses_fortran_2d(op, dtype, threshold):
    x = np.asfortranarray(np.ones((threshold, 2), dtype=dtype))
    assert pyoverdrive.explain(f"numpy.{op}", x)[0] == "stock"


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_predicate_refuses_unsupported_dtype(op, dtype, threshold):
    x = np.ones(threshold, dtype=np.int64)
    assert pyoverdrive.explain(f"numpy.{op}", x)[0] == "stock"


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_predicate_refuses_size_below_threshold(op, dtype, threshold):
    x = np.ones(threshold - 1, dtype=dtype)
    assert pyoverdrive.explain(f"numpy.{op}", x)[0] == "stock"


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_predicate_refuses_where_kwarg(op, dtype, threshold):
    x = np.ones(threshold, dtype=dtype)
    assert pyoverdrive.explain(f"numpy.{op}", x, where=True)[0] == "stock"


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_predicate_refuses_out_wrong_shape(op, dtype, threshold):
    x = np.ones(threshold, dtype=dtype)
    out = np.ones(threshold + 1, dtype=dtype)
    assert pyoverdrive.explain(f"numpy.{op}", x, out=out)[0] == "stock"


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_predicate_refuses_out_wrong_dtype(op, dtype, threshold):
    x = np.ones(threshold, dtype=dtype)
    out = np.ones(threshold, dtype=_different_dtype(dtype))
    assert pyoverdrive.explain(f"numpy.{op}", x, out=out)[0] == "stock"


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_predicate_refuses_out_non_contiguous(op, dtype, threshold):
    x = np.ones(threshold, dtype=dtype)
    out = np.ones(threshold * 2, dtype=dtype)[::2]
    assert pyoverdrive.explain(f"numpy.{op}", x, out=out)[0] == "stock"


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_predicate_refuses_out_read_only(op, dtype, threshold):
    x = np.ones(threshold, dtype=dtype)
    out = np.ones(threshold, dtype=dtype)
    out.flags.writeable = False
    assert pyoverdrive.explain(f"numpy.{op}", x, out=out)[0] == "stock"


@pytest.mark.parametrize("op,dtype,threshold", SUPPORTED_CASES)
def test_predicate_refuses_two_positional_args(op, dtype, threshold):
    x = np.ones(threshold, dtype=dtype)
    assert pyoverdrive.explain(f"numpy.{op}", x, x)[0] == "stock"


# -- Gearbox mirroring ----------------------------------------------------------

def test_gearbox_mirrors_ufunc_attributes_and_restores_on_disable():
    pyoverdrive.enable(["numpy.sin"])
    try:
        assert np.sin.nin == 1
        assert np.sin.nout == 1
        assert callable(np.sin.at)
        arr = np.array([0.0, 1.0, 2.0])
        arr_stock = arr.copy()
        np.sin.at(arr, [0])
        STOCK_SIN.at(arr_stock, [0])
        _assert_bit_identical(arr, arr_stock)
        assert np.sin.__pyoverdrive__ is True
    finally:
        pyoverdrive.disable()
    assert np.sin is STOCK_SIN


# -- shutdown safety at interpreter exit -----------------------------------------

def test_shutdown_safe_in_subprocess():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import numpy as np, pyoverdrive; pyoverdrive.enable(); "
            "x = np.linspace(0, 1, 2_000_000); np.sin(x); print('ok')",
        ],
        cwd=str(repo_root),
        env=env,
        timeout=120,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --- np.errstate is thread-local: the caller's mode must govern the chunks ---

def test_errstate_ignore_is_mirrored_into_workers():
    import warnings

    op, dtype, threshold = SUPPORTED_CASES[0]
    x = np.linspace(0.0, 1.0, 2 * threshold, dtype=dtype)
    x[5] = np.inf  # sin/cos/exp/log/... of inf is invalid or overflow
    pyoverdrive.enable([f"numpy.{op}"])
    try:
        with np.errstate(all="ignore"):
            assert pyoverdrive.explain(f"numpy.{op}", x)[0] == f"pyrallel_{op}"
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # any RuntimeWarning fails here
                getattr(np, op)(x)
    finally:
        pyoverdrive.disable()
        pyrallel.shutdown()


def test_errstate_raise_stays_on_stock_and_raises_like_stock():
    op, dtype, threshold = SUPPORTED_CASES[0]
    x = np.linspace(0.0, 1.0, 2 * threshold, dtype=dtype)
    x[5] = np.inf
    pyoverdrive.enable([f"numpy.{op}"])
    try:
        with np.errstate(all="raise"):
            assert pyoverdrive.explain(f"numpy.{op}", x) == ("stock", "no-applicable-path")
            with pytest.raises(FloatingPointError):
                getattr(np, op)(x)
    finally:
        pyoverdrive.disable()
        pyrallel.shutdown()
