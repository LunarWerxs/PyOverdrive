"""PyRallel: PyOverdrive's persistent thread-pool execution core (Phase 4).

Mechanism (OPP-000008, numpy/numpy#8208): NumPy's C ufunc inner loops release
the GIL, so splitting one large elementwise call into contiguous chunks and
running the chunks on a thread pool genuinely scales on multicore hardware,
7.2x for np.sin at 1e7 elements on 16 threads on the first calibrated machine.

Design rules:

- ONE persistent process-wide pool, created lazily on first parallel dispatch,
  never per call (executor startup would eat the win; measured in the
  OPP-000008 reproducer, which pre-creates its executors for the same reason).
- Worker threads are plain ThreadPoolExecutor threads; tasks never submit
  further tasks and never wait on other futures, so pool deadlock is
  impossible by construction.
- The pool only ever runs *stock* NumPy kernels, obtained by the caller via
  ``GEARBOX.stock_fn`` (never a patched public name; OPP-000000 recursion
  incident).
- Any chunk exception propagates to the caller; Gearbox catches it there and
  falls back to stock NumPy. The partially written output buffer is private
  to the failed call, so no caller-visible state is corrupted.
- ``PYOVERDRIVE_THREADS`` (env) caps pool width; ``PYOVERDRIVE_THREADS=1``
  disables parallel dispatch entirely (``available()`` goes False).
- NumPy's floating-point error state (np.errstate / np.seterr) is
  THREAD-LOCAL, so a worker thread does not see the caller's settings. Every
  chunk therefore runs under the caller's ``np.geterr()`` snapshot; the
  ``warn``/``ignore``/``print`` modes then behave as on stock (a warning is
  raised from the worker thread; the warnings machinery is process-global).
  ``raise``, ``call`` and ``log`` modes are refused by the fast-path
  predicate (``errstate_parallel_safe``) and stay on stock, so a
  FloatingPointError or a user callback is never produced off-thread.
- Interpreter shutdown: concurrent.futures joins its threads at exit; tasks
  here are short-lived chunk kernels, so shutdown cannot hang on us. Verified
  by tests/test_pyrallel.py in a subprocess.

Oversubscription posture: chunk kernels are pure elementwise loops - they do
not call BLAS or OpenMP, so PyRallel never *nests* inside another threading
runtime. The pool is capped at the logical core count (default further capped
at 16, the widest measured configuration), so a concurrently running BLAS
workload degrades gracefully by OS scheduling rather than catastrophically by
runaway thread creation.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait as _wait_all

import numpy as np

_DEFAULT_MAX_THREADS = 16  # widest configuration with committed evidence

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_override: int | None = None  # set_max_threads(); wins over the env variable


def set_max_threads(n: int | None) -> None:
    """Programmatic pool width (``pyoverdrive.configure(threads=...)``).

    ``None`` restores env/default resolution. A live pool of a different
    width is shut down and re-created lazily at the new width on next use.
    """
    global _override
    if n is not None:
        n = int(n)
        if n < 1:
            raise ValueError("threads must be >= 1")
    with _lock:
        _override = n
    if pool_size() not in (0, max_threads()):
        shutdown()


def max_threads() -> int:
    """The configured pool width (>= 1). Precedence: set_max_threads() >
    PYOVERDRIVE_THREADS (env) > min(cpu_count, 16)."""
    cores = os.cpu_count() or 1
    if _override is not None:
        return min(_override, cores)
    raw = os.environ.get("PYOVERDRIVE_THREADS", "")
    if raw.strip():
        try:
            requested = int(raw)
        except ValueError:
            requested = 0
        if requested >= 1:
            return min(requested, cores)
    return min(cores, _DEFAULT_MAX_THREADS)


def available() -> bool:
    """True when parallel dispatch can help at all (pool width >= 2)."""
    return max_threads() >= 2


def _pool() -> ThreadPoolExecutor:
    global _executor
    ex = _executor
    if ex is None:
        with _lock:
            ex = _executor
            if ex is None:
                ex = ThreadPoolExecutor(
                    max_workers=max_threads(), thread_name_prefix="pyrallel"
                )
                _executor = ex
    return ex


def _forget_pool_in_child() -> None:
    # Worker threads do not survive fork(); the child must not reuse the
    # parent's executor object. Drop the reference so first use re-creates it.
    global _executor
    _executor = None


if hasattr(os, "register_at_fork"):  # POSIX only
    os.register_at_fork(after_in_child=_forget_pool_in_child)


def shutdown() -> None:
    """Tear the pool down (tests / explicit cleanup). Recreated on next use."""
    global _executor
    with _lock:
        ex = _executor
        _executor = None
    if ex is not None:
        ex.shutdown(wait=True)


def pool_size() -> int:
    """Width of the live pool, or 0 if no pool has been created yet."""
    ex = _executor
    return ex._max_workers if ex is not None else 0


_PARALLEL_SAFE_ERR_MODES = frozenset({"ignore", "warn", "print"})


def errstate_parallel_safe() -> bool:
    """True when the calling thread's FP error modes can be mirrored in
    workers without changing behavior (no raise/call/log)."""
    return all(mode in _PARALLEL_SAFE_ERR_MODES for mode in np.geterr().values())


def _run_chunk(ufunc, ins: tuple, xout, err: dict) -> None:
    with np.errstate(**err):
        ufunc(*ins, out=xout)


def parallel_elementwise(ufunc, inputs: tuple, threads: int, out: np.ndarray | None = None) -> np.ndarray:
    """Run ``ufunc(*inputs, out=out)`` split across ``threads`` contiguous chunks.

    Caller contract (enforced by the fast-path predicates, asserted here only
    cheaply): every input is a C-contiguous plain ndarray of one common
    shape (NO broadcasting), ``ufunc`` is the STOCK ufunc with
    ``nin == len(inputs)`` and ``nout == 1``, and ``out`` (if given) is a
    C-contiguous plain ndarray of that shape and of the dtype the ufunc
    would produce. ``out`` aliasing an input (``np.add(x, y, out=x)``) is
    fine: each chunk reads and writes only its own index range and the
    kernel is elementwise. Without ``out`` a fresh array is returned,
    allocated with the dtype stock would produce (resolved from the ufunc's
    loop table, never guessed). Elementwise kernels have no cross-element
    data flow, so the result is bit-identical to the unchunked call; the
    differential suites assert that, it is not assumed.
    """
    threads = min(threads, max_threads())
    if threads < 2:
        return ufunc(*inputs) if out is None else ufunc(*inputs, out=out)
    if out is None:
        resolve = getattr(ufunc, "resolve_dtypes", None)  # absent on test stubs
        if resolve is not None:
            out_dtype = resolve(tuple(a.dtype for a in inputs) + (None,))[-1]
        else:
            out_dtype = np.result_type(*inputs)
        out = np.empty(inputs[0].shape, dtype=out_dtype)
    flats = tuple(a.reshape(-1) for a in inputs)
    of = out.reshape(-1)
    bounds = np.linspace(0, of.size, threads + 1).astype(np.int64)
    err = np.geterr()  # caller's thread-local FP error state, mirrored per chunk
    ex = _pool()
    futures = [
        ex.submit(
            _run_chunk,
            ufunc,
            tuple(f[bounds[i] : bounds[i + 1]] for f in flats),
            of[bounds[i] : bounds[i + 1]],
            err,
        )
        for i in range(threads)
    ]
    # Wait for EVERY chunk before surfacing an error: with out= the buffer
    # belongs to the caller, and stock's fallback rewrite must not race a
    # straggling worker that is still writing its range.
    _wait_all(futures)
    for f in futures:
        f.result()  # re-raises the first chunk exception -> Gearbox falls back
    return out


def parallel_unary(ufunc, x: np.ndarray, threads: int, out: np.ndarray | None = None) -> np.ndarray:
    """``parallel_elementwise`` for one input; see there for the contract."""
    return parallel_elementwise(ufunc, (x,), threads, out=out)
