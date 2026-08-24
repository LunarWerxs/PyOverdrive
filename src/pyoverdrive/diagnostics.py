"""Diagnostics and configuration: what is active, and a self-check against stock.

``status()``     machine-readable snapshot (dict): version, activation state,
                 every registered fast path with its kill-switch state and
                 provenance, PyRallel pool state, NumPy version.
``report()``     the same, printed for humans.
``configure()``  one call for the runtime knobs: thread cap, per-path
                 enable/disable, debug.
``selfcheck()``  exercises every fast path on this machine against stock
                 NumPy with a representative dispatching input and the path's
                 own comparison mode; returns per-path PASS/FAIL. This is the
                 post-install proof and the first thing to run on new
                 hardware (``python -m pyoverdrive --selfcheck``).

Nothing here depends on the lab; it ships in the wheel.
"""

from __future__ import annotations

import sys
from typing import Callable

import numpy as np

from . import __version__
from .dispatcher.gearbox import GEARBOX
from .parallel import pyrallel


def status() -> dict:
    paths = []
    for op, registered in sorted(GEARBOX._paths.items()):
        for p in registered:
            paths.append(
                {
                    "name": p.name,
                    "op": op,
                    "enabled": p.enabled,
                    "priority": p.priority,
                    "opportunity": p.provenance.get("opportunity"),
                    "comparison_mode": p.provenance.get("comparison_mode"),
                }
            )
    return {
        "version": __version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "activated": GEARBOX.patched,
        "patched_operations": sorted(GEARBOX._stock),
        "fast_paths": paths,
        "pyrallel": {
            "available": pyrallel.available(),
            "max_threads": pyrallel.max_threads(),
            "pool_size": pyrallel.pool_size(),
        },
        "debug": GEARBOX._debug,
    }


def report(file=None) -> None:
    s = status()
    out = file or sys.stdout
    w = out.write
    w(f"PyOverdrive {s['version']} on numpy {s['numpy']} / Python {s['python']}\n")
    w(f"activated: {s['activated']}")
    if s["activated"]:
        w(f" ({len(s['patched_operations'])} operations patched)")
    w("\n")
    w(
        f"PyRallel: available={s['pyrallel']['available']} "
        f"max_threads={s['pyrallel']['max_threads']} "
        f"pool_size={s['pyrallel']['pool_size']}\n"
    )
    w(f"debug: {s['debug']}\n\nfast paths:\n")
    for p in s["fast_paths"]:
        flag = "on " if p["enabled"] else "OFF"
        w(f"  [{flag}] {p['name']:<22} {p['op']:<22} {p['opportunity'] or '-':<11} {p['comparison_mode'] or ''}\n")


def configure(
    *,
    threads: int | None = None,
    disable: tuple[str, ...] | list[str] = (),
    enable: tuple[str, ...] | list[str] = (),
    debug: bool | None = None,
) -> dict:
    """Apply runtime knobs in one call; returns ``status()`` afterwards.

    ``threads``: PyRallel pool cap (``1`` switches every parallel path off);
    ``None`` leaves it unchanged. ``disable``/``enable``: fast-path names.
    """
    if threads is not None:
        pyrallel.set_max_threads(threads)
    for name in disable:
        GEARBOX.set_path_enabled(name, False)
    for name in enable:
        GEARBOX.set_path_enabled(name, True)
    if debug is not None:
        GEARBOX.set_debug(debug)
    return status()


# --- self-check -------------------------------------------------------------

# Representative DISPATCHING inputs per fast path. Each entry is a callable
# returning (args, kwargs). Sizes sit above the calibrated thresholds so the
# predicate accepts; the check then compares patched vs stock output under
# the path's declared comparison mode. Keep these in sync with the
# thresholds in the fast-path modules (the selfcheck asserts dispatch, so a
# stale size fails loudly rather than silently testing stock against stock).
def _inputs_unique():
    rng = np.random.default_rng(1)
    return (rng.integers(0, 1_000_000, size=50_000, dtype=np.int64),), {}


def _inputs_inner():
    rng = np.random.default_rng(2)
    return (rng.standard_normal((4, 5, 64)), rng.standard_normal((32, 64))), {}


def _inputs_intersect():
    rng = np.random.default_rng(3)
    return (
        rng.integers(0, 30_000, size=10_000, dtype=np.int64),
        rng.integers(0, 30_000, size=2_000, dtype=np.int64),
    ), {}


def _inputs_unique_axis0():
    from .fastpaths import unique_axis0_column

    rng = np.random.default_rng(6)
    n = 2 * unique_axis0_column.SIZE_THRESHOLD
    return (rng.integers(0, 500, size=(n, 1), dtype=np.int64),), {"axis": 0}


def _inputs_relayout():
    from .fastpaths import relayout_blocked

    n = int(max(relayout_blocked.SUPPORTED.values()) ** 0.5)
    rng = np.random.default_rng(5)
    return (rng.standard_normal((n, n)).T,), {}  # F-contiguous view


def _inputs_fftconvolve():
    from .fastpaths import fftconvolve

    spec = fftconvolve.SUPPORTED[np.dtype(np.float64)]
    n = max(spec.min_len, -(-spec.product // spec.min_len))
    rng = np.random.default_rng(7)
    # positive inputs in [1, 2): every full-mode lag is then >= 1, so the
    # numeric selfcheck tolerance (rtol 1e-9) sits ~200x above the FFT's
    # rounding difference even at the edge lags (single products)
    return (rng.random(n) + 1.0, rng.random(spec.min_len) + 1.0), {}


def _inputs_fftcorrelate():
    from .fastpaths import fftconvolve

    spec = fftconvolve.SUPPORTED[np.dtype(np.int64)]
    n = max(spec.min_len, -(-spec.product // spec.min_len))
    rng = np.random.default_rng(8)
    a = rng.integers(-100, 101, size=n, dtype=np.int64)
    v = rng.integers(-100, 101, size=spec.min_len, dtype=np.int64)
    return (a, v), {"mode": "full"}


def _inputs_nanquantile():
    rng = np.random.default_rng(9)
    a = rng.uniform(-5.0, 5.0, size=(27, 100))
    a[rng.random((27, 100)) < 0.1] = np.nan
    return (a, 0.8), {"axis": 0}


def _inputs_nanpercentile():
    rng = np.random.default_rng(31)
    a = rng.uniform(-5.0, 5.0, size=(27, 100))
    a[rng.random((27, 100)) < 0.1] = np.nan
    return (a, 80.0), {"axis": 0}


def _inputs_einsum():
    rng = np.random.default_rng(10)
    # positive operands: contraction outputs then sit far from zero, so the
    # numeric tolerance is comfortably above optimize=True's reordering noise
    a = rng.uniform(0.5, 1.5, size=(120, 120))
    b = rng.uniform(0.5, 1.5, size=(120, 120))
    return ("ij,jk->ik", a, b), {}


def _inputs_searchsorted():
    rng = np.random.default_rng(11)
    x = np.sort(rng.standard_normal(50_000))
    v = rng.standard_normal(50_000)  # random order: passes the disorder gate
    return (x, v), {}


def _inputs_isclose():
    rng = np.random.default_rng(12)
    a = rng.uniform(-10.0, 10.0, size=500)
    return (a, a + rng.uniform(-1e-7, 1e-7, size=500)), {}


def _inputs_isin_string():
    rng = np.random.default_rng(13)
    words = np.array([f"w{i}" for i in range(50)], dtype=np.dtypes.StringDType())
    element = words[rng.integers(0, 50, size=2_000)]
    test = words[:20].copy()
    return (element, test), {}


def _inputs_dot_mixed():
    rng = np.random.default_rng(14)
    a = rng.standard_normal((300, 100))
    b = rng.standard_normal(100) + 1j * rng.standard_normal(100)
    return (a, b), {}


def _inputs_quantile_dense():
    rng = np.random.default_rng(15)
    a = rng.standard_normal((40, 1024))
    q = np.linspace(0.0, 1.0, 128)
    return (a, q), {"axis": -1}


def _inputs_percentile_dense():
    rng = np.random.default_rng(30)
    a = rng.standard_normal((40, 1024))
    q = np.linspace(0.0, 100.0, 128)
    return (a, q), {"axis": -1}


def _inputs_sort_char():
    rng = np.random.default_rng(16)
    alphabet = np.array(list("ASDFGHJKLZ"), dtype="U1")
    return (alphabet[rng.integers(0, 10, size=5_000)],), {}


def _inputs_unique_char():
    rng = np.random.default_rng(17)
    alphabet = np.array(list("ASDFGHJKLZ"), dtype="U1")
    return (alphabet[rng.integers(0, 10, size=5_000)],), {"return_counts": True}


def _inputs_mean_tiny():
    rng = np.random.default_rng(18)
    return (rng.random(size=(20_000, 3)),), {"axis": 0}


def _inputs_sum_tiny():
    rng = np.random.default_rng(19)
    return (rng.random(size=(150, 150, 3)),), {"axis": (0, 1)}


def _inputs_eigvalsh_2x2():
    rng = np.random.default_rng(20)
    a = rng.uniform(-1.0, 1.0, size=(500, 2, 2))
    a = np.ascontiguousarray(a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(2))
    return (a,), {}


def _inputs_matmul_split():
    rng = np.random.default_rng(21)
    c = rng.uniform(0.5, 1.5, size=(64, 1200)) + 1j * rng.uniform(0.5, 1.5, size=(64, 1200))
    r = rng.uniform(0.5, 1.5, size=(1200, 600))
    return (c, r), {}


def _inputs_roll_concat():
    rng = np.random.default_rng(22)
    return (rng.random(500), 7), {}


def _inputs_argmax_blocked():
    rng = np.random.default_rng(23)
    return (rng.random(size=(3_200, 3_200)),), {"axis": 0}


def _inputs_inv_small():
    rng = np.random.default_rng(24)
    a = rng.uniform(-1.0, 1.0, size=(500, 3, 3))
    return (np.ascontiguousarray(a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(3)),), {}


def _inputs_isin_object():
    rng = np.random.default_rng(25)
    vocab = [f"w{i}" for i in range(50)]
    el = np.array([vocab[i] for i in rng.integers(0, 50, size=400)], dtype=object)
    te = np.array(vocab[:20], dtype=object)
    return (el, te), {}


def _inputs_median_partition():
    rng = np.random.default_rng(26)
    return (rng.random(1_001),), {}


def _inputs_searchsorted_extreme():
    rng = np.random.default_rng(27)
    a = np.sort(rng.integers(-(2**40), 2**40, size=500, dtype=np.int64))
    return (a, 2**70), {}


def _inputs_unique_rows():
    rng = np.random.default_rng(28)
    return (rng.integers(-50, 50, size=(5_000, 3), dtype=np.int64),), {
        "axis": 0,
        "return_counts": True,
    }


def _inputs_hist2d_uniform():
    rng = np.random.default_rng(29)
    x = rng.normal(0.0, 1.0, size=20_000)
    y = rng.normal(0.0, 1.0, size=20_000)
    return (x, y), {"bins": [40, 40], "range": [[-3.0, 3.0], [-3.0, 3.0]]}


def _inputs_nanreduce(size):
    def make():
        rng = np.random.default_rng(32)
        return (rng.standard_normal(size),), {}

    return make


def _inputs_nanarg():
    rng = np.random.default_rng(33)
    return (rng.standard_normal(5_000),), {}


def _inputs_nanmedian_scan():
    rng = np.random.default_rng(34)
    return (rng.standard_normal((500, 500)),), {"axis": 1}


def _inputs_int_matmul():
    rng = np.random.default_rng(35)
    x = rng.integers(-1000, 1000, (100, 100)).astype(np.int64)
    y = rng.integers(-1000, 1000, (100, 100)).astype(np.int64)
    return (x, y), {}


def _inputs_det_batch():
    rng = np.random.default_rng(36)
    return (rng.standard_normal((500, 3, 3)),), {}


def _inputs_cholesky_batch():
    rng = np.random.default_rng(47)
    m = rng.standard_normal((1_200, 3, 3))
    return (np.ascontiguousarray(m @ np.swapaxes(m, -1, -2) + 3.0 * np.eye(3)),), {}


def _inputs_eigvalsh_3x3():
    rng = np.random.default_rng(48)
    a = rng.uniform(-1.0, 1.0, size=(500, 3, 3))
    a = np.ascontiguousarray(a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(3))
    return (a,), {}


def _inputs_einsum_chain():
    rng = np.random.default_rng(49)
    return (
        "ij,jk,kl->il",
        rng.standard_normal((32, 32)),
        rng.standard_normal((32, 32)),
        rng.standard_normal((32, 32)),
    ), {}


def _inputs_interp_uniform():
    rng = np.random.default_rng(51)
    xp = np.linspace(0.0, 1.0, 1_000)
    return (rng.uniform(-0.1, 1.1, 20_000), xp, rng.standard_normal(1_000)), {}


def _inputs_apply_along_axis():
    rng = np.random.default_rng(55)
    return (np.mean, 1, rng.standard_normal((2_000, 100))), {}


def _inputs_svd_batch():
    rng = np.random.default_rng(56)
    return (rng.standard_normal((2_000, 3, 3)),), {}


def _inputs_norm2_batch():
    rng = np.random.default_rng(57)
    return (rng.standard_normal((2_000, 3, 3)),), {"ord": 2, "axis": (-2, -1)}


def _inputs_svdvals_batch():
    rng = np.random.default_rng(58)
    return (rng.standard_normal((2_000, 3, 3)),), {"compute_uv": False}


def _inputs_qr_batch():
    rng = np.random.default_rng(53)
    return (rng.standard_normal((1_000, 3, 3)),), {}


def _inputs_take_out():
    # the fast path and stock share this out buffer, so the value compare
    # is vacuous here; the selfcheck still proves dispatch, no-raise, and
    # shape/dtype, and the differential suite owns the values
    rng = np.random.default_rng(52)
    a = rng.standard_normal(100_000)
    idx = rng.integers(0, a.size, 20_000).astype(np.intp)
    return (a, idx), {"out": np.empty(idx.size)}


def _inputs_solve_batch():
    rng = np.random.default_rng(37)
    return (
        rng.standard_normal((2_000, 3, 3)),
        rng.standard_normal((2_000, 3, 1)),
    ), {}


def _inputs_nan_to_num():
    rng = np.random.default_rng(38)
    a = rng.standard_normal(20_000)
    a[rng.random(20_000) < 0.01] = np.nan
    return (a,), {}


def _inputs_ufunc(op_name: str) -> Callable[[], tuple[tuple, dict]]:
    from .fastpaths import parallel_ufunc

    domains = {
        "sin": (0.0, 6.0), "cos": (0.0, 6.0), "tan": (-1.4, 1.4), "exp": (-4.0, 4.0),
        "log": (0.1, 50.0), "log10": (0.1, 50.0), "tanh": (-3.0, 3.0), "sqrt": (0.0, 50.0),
    }

    def make():
        table = parallel_ufunc.SUPPORTED[op_name]
        dtype = np.dtype(np.float64) if np.dtype(np.float64) in table else next(iter(table))
        n = table[dtype]
        lo, hi = domains[op_name]
        return (np.linspace(lo, hi, n, dtype=dtype),), {}

    return make


def _inputs_binary(op_name: str) -> Callable[[], tuple[tuple, dict]]:
    from .fastpaths import parallel_binary

    def make():
        table = parallel_binary.SUPPORTED[op_name]
        dtype = np.dtype(np.int64) if np.dtype(np.int64) in table else next(iter(table))
        n = table[dtype]
        rng = np.random.default_rng(4)
        if dtype.kind == "f":
            a = rng.random(n).astype(dtype) + 0.5
            b = rng.random(n).astype(dtype) + 0.5
        else:
            a = rng.integers(1, 1000, size=n, dtype=dtype)
            b = rng.integers(1, 1000, size=n, dtype=dtype)
        return (a, b), {}

    return make


def _inputs_vectorize_ufunc():
    """((construction args, kwargs), (call args, kwargs)) for the class path."""
    rng = np.random.default_rng(54)
    return ((np.sin,), {}), ((rng.standard_normal(50_000),), {})


def _selfcheck_class_inputs() -> dict[str, Callable[[], tuple]]:
    return {"vectorize_ufunc_direct": _inputs_vectorize_ufunc}


def _selfcheck_inputs() -> dict[str, Callable[[], tuple[tuple, dict]]]:
    from .fastpaths import parallel_binary, parallel_ufunc

    table = {
        "unique_sort": _inputs_unique,
        "unique_values_sort": _inputs_unique,
        "inner_tensordot": _inputs_inner,
        "intersect_sorted": _inputs_intersect,
        "relayout_blocked": _inputs_relayout,
        "unique_axis0_column": _inputs_unique_axis0,
        "fftconvolve": _inputs_fftconvolve,
        "fftcorrelate": _inputs_fftcorrelate,
        "nanquantile_masked": _inputs_nanquantile,
        "nanpercentile_masked": _inputs_nanpercentile,
        "nanmean_scan": _inputs_nanreduce(5_000),
        "nansum_scan": _inputs_nanreduce(20_000),
        "nanstd_scan": _inputs_nanreduce(10_000),
        "nanvar_scan": _inputs_nanreduce(10_000),
        "nanargmax_scan": _inputs_nanarg,
        "nanargmin_scan": _inputs_nanarg,
        "nanmedian_scan": _inputs_nanmedian_scan,
        "matmul_int_blas": _inputs_int_matmul,
        "dot_int_blas": _inputs_int_matmul,
        "det_small_batch": _inputs_det_batch,
        "slogdet_small_batch": _inputs_det_batch,
        "solve_small_batch": _inputs_solve_batch,
        "cholesky_small_batch": _inputs_cholesky_batch,
        "qr_small_batch": _inputs_qr_batch,
        "pinv_small_batch": _inputs_svd_batch,
        "norm2_small_batch": _inputs_norm2_batch,
        "svdvals_small_batch": _inputs_svdvals_batch,
        "apply_along_axis_reduce": _inputs_apply_along_axis,
        "eigvalsh_3x3_trig": _inputs_eigvalsh_3x3,
        "einsum_optimize_chain": _inputs_einsum_chain,
        "interp_uniform_grid": _inputs_interp_uniform,
        "take_index_assign": _inputs_take_out,
        "nan_to_num_where": _inputs_nan_to_num,
        "einsum_optimize": _inputs_einsum,
        "searchsorted_sortqueries": _inputs_searchsorted,
        "isclose_fused": _inputs_isclose,
        "isin_string_hash": _inputs_isin_string,
        "dot_mixed_view": _inputs_dot_mixed,
        "quantile_dense_sort": _inputs_quantile_dense,
        "percentile_dense": _inputs_percentile_dense,
        "sort_char_view": _inputs_sort_char,
        "unique_char_view": _inputs_unique_char,
        "mean_tiny_trailing": _inputs_mean_tiny,
        "sum_tiny_trailing": _inputs_sum_tiny,
        "eigvalsh_2x2_closed": _inputs_eigvalsh_2x2,
        "matmul_split_complex": _inputs_matmul_split,
        "roll_concat_1d": _inputs_roll_concat,
        "argmax_blocked_transpose": _inputs_argmax_blocked,
        "inv_small_batch": _inputs_inv_small,
        "isin_object_hash": _inputs_isin_object,
        "median_partition": _inputs_median_partition,
        "searchsorted_extreme_key": _inputs_searchsorted_extreme,
        "unique_rows_lexsort": _inputs_unique_rows,
        "hist2d_uniform": _inputs_hist2d_uniform,
    }
    for op_name in parallel_ufunc.SUPPORTED:
        table[f"pyrallel_{op_name}"] = _inputs_ufunc(op_name)
    for op_name in parallel_binary.SUPPORTED:
        table[f"pyrallel_{op_name}"] = _inputs_binary(op_name)
    return table


def _equal(mode: str | None, got, expected) -> bool:
    if isinstance(got, tuple) or isinstance(expected, tuple):
        return len(got) == len(expected) and all(_equal(mode, g, e) for g, e in zip(got, expected))
    got = np.asarray(got)
    expected = np.asarray(expected)
    if got.dtype != expected.dtype or got.shape != expected.shape:
        return False
    if mode and mode.startswith("set-equal"):
        return bool(np.array_equal(np.sort(got, axis=None), np.sort(expected, axis=None), equal_nan=True))
    if mode and mode.startswith("numeric"):
        tol = dict(rtol=1e-9, atol=1e-12) if got.dtype == np.float64 else dict(rtol=1e-4, atol=1e-5)
        return bool(np.allclose(got, expected, equal_nan=True, **tol))
    if got.dtype.kind in "fc":
        return bool(np.array_equal(got, expected, equal_nan=True))
    return bool(np.array_equal(got, expected))  # equal_nan TypeErrors on string dtypes


def selfcheck(verbose: bool = True, file=None) -> dict[str, str]:
    """Run every registered fast path against stock on this machine.

    Returns {path name: "PASS" | "FAIL: reason" | "SKIP: reason"}. Temporarily
    activates the affected operations and restores the prior activation state.
    """
    out = file or sys.stdout
    inputs = _selfcheck_inputs()
    was_active = GEARBOX.patched
    previously_patched = set(GEARBOX._stock)
    results: dict[str, str] = {}
    try:
        for op, registered in sorted(GEARBOX._paths.items()):
            for path in registered:
                if not path.enabled:
                    results[path.name] = "SKIP: disabled"
                    continue
                make = inputs.get(path.name)
                if make is None:
                    results[path.name] = "SKIP: no selfcheck input registered"
                    continue
                args, kwargs = make()
                stock = GEARBOX.stock_fn(op)
                chosen, reason = GEARBOX.decide(op, args, kwargs)
                if chosen != path.name:
                    results[path.name] = f"FAIL: did not dispatch ({chosen}: {reason})"
                    continue
                GEARBOX.patch([op])
                try:
                    patched_fn = GEARBOX._resolve(np, op)
                    got = getattr(*patched_fn)(*args, **kwargs)
                    expected = stock(*args, **kwargs)
                except Exception as exc:  # a raising path is a FAIL, not a crash
                    results[path.name] = f"FAIL: raised {exc!r}"
                    continue
                ok = _equal(path.provenance.get("comparison_mode"), got, expected)
                results[path.name] = "PASS" if ok else "FAIL: result differs from stock"
        # class-backed paths (ClassPath): construct through the patched
        # class and through stock, call both, compare. Same contract as a
        # FastPath, one extra step because the work lives on an instance.
        for op, cpath in sorted(GEARBOX._class_paths.items()):
            if not cpath.enabled:
                results[cpath.name] = "SKIP: disabled"
                continue
            make = _selfcheck_class_inputs().get(cpath.name)
            if make is None:
                results[cpath.name] = "SKIP: no selfcheck input registered"
                continue
            (cargs, ckwargs), (args, kwargs) = make()
            stock_cls = GEARBOX.stock_fn(op)
            chosen, reason = GEARBOX.decide(op, cargs, ckwargs)
            if chosen != cpath.name:
                results[cpath.name] = f"FAIL: did not dispatch ({chosen}: {reason})"
                continue
            GEARBOX.patch([op])
            try:
                patched_cls = getattr(*GEARBOX._resolve(np, op))
                got = patched_cls(*cargs, **ckwargs)(*args, **kwargs)
                expected = stock_cls(*cargs, **ckwargs)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - a raising path is a FAIL
                results[cpath.name] = f"FAIL: raised {exc!r}"
                continue
            ok = _equal(cpath.provenance.get("comparison_mode"), got, expected)
            results[cpath.name] = "PASS" if ok else "FAIL: result differs from stock"
    finally:
        GEARBOX.unpatch()
        if was_active:
            GEARBOX.patch(sorted(previously_patched))
    if verbose:
        for name, verdict in results.items():
            out.write(f"  {verdict.split(':')[0]:4}  {name:<22} {verdict[5:] if ':' in verdict else ''}\n")
        n_fail = sum(v.startswith("FAIL") for v in results.values())
        out.write(f"selfcheck: {len(results)} paths, {n_fail} failing\n")
    return results
