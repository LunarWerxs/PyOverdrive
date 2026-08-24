"""The 60-second demo: headline operations, stock vs PyOverdrive, THIS machine.

``python -m pyoverdrive --demo`` runs a handful of representative calls
twice - once against stock NumPy, once with ``enable()`` - and prints the
measured wall-clock side by side. Nothing is precomputed or replayed: the
numbers are produced live on the caller's hardware, with a warmup call and
a best-of-reps timing per side (reps sized so each side costs tenths of a
second, not minutes). It is a demonstration, not evidence: the committed
Dyno batteries under benchmarks/results/ are the calibrated record, and
``--selfcheck`` is the correctness proof.

Ships in the wheel; imports nothing from the lab.
"""

from __future__ import annotations

import os
import time
import timeit

import numpy as np


def _fmt_t(seconds: float) -> str:
    if seconds >= 0.1:
        return f"{seconds * 1e0:6.2f} s "
    if seconds >= 1e-3:
        return f"{seconds * 1e3:6.1f} ms"
    return f"{seconds * 1e6:6.0f} us"


def _time(fn, budget: float = 0.35) -> float:
    """Best-of-reps wall time for fn, sized to roughly `budget` seconds.

    A call slower than the whole budget is reported from its single timed
    run (after the warmup) instead of being repeated: the multi-second
    stock rows would otherwise dominate the demo's runtime, and at that
    scale run-to-run variance is small relative to the gaps shown."""
    fn()  # warmup (allocations, pool spin-up, cache)
    t0 = time.perf_counter()
    fn()
    once = max(time.perf_counter() - t0, 1e-9)
    if once >= budget:
        return once
    reps = max(3, min(30, int(budget / once)))
    return min(timeit.repeat(fn, number=1, repeat=reps))


def _cases(quick: bool):
    rng = np.random.default_rng(2026)
    k = 4 if quick else 1

    conv_n = 20_000 // k
    conv = rng.standard_normal(conv_n)
    unique_arr = rng.integers(
        np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=1_000_000 // k, dtype=np.int64
    )
    nanq = rng.uniform(size=(50 // k, 100, 100))
    nanq[rng.random(nanq.shape) < 0.1] = np.nan
    haystack = np.sort(rng.integers(0, 3_000_000, size=1_000_000 // k, dtype=np.int64))
    needles = rng.integers(0, 3_000_000, size=100_000 // k, dtype=np.int64)
    es_x = rng.standard_normal((1_000 // k, 1, 500))
    es_y = rng.standard_normal((1_000 // k, 1, 500))
    small = rng.integers(-30_000, 30_000, size=1_000_000 // k, dtype=np.int16)
    sin_x = rng.standard_normal(10_000_000 // k)
    alpha = np.array(list("ASDFGHJKLZ"), dtype="U1")
    chars = alpha[rng.integers(0, 10, size=100_000 // k)]
    img = rng.random(size=(1_000 // k, 1_000, 3))
    covs = rng.uniform(-1.0, 1.0, size=(20_000 // k, 2, 2))
    covs = np.ascontiguousarray(covs @ np.swapaxes(covs, -1, -2) + 0.1 * np.eye(2))
    mats = rng.uniform(-1.0, 1.0, size=(8_000 // k, 3, 3))
    mats = np.ascontiguousarray(mats @ np.swapaxes(mats, -1, -2) + 0.1 * np.eye(3))
    words = [f"key_{i:05d}" for i in range(5_000)]
    obj_el = np.array([words[i] for i in rng.integers(0, 5_000, size=20_000 // k)], dtype=object)
    obj_te = np.array([words[i] for i in rng.choice(5_000, size=2_000 // k, replace=False)], dtype=object)
    imm_n = 400 if k == 1 else 200
    imm_a = rng.integers(-1000, 1000, (imm_n, imm_n)).astype(np.int64)
    imm_b = rng.integers(-1000, 1000, (imm_n, imm_n)).astype(np.int64)
    nanm = rng.standard_normal((1_000 // k, 100))

    return [
        (f"np.convolve(a, v)  {conv_n}x{conv_n} float64",
         lambda f: f.convolve(conv, conv)),
        (f"np.unique(a)  {unique_arr.size:,} int64",
         lambda f: f.unique(unique_arr)),
        (f"np.nanquantile(a, 0.8, axis=0)  {'x'.join(map(str, nanq.shape))}",
         lambda f: f.nanquantile(nanq, 0.8, axis=0)),
        (f"np.intersect1d(a, b)  {haystack.size:,} x {needles.size:,} int64",
         lambda f: f.intersect1d(haystack, needles)),
        (f"np.einsum('thd,Thd->thT', x, y)  {es_x.shape[0]}x1x500",
         lambda f: f.einsum("thd,Thd->thT", es_x, es_y)),
        (f"np.unique(a)  {small.size:,} int16",
         lambda f: f.unique(small)),
        (f"np.sort(a)  {chars.size:,} single-char U1",
         lambda f: f.sort(chars)),
        (f"np.mean(img, axis=(0, 1))  {'x'.join(map(str, img.shape))}",
         lambda f: f.mean(img, axis=(0, 1))),
        (f"np.linalg.eigvalsh(covs)  {covs.shape[0]:,} 2x2 matrices",
         lambda f: f.linalg.eigvalsh(covs)),
        (f"np.linalg.inv(mats)  {mats.shape[0]:,} 3x3 matrices",
         lambda f: f.linalg.inv(mats)),
        (f"np.isin(a, b)  {obj_el.size:,} x {obj_te.size:,} object strings",
         lambda f: f.isin(obj_el, obj_te)),
        (f"np.matmul(a, b)  {imm_n}x{imm_n} int64",
         lambda f: f.matmul(imm_a, imm_b)),
        (f"np.nanmean(a, axis=1)  {'x'.join(map(str, nanm.shape))}",
         lambda f: f.nanmean(nanm, axis=1)),
        (f"np.sin(x)  {sin_x.size:,} float64",
         lambda f: f.sin(sin_x)),
    ]


class _Stock:
    """Attribute proxy resolving numpy names through Gearbox's stock table,
    so the same case lambdas run both sides of the comparison. Submodule
    access (f.linalg.eigvalsh) returns a nested proxy: resolving through
    the real module would hand back the PATCHED attribute and silently
    compare patched against patched."""

    def __init__(self, prefix: str = "numpy"):
        self._prefix = prefix

    def __getattr__(self, name):
        import types

        import numpy

        from .dispatcher.gearbox import GEARBOX

        full = f"{self._prefix}.{name}"
        obj = numpy
        for part in full.split(".")[1:]:
            obj = getattr(obj, part)
        if isinstance(obj, types.ModuleType):
            return _Stock(full)
        return GEARBOX.stock_fn(full)


def demo(quick: bool = False, file=None) -> int:
    import sys

    import pyoverdrive

    from .dispatcher.gearbox import GEARBOX

    out = file or sys.stdout
    w = out.write
    w(
        f"PyOverdrive demo - measured on THIS machine, right now "
        f"(numpy {np.__version__}, {os.cpu_count()} logical CPUs)\n\n"
    )
    budget = 0.05 if quick else 0.35
    was_enabled = GEARBOX.patched
    pyoverdrive.enable()
    stock = _Stock()
    try:
        for label, call in _cases(quick):
            t_stock = _time(lambda: call(stock), budget)
            t_over = _time(lambda: call(np), budget)
            ratio = t_stock / t_over if t_over > 0 else float("inf")
            w(f"  {label:<52} {_fmt_t(t_stock)} -> {_fmt_t(t_over)}  {ratio:6.1f}x\n")
    finally:
        if not was_enabled:
            pyoverdrive.disable()
    w(
        "\nOnly measured regimes dispatch; everything else runs stock NumPy\n"
        "unchanged. Verify every path on this machine:\n"
        "    python -m pyoverdrive --selfcheck\n"
        "Full calibrated evidence: benchmarks/results/ in the repository.\n"
    )
    return 0
