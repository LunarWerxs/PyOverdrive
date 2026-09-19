"""Prove no fast path is SLOWER than stock on its own canonical input.

A fast path that dispatches into a loss is the worst failure this project
can have, and nothing else catches it. The tests check correctness. The
benchmark batteries time the CANDIDATE, which is not the same thing as the
dispatched route - the candidate has no predicate in front of it and no
guard inside it. So a path can advertise a large win, pass every test, and
still make a user's call slower.

That is not hypothetical here. Sweeping end to end found det 2x2 running at
0.70x AT ITS OWN ADVERTISED FLOOR, slogdet 3x3 at 0.96x, and solve at 0.91x
and 0.94x - four dispatched losses across three shipped paths, all of them
because a guard recomputed work the run was about to do anyway.

Method: every path's OWN selfcheck input (the canonical input its author
chose), timed through the PUBLIC API with pyoverdrive enabled, against the
same call with it disabled, with the result consumed so nothing hides in a
lazy allocation. A path is only judged if it actually dispatches on that
input.

ISOLATION IS THE WHOLE METHOD. Every path is measured in its own fresh
process, and that is not caution, it is what makes the number mean
anything. Measuring all 69 in one process reported pyrallel_tan at 0.89x -
reproducibly, four independent re-measures in a row - when the same path on
the same array on the same idle machine is 1.15x in isolation, in every
configuration tried (only-tan enabled, everything enabled, thread pool
already warm). Sixty-eight other paths' allocations reach the threaded ones
through cache and allocator state, and no amount of re-measuring inside
that process escapes it.

The two sides are also timed INTERLEAVED rather than in separate blocks, so
anything that drifts across a measurement cannot land on one side only.
That alone fixed two earlier false reds.

AND, on a hybrid CPU, ONLY ON A FAST CORE - because otherwise this tool
returns FALSE GREENS. A single-threaded stock call runs on whatever core
class the process was handed and stays there; a threaded fast path spans
cores and averages over the difference. So a process that draws an
efficiency core has a slow stock side and a normal patched side, which
flatters the ratio by the class ratio - 1.44x on the box this was found on.
A threaded path that genuinely runs at 0.9x reports 1.3x and passes. Every
process here probes the core it was given and re-draws if it is a slow one
(src/pyoverdrive/_cpuclass.py); the fast class is also the honest one to judge on,
since stock is quickest there and a fast path has the least to offer.

Two limits worth stating even so.

First, this catches gross losses, not the difference between 0.98x and
1.02x. Run it on an idle machine, and settle anything within a few percent
of 1.0 with a dedicated probe.

Second, coverage is per CELL, not per path, and the difference matters. It
used to probe one canonical input per path, which is not coverage where a
path's table spans several dtypes: the selfcheck input picks float64 when
there is one, and pyrallel_subtract passed at 1.13x on its float64 floor
while its float32 row was running at 0.97x. The PyRallel families
contribute one cell per dtype, each at that dtype's own floor, and --rows
(below) covers every other table in the package.

--sizes closes the other half: every cell is also judged at 3x, 10x, 30x
and 100x its canonical size AND at a third, a tenth, a thirtieth and a
hundredth of it. Both directions were needed and both found real losses.
Upward, three det/slogdet cells were losing at the TOP of their window
(1.01x, 0.84x, 0.85x) while passing at their canonical input. Downward,
np.inner had no size gate at all and ran at 0.38x on operands smaller than
its canonical one - which was, it turned out, the smallest shape in the
sweep that wins.

That is the pattern to distrust: a hand-picked canonical input is evidence
about whoever picked it, and they picked one where the path works. A path
with a real floor simply reports NODISPATCH as the cells shrink, which
costs nothing; a path without one keeps accepting, and that is the case
worth finding.

WHAT "3x" MEANS, because it did not always mean this. A cell whose maker
can build its input at a size is REBUILT at that size, so the number of
distinct values grows with n the way a real caller's array does. Everything
else is grown by resampling with replacement from its own values, which
keeps the distribution and re-sorts anything that arrived sorted. Neither
is a tile. `np.concatenate([a] * mult)` was the original and it silently
measured something else: it adds no distinct values, so cardinality stayed
pinned while n grew; it is perfectly periodic, which costs 0.5-1.0x on its
own; and it turns a sorted haystack into a sawtooth, so searchsorted's
size cells were not measuring searchsorted. See _scaled.

--shapes closes the aspect-ratio class that scaling can never reach. At
roughly constant volume, each cell with a 2-D-or-deeper input is re-judged
with its trailing axis grown 4x and 16x while the leading axis shrinks by
the same factor (long rows / long contraction, few of them - exactly the
regime where np.inner ran at 0.38x), and the reverse. Operands that share
their trailing length move together; matmul-shaped pairs get a chain
variant that grows the shared inner dimension instead, so (m,k)x(k,n)
stays a valid product. A PAIR OF 1-D OPERANDS has the same degree of
freedom - one against the other - and gets it moved too, which is the axis
searchsorted's whole regime lived on (its withdrawn int64 row climbed
0.26x to 0.74x purely as the haystack grew under a fixed query count).
Ratio-LOCKED pairs are excluded: elementwise operands must be equal
length, so moving their ratio is a broadcast error rather than a shape.

--rows judges one cell per SHIPPED CALIBRATION ROW, and it is the axis that
was hiding the most. A path's gate is usually a table, and building the
input from the path's ONE fixture judges whichever row that fixture lands
on. Every small-matrix linalg fixture here is d=3, so nine paths shipped
their d=2 and d=4 rows - different closed-form kernels - with no end-to-end
measurement at all. See the block above _ROW_AXES.

--values moves the CARDINALITY axis, redrawing every integer operand from
16 distinct values and from the dtype's whole range. The unique and
intersect families have measured cardinality sensitivity written in their
own comments and never had a cell for it.

What is STILL not covered even so: layout beyond contiguity (every
constructed cell is C- or F-contiguous, never strided); the conditioning
axis for float matrices, which --values deliberately leaves alone because
redrawing a positive-definite stack makes a different call rather than a
different distribution; and DEPENDENCY ROT - a row measured honestly
against one numpy can stop paying when numpy improves, which is not a
hypothetical (see searchsorted_sortqueries.SUPPORTED). For that one, run
tools/verify_across_numpy.py, which runs this whole sweep once per
supported numpy.

Usage:
    .venv/Scripts/python tools/verify_no_pessimization.py [--min 1.0] [-v]
    .venv/Scripts/python tools/verify_no_pessimization.py --rows --sizes --shapes --values --json evidence.json --require-quiet

--json records the machine fingerprint, source content hashes, selected cells and
axes, foreign CPU load before/after, and each fresh child's raw consumed
public-API timings, dispatch decision and declared correctness comparison.
Cell names plus the recorded source snapshot replay the deterministic input
makers; large operand arrays are described rather than embedded. Child JSON
travels through temporary sidecars, so the stdout protocol is unchanged.
--require-quiet refuses load above 20% (or unavailable load), and returns
nonzero if the final load is contended. Samples bracket the run; they do not
prove the machine stayed idle throughout it.
Every timing side must also have median absolute deviation / median <= 20%;
unstable samples are unverified regardless of their reported speed ratio.
--rounds and --sample-seconds can predeclare longer measurements for noisy
cells. Defaults remain nine rounds and a 0.02-second target per side/sample;
the same iteration count applies to stock and patched calls. Actual durations
depend on the initial stock estimate and are retained in the raw evidence.

Exit 0 = a completed sweep judged at least one cell and found no confirmed
loss below --min. Exit 1 = a confirmed loss, missing coverage, or no judged
cells. Child-process crashes or malformed results abort with a nonzero exit
and the child's error; they are never counted as dispatch/shape skips.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import operator
import re
import statistics
import subprocess
import sys
import timeit
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from pyoverdrive import _cpuclass as cpuclass  # noqa: E402

import pyoverdrive  # noqa: E402
from pyoverdrive import diagnostics as D  # noqa: E402
from pyoverdrive.dispatcher.gearbox import GEARBOX  # noqa: E402


def _consume(value) -> None:
    """Touch the result so a lazily allocated one cannot look free."""
    if isinstance(value, tuple):
        for v in value:
            _consume(v)
        return
    arr = np.asarray(value)
    if arr.dtype.kind in "biufc" and arr.size:
        float(np.asarray(arr.sum()).real)


def _resolve(op: str):
    holder = np
    parts = op.split(".")
    for p in parts[1:-1]:
        holder = getattr(holder, p)
    return holder, parts[-1]


def _check_no_fallback(evidence):
    name = evidence.get("path")
    if name is None:
        return
    failed = name in GEARBOX._warned_paths
    evidence.setdefault("dispatch", {})["fallback_observed"] = failed
    if failed:
        raise RuntimeError(f"fast path {name} fell back after an error; execution not verified")


def validate_timing_stability(samples, label="timing"):
    """Read-only robust dispersion check shared by live and saved evidence.

    Unscaled median absolute deviation tolerates isolated scheduling spikes;
    broad variation on either side disqualifies wins and losses equally.
    """
    if (not isinstance(samples, (list, tuple)) or not samples
            or not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples)):
        raise ValueError(f"{label}: invalid timing samples")
    median = statistics.median(samples)
    mad = statistics.median(abs(x - median) for x in samples)
    relative_mad = mad / median
    if relative_mad > 0.20:
        raise RuntimeError(f"{label}: unstable timing; MAD/median={relative_mad:.2%} exceeds 20%; unverified")
    return {"median_seconds": median, "mad_seconds": mad,
            "relative_mad": relative_mad, "threshold": 0.20}


def _measure(op: str, args, kwargs, rounds: int, *, evidence=None,
             comparison_mode: str | None = None, sample_seconds: float = 0.02) -> float | None:
    """Median ratio, measured with the two sides INTERLEAVED.

    Timing all of stock's rounds and then all of the patched rounds looks
    equivalent and is not: anything that drifts over the run - thermal
    state, a thread pool spinning down, another process arriving - lands on
    one side only. That produced two false reds here, reporting threaded
    paths at 0.91x and 0.99x that an alternating probe measured at 1.17x
    and 1.18x on the same idle machine. A pessimization detector that cries
    wolf is worse than none, so the rounds alternate.
    """
    holder, name = _resolve(op)
    stock = GEARBOX.stock_fn(op)
    patched = getattr(holder, name)

    if evidence is not None:
        # out= can make both calls return the same buffer (including inside
        # tuples). Freeze stock's values before the patched call overwrites it.
        expected = copy.deepcopy(stock(*args, **kwargs))
        got = patched(*args, **kwargs)
        _check_no_fallback(evidence)
        correct = D._equal(comparison_mode, got, expected)
        evidence["correctness"] = {
            "passed": correct, "mode": comparison_mode,
            "comparator": "pyoverdrive.diagnostics._equal",
        }
        del expected, got
        if not correct:
            raise RuntimeError(f"correctness check failed for {op}")

    def run_stock():
        _consume(stock(*args, **kwargs))

    def run_patched():
        _consume(patched(*args, **kwargs))

    try:
        run_stock()
        run_patched()
    except Exception:
        return None  # a path whose canonical input raises is selfcheck's problem

    t = timeit.timeit(run_stock, number=1)
    number = 1 if t > sample_seconds else max(1, int(sample_seconds / max(t, 1e-9)))
    for _ in range(max(2, number // 4)):
        run_stock()
        run_patched()

    stock_times, patched_times = [], []
    for _ in range(rounds):
        stock_times.append(timeit.timeit(run_stock, number=number) / number)
        patched_times.append(timeit.timeit(run_patched, number=number) / number)
    s = sorted(stock_times)[rounds // 2]
    c = sorted(patched_times)[rounds // 2]
    ratio = s / c if c > 0 else None
    if evidence is not None:
        evidence.update(stock_seconds=stock_times, patched_seconds=patched_times,
                        iterations_per_sample=number, rounds=rounds,
                        sample_seconds=sample_seconds,
                        stock_median_seconds=s, patched_median_seconds=c, ratio=ratio,
                        consumption="_consume after each public call",
                        sampling="interleaved stock then patched; per-call seconds")
        # Check outside the timed functions: fallback after a late failure
        # must not become a passing stock-vs-stock measurement.
        _check_no_fallback(evidence)
    validate_timing_stability(stock_times, "stock")
    validate_timing_stability(patched_times, "patched")
    return ratio


def cells(mults: tuple = (), divs: tuple = (), shapes: tuple = (),
          rows: bool = False, values: bool = False) -> list[str]:
    """Every cell to judge, as "path", "path@dtype", optionally "*mult",
    "/div", ">f" (trailing axis f-fold longer, leading f-fold shorter) or
    "<f" (the reverse).

    One canonical input per path is not coverage where a path's table spans
    several dtypes: the selfcheck input picks float64 when there is one, and
    pyrallel_subtract passed at 1.13x on float64 while its float32 row was
    running at 0.97x. The two PyRallel families are the only modules here
    with a dtype-keyed threshold table, so they get one cell PER DTYPE, each
    at that dtype's own floor - the weakest size its predicate admits.
    """
    from pyoverdrive.fastpaths import parallel_binary, parallel_ufunc

    tabled: dict[str, list[str]] = {}
    for mod in (parallel_ufunc, parallel_binary):
        for op, row in mod.SUPPORTED.items():
            tabled[f"pyrallel_{op}"] = [np.dtype(d).name for d in row]

    out = []
    for name in sorted(D._selfcheck_inputs()):
        dtypes = tabled.get(name)
        out.extend([f"{name}@{d}" for d in dtypes] if dtypes else [name])
    if rows:
        out.extend(sorted(_row_cells()))
    if not (mults or divs or shapes or values):
        return out
    scaled = []
    for c in out:
        # THE CANONICAL CELL IS ALWAYS JUDGED. It used to be re-added only
        # because SIZE_MULTS happens to contain 1, so any axis used on its
        # own - --shapes, --values - silently dropped every canonical input
        # and returned a green that said nothing about them.
        scaled.append(c)
        for m in mults:
            if m != 1:
                scaled.append(f"{c}*{m}")
        for d in divs:
            scaled.append(f"{c}/{d}")
        for f in shapes:
            scaled.append(f"{c}>{f}")
            scaled.append(f"{c}<{f}")
        if values:
            scaled.append(f"{c}%d")
            scaled.append(f"{c}%u")
    return scaled


# Scaling factors for the deep sweep. Each path's canonical input sits near
# the BOTTOM of what it accepts, so the unsampled axis is upward - and that
# is where the losses were: det 3x3 measured 1.01x at 1e5 and slogdet 3x3
# 0.84x, both far above their canonical cell, both passing the shallow sweep.
SIZE_MULTS = (1, 3, 10, 30, 100)

# ...and DOWNWARD, which is the half that actually bit. np.inner had no size
# gate at all and ran at 0.38x on operands smaller than its canonical input,
# for months. A path with a floor simply stops dispatching as these shrink,
# which reports as NODISPATCH and costs nothing; a path WITHOUT one keeps
# accepting, and that is exactly what needs finding.
SIZE_DIVS = (3, 10, 30, 100)

# Aspect-ratio factors for --shapes. Scaling moves volume; these move SHAPE
# at (roughly) constant volume, which is the axis scaling can never sample.
# np.inner's 0.38x corner was exactly this class - a long contraction
# against few rows has the same element count as its canonical input, so no
# multiple or fraction of that input would ever have produced it.
ASPECT_FACTORS = (4, 16)

# A scaled cell that would allocate more than this is skipped rather than
# risking an OOM on someone's machine; the skip is printed, not silent.
MAX_ELEMENTS = 120_000_000
# Fixed, so a resampled cell is the same cell every time it is measured.
_SCALE_SEED = 5150


def _shapes_fit(shapes) -> bool:
    """Check Python-integer shape arithmetic before NumPy can allocate."""
    limit = int(np.iinfo(np.intp).max)
    total = 0
    for shape in shapes:
        if any(d < 0 or d > limit for d in shape):
            return False
        total += math.prod(shape)
        if total > MAX_ELEMENTS:
            return False
    return True


def _aspect_shape(a, factor, grow_axis, cut_axis=None):
    factor = operator.index(factor)
    shape = list(a.shape)
    if cut_axis is not None:
        n = shape[cut_axis]
        shape[cut_axis] = min(n, max(1, n // factor))
    if grow_axis is not None:
        shape[grow_axis] *= factor
    return tuple(shape)


def _scaled(args: tuple, mult: int, div: int = 1) -> tuple | None:
    """Every array argument on the SIZE axis, grown by `mult`.

    Which axis is the size axis is not knowable in general, so the rule is:
    grow the leading axis of every ndarray whose leading axis is already the
    longest among the arguments. That keeps paired operands in step
    (solve's a and b, a binary ufunc's two inputs) while leaving small
    parameter arrays alone - a 128-element quantile vector beside a
    1e6-element sample must not be "scaled" into something else entirely.

    GROWTH IS A RESAMPLE, NOT A TILE, and that distinction is the whole
    reason this comment is long. `np.concatenate([a] * mult)` was the
    original and it does not make a bigger input - it makes the SAME input,
    repeated. For anything whose cost depends on the value distribution
    that is a different measurement wearing the size axis' name:

    - It adds no new distinct values, so cardinality stays pinned at the
      canonical fixture's while n grows. Batch 16 chased a phantom
      signed/unsigned difference for an evening because of this;
      `unique_values_sort#uint64*30` was never a 30,000-element input, it
      was 1,000 values repeated thirty times.
    - It is PERFECTLY PERIODIC, and periodicity has its own cost. Measured
      at equal multiset, n=30,000, ~1,000 distinct: tiled 1.31x, the same
      values shuffled 2.33x, redrawn 1.95x (int64). So a tiled cell reports
      a number a caller only meets if they passed np.tile.
    - It DESTROYS SORTEDNESS. A tiled sorted array is a sawtooth, so
      `searchsorted*30` was scaling its haystack into something that is not
      a haystack at all. `_reshaped_1d` already takes care over exactly
      this; the size axis did not.

    Resampling with replacement from the array's own values fixes the last
    two outright and preserves the empirical distribution. It does NOT fix
    the first: the value pool is still the canonical fixture's, so
    cardinality remains capped. Where a fixture knows how to build itself at
    a size - every `#row` cell does, since its maker takes the row's floor -
    the caller regenerates instead and gets cardinality that scales with n
    honestly. This path is the fallback for fixtures that cannot.

    Deterministic by construction: a fixed seed, so a cell measured twice is
    the same cell.
    """
    mult, div = operator.index(mult), operator.index(div)
    if mult < 1 or div < 1:
        raise ValueError("scale factors must be positive integers")
    if mult == 1 and div == 1:
        return args
    arrays = [a for a in args if isinstance(a, np.ndarray) and a.ndim >= 1]
    if not arrays:
        return None
    lead = max(a.shape[0] for a in arrays)
    shapes = []
    for a in arrays:
        if a.shape[0] == lead:
            rows = min(lead, max(1, lead // div)) if div > 1 else lead * mult
            shapes.append((rows, *a.shape[1:]))
    # Preserve the existing cap's coverage: it counts the arrays being
    # resized, not untouched parameter arrays. No copies have happened yet.
    if not _shapes_fit(shapes):
        return None
    out = []
    for a in args:
        if isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[0] == lead:
            if div > 1:
                grown = np.ascontiguousarray(a[: max(1, a.shape[0] // div)])
            else:
                grown = _resample_grown(a, mult)
            out.append(grown)
        else:
            out.append(a)
    return tuple(out)


def _resample_grown(a: np.ndarray, mult: int) -> np.ndarray:
    """Resample one array with replacement to `mult` times its length.

    KEEP THE LAYOUT. ascontiguousarray forces C order, and a path gated on
    F-contiguity then stops dispatching at every size - relayout_blocked's
    fixture is a transposed view, and six of its size cells went from
    1.78-2.68x to no-dispatch before compare_sweep_logs.py flagged them.
    Tiling happened to preserve it; an explicit copy does not unless it is
    asked to.

    A sorted operand that stops being sorted is a different call, not a
    bigger one - searchsorted's haystack is the case that matters here.
    """
    rng = np.random.default_rng(_SCALE_SEED + a.shape[0])
    take = rng.integers(0, a.shape[0], size=a.shape[0] * mult)
    order = "F" if (a.flags.f_contiguous and not a.flags.c_contiguous) else "C"
    grown = np.asarray(a[take], order=order)
    if a.ndim == 1 and a.size > 1 and bool(np.all(a[1:] >= a[:-1])):
        grown.sort()
    return grown


def _revalued(args: tuple, mode: str) -> list[tuple]:
    """The same shapes, at the two ends of the CARDINALITY axis.

    Shape is not the only thing a sorting or hashing path's cost depends on,
    and these modules say so themselves. unique_sort's own comment records
    int16 at 8.5-21.6x high-cardinality but 1.22-1.48x at n=1000 low, and
    intersect_sorted's records low-cardinality inputs "marginal-to-losing"
    where high-cardinality ones win 1.85-10.1x. Both numbers came from a
    calibration battery; neither has ever had a cell in the safety sweep,
    which builds one fixture per path and takes whatever distribution its
    author happened to draw.

    'd' draws from 16 distinct values, so almost every element is a
    duplicate. 'u' draws across the dtype's whole range, which is as close
    to all-distinct as the dtype allows - and for int8 that is only ~256
    values, which is the honest answer for int8 rather than a failure.

    INTEGER operands only. A float array in these fixtures is usually a
    matrix whose VALUES are load-bearing (positive-definite, well
    conditioned, sorted grid), and redrawing it produces a different call
    rather than a different distribution. Conditioning is a real axis and
    it is not this one.
    """
    rng = np.random.default_rng(_ROW_SEED + 11)
    out, touched = [], False
    for a in args:
        if (isinstance(a, np.ndarray) and a.dtype.kind in "iu"
                and a.size > 1 and a.ndim >= 1):
            info = np.iinfo(a.dtype)
            if mode == "d":
                vals = rng.integers(0, min(16, int(info.max)) + 1, size=a.size)
            else:
                vals = rng.integers(max(int(info.min), -(2 ** 40)),
                                    min(int(info.max), 2 ** 40), size=a.size)
            vals = vals.astype(a.dtype).reshape(a.shape)
            # a sorted operand stays sorted, or the call means something else
            if a.ndim == 1 and a.size > 1 and bool(np.all(a[1:] >= a[:-1])):
                vals = np.sort(vals)
            out.append(vals)
            touched = True
        else:
            out.append(a)
    return [tuple(out)] if touched else []


def _reshaped_1d(args: tuple, f: int, first_heavy: bool) -> list[tuple]:
    """The aspect axis of a pair of 1-D operands: their RATIO.

    A 2-D cell's aspect is rows against columns. A pair of 1-D arrays has
    exactly the same degree of freedom - one operand against the other, at
    constant total volume - and the size sweep cannot reach it either,
    because scaling moves both operands together by construction.

    This is not hypothetical. searchsorted_sortqueries claims a CACHE
    mechanism, so what it needs is a big HAYSTACK, and its gate asks for a
    big QUERY COUNT with only a flat floor on the haystack. Batch 16
    measured its int64 row climbing 0.26x -> 0.74x purely as the haystack
    grew under a fixed query count - the whole regime lives on this axis,
    and every cell of the sweep held both operands in step.

    Only same-dtype pairs move: a paired operand is one the op consumes
    alongside the first, whereas an index array or a quantile vector is a
    parameter, and stretching a parameter produces a different call rather
    than a different shape of the same one.
    """
    idx = [i for i, a in enumerate(args)
           if isinstance(a, np.ndarray) and a.ndim == 1]
    if len(idx) != 2:
        return []
    i, j = idx
    a, b = args[i], args[j]
    if a.dtype != b.dtype:
        return []
    big, small = (i, j) if first_heavy else (j, i)
    shapes = []
    for k, x in enumerate(args):
        if isinstance(x, np.ndarray):
            if k == big:
                shapes.append((int(x.size) * f,))
            elif k == small:
                shapes.append((min(x.size, max(1, x.size // f)),))
            else:
                shapes.append(x.shape)
    if not _shapes_fit(shapes):
        raise OverflowError("shape cell exceeds allocation limit")

    def grow(x):
        return np.concatenate([x] * f)

    def cut(x):
        return np.ascontiguousarray(x[: max(1, x.size // f)])

    out = list(args)
    out[big] = grow(args[big])
    out[small] = cut(args[small])
    # a sorted operand must stay sorted or the call means something else -
    # searchsorted's haystack is the case, and a tiled concatenation of a
    # sorted array is not sorted
    for k in (big, small):
        if isinstance(args[k], np.ndarray) and args[k].size > 1:
            orig = args[k]
            if bool(np.all(orig[1:] >= orig[:-1])):
                out[k] = np.sort(out[k])
    total = sum(x.size for x in out if isinstance(x, np.ndarray))
    return [tuple(out)] if total <= MAX_ELEMENTS else []


def _reshaped(args: tuple, f: int, trail_heavy: bool) -> list[tuple]:
    """Candidate reshapes of `args` at ~constant volume, aspect moved by `f`.

    trail_heavy grows every trailing axis f-fold and shrinks the leading
    axis f-fold (long rows / long contraction, few of them); the reverse
    direction does the opposite. Two rules, tried in turn, because operand
    coupling is not knowable in general:

    - shared-trailing: every array whose last axis matches the longest last
      axis moves together. That keeps np.inner's stacked operands and a
      matrix-vector product valid, and leaves small parameter arrays (a
      quantile vector beside its sample) alone.
    - chain: two 2-D operands with a.shape[-1] == b.shape[0] are a product
      chain, so the SHARED inner dimension is what grows while the outer
      dimensions shrink - the shared-trailing rule would break the chain.

    A rule that produces an input the op cannot run, or that the path's
    predicate refuses, costs a printed skip and nothing else.
    """
    f = operator.index(f)
    if f < 1:
        raise ValueError("aspect factor must be a positive integer")
    arrays = [a for a in args if isinstance(a, np.ndarray) and a.ndim >= 1]
    if not any(a.ndim >= 2 for a in arrays):
        return _reshaped_1d(args, f, trail_heavy)

    t = max(a.shape[-1] for a in arrays)
    shared = _reshaped_shared_trailing(args, f, trail_heavy, t)
    variants = [shared] if shared is not None else []

    chain = _reshaped_chain(args, f, trail_heavy)
    if chain is not None:
        variants.append(chain)
    if not variants:
        raise OverflowError("shape cell exceeds allocation limit")
    return variants


def _grow_axis(a: np.ndarray, f: int, axis: int) -> np.ndarray:
    return np.concatenate([a] * f, axis=axis)


def _cut_axis(a: np.ndarray, f: int, axis: int) -> np.ndarray:
    idx = [slice(None)] * a.ndim
    idx[axis] = slice(0, max(1, a.shape[axis] // f))
    return np.ascontiguousarray(a[tuple(idx)])


def _reshaped_shared_trailing(args: tuple, f: int, trail_heavy: bool, t: int) -> tuple | None:
    """The shared-trailing variant: every array whose last axis matches the
    longest last axis (`t`) moves together - see `_reshaped`."""
    shapes = []
    for a in args:
        if not isinstance(a, np.ndarray):
            continue
        if a.ndim >= 1 and a.shape[-1] == t:
            if trail_heavy:
                shape = _aspect_shape(a, f, -1, 0 if a.ndim >= 2 else None)
            else:
                shape = _aspect_shape(a, f, 0 if a.ndim >= 2 else None, -1)
            shapes.append(shape)
        else:
            shapes.append(a.shape)
    if not _shapes_fit(shapes):
        return None
    out = []
    for a in args:
        if isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[-1] == t:
            if trail_heavy:
                # The axes are independent: crop before replicating so the
                # intermediate never grows beyond the already-checked result.
                if a.ndim >= 2:
                    a = _cut_axis(a, f, 0)
                a = _grow_axis(a, f, -1)
            else:
                a = _cut_axis(a, f, -1)
                if a.ndim >= 2:
                    a = _grow_axis(a, f, 0)
        out.append(a)
    return tuple(out)


def _reshaped_chain(args: tuple, f: int, trail_heavy: bool) -> tuple | None:
    """The product-chain variant: two 2-D operands with a.shape[-1] ==
    b.shape[0] grow/shrink on the SHARED inner dimension - see `_reshaped`.
    None when `args` does not contain such a chain.
    """
    two = [a for a in args if isinstance(a, np.ndarray) and a.ndim == 2]
    if len(two) != 2 or two[0].shape[-1] != two[1].shape[0]:
        return None
    a, b = two
    if trail_heavy:
        plan = {id(a): _aspect_shape(a, f, -1, 0), id(b): _aspect_shape(b, f, 0, -1)}
    else:
        plan = {id(a): _aspect_shape(a, f, 0, -1), id(b): _aspect_shape(b, f, -1, 0)}
    if not _shapes_fit(plan.get(id(x), x.shape) for x in args if isinstance(x, np.ndarray)):
        return None
    if trail_heavy:
        na, nb = _grow_axis(_cut_axis(a, f, 0), f, -1), _grow_axis(_cut_axis(b, f, -1), f, 0)
        if b.flags.f_contiguous and not b.flags.c_contiguous and nb.shape[1] == 1:
            # The old grow-then-crop route retained this unused F-order
            # column stride. Only a singleton stride changes: no accessed
            # address moves, and no large intermediate is needed.
            nb = np.lib.stride_tricks.as_strided(
                nb, shape=nb.shape, strides=(nb.strides[0], nb.shape[0] * nb.itemsize))
    else:
        na, nb = _grow_axis(_cut_axis(a, f, -1), f, 0), _grow_axis(_cut_axis(b, f, 0), f, -1)
    repl = {id(a): na, id(b): nb}
    return tuple(repl.get(id(x), x) for x in args)


# ---------------------------------------------------------------------------
# --rows: one cell per SHIPPED CALIBRATION ROW.
#
# The gap this closes is the same one batch 14 found one axis further in, and
# it is structural rather than accidental. A path's dispatch gate is usually a
# TABLE - a floor per matrix dimension, a window per dtype, a work floor per
# convolution mode - and every row of that table is a separately measured,
# separately shipped promise. The sweep, however, builds its input from the
# path's ONE selfcheck fixture, so it judges whichever row that fixture
# happens to land on and no other. Every small-matrix linalg fixture is d=3;
# det, slogdet, cholesky, qr, inv, pinv, svdvals and norm2 therefore shipped
# their d=2 and d=4 rows with no end-to-end measurement behind them at all,
# and those are DIFFERENT closed-form kernels, not the same code at another
# size. The dtype-keyed tables are the same story: unique_sort's small-int
# rows go through a radix route the int64 fixture never touches.
#
# So the cells here are derived FROM THE SHIPPED TABLE, never hand-written. A
# row added to a table becomes a cell with no edit to this file, and a table
# no axis claims fails _table_audit() loudly rather than passing in silence -
# which is the only way a coverage instrument can be believed later.
#
# Where a row is a WINDOW, both ends are judged: "#3" at the floor and "#3^"
# at the cap. That is not symmetry for its own sake - three of batch 14's
# five losses were at the TOP of a window, where scaling up from the floor by
# a round factor does not necessarily land.
# ---------------------------------------------------------------------------

_ROW_SEED = 91


def _sq(batch: int, d: int, dtype=np.float64, spd: bool = False, seed: int = _ROW_SEED):
    """A (batch, d, d) stack, the shape every small-matrix linalg row wants."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((int(batch), int(d), int(d)))
    if spd:
        a = a @ np.swapaxes(a, -1, -2) + float(d) * np.eye(int(d))
    return np.ascontiguousarray(a.astype(dtype, copy=False))


def _ints(rng, n, dtype, span: int = 1_000_000):
    """Random integers that fit `dtype`, whatever its width."""
    info = np.iinfo(dtype)
    hi = min(span, int(info.max))
    lo = max(-hi, int(info.min))
    return rng.integers(lo, hi + 1, size=int(n)).astype(dtype)


def _window_rows(table, key_of, build):
    """(label, maker) per row of a {key: (lo, hi)} window table, at BOTH ends."""
    out = []
    for key, window in sorted(table.items(), key=lambda kv: str(kv[0])):
        lo, hi = window if isinstance(window, tuple) else (window, None)
        label = key_of(key)
        out.append((label, build(key, lo)))
        if hi is not None:
            out.append((f"{label}^", build(key, hi)))
    return out


def _axes_linalg() -> dict[str, list]:
    """det / slogdet / solve: one window per (kind, d)."""
    from pyoverdrive.fastpaths import linalg_small_batch

    axes: dict[str, list] = {}

    def _linalg(kind: str, path: str):
        def build(key, batch):
            d = key[1]

            def make(batch=batch, d=d):
                a = _sq(batch, d)
                if kind != "solve":
                    return (a,), {}
                rng = np.random.default_rng(_ROW_SEED + 1)
                return (a, rng.standard_normal((int(batch), d, 1))), {}

            return make

        rows = {k: v for k, v in linalg_small_batch._WINDOWS.items() if k[0] == kind}
        axes[path] = _window_rows(rows, lambda k: str(k[1]), build)

    _linalg("det", "det_small_batch")
    _linalg("slogdet", "slogdet_small_batch")
    _linalg("solve", "solve_small_batch")
    return axes


def _axes_inv() -> dict[str, list]:
    """inv: floors keyed by (d, dtype) - the only dtype-keyed table outside
    the PyRallel families, and its float32 row had never been measured."""
    from pyoverdrive.fastpaths import inv_small_batch

    return {
        "inv_small_batch": [
            (f"{d}-{np.dtype(dt).name}",
             (lambda d=d, dt=dt, lo=lo: ((_sq(lo, d, dt, spd=True),), {})))
            for (d, dt), lo in sorted(inv_small_batch._FLOORS.items(),
                                      key=lambda kv: str(kv[0]))
        ]
    }


def _axes_cholesky_qr() -> dict[str, list]:
    from pyoverdrive.fastpaths import cholesky_small_batch, qr_small_batch

    return {
        "cholesky_small_batch": _window_rows(
            cholesky_small_batch._WINDOWS, str,
            lambda d, batch: (lambda d=d, batch=batch: ((_sq(batch, d, spd=True),), {}))),
        "qr_small_batch": _window_rows(
            qr_small_batch._FLOORS, str,
            lambda d, batch: (lambda d=d, batch=batch: ((_sq(batch, d),), {}))),
    }


def _axes_svd_family() -> dict[str, list]:
    """pinv / svdvals / norm2 share one batch floor and one d axis."""
    from pyoverdrive.fastpaths import svd_small_batch

    axes: dict[str, list] = {}
    for path, kw in (("pinv_small_batch", {}),
                     ("svdvals_small_batch", {"compute_uv": False}),
                     ("norm2_small_batch", {"ord": 2, "axis": (-2, -1)})):
        axes[path] = [
            (str(d), (lambda d=d, kw=kw: ((_sq(svd_small_batch.BATCH_MIN, d),), dict(kw))))
            for d in sorted(svd_small_batch.SVDVALS_SIGMA_RATIO_MIN)
        ]
    return axes


def _axes_isclose() -> dict[str, list]:
    """isclose: _CAPS is a ceiling per dtype, so the row's own edge is the TOP."""
    from pyoverdrive.fastpaths import isclose_fused

    def _isclose(dtype, cap):
        def make():
            rng = np.random.default_rng(_ROW_SEED + 2)
            a = rng.uniform(-10.0, 10.0, size=int(cap)).astype(dtype)
            b = (a + rng.uniform(-1e-7, 1e-7, size=int(cap))).astype(dtype)
            return (a, b), {}
        return make

    return {
        "isclose_fused": [
            (f"{np.dtype(dt).name}^", _isclose(dt, cap))
            for dt, cap in sorted(isclose_fused._CAPS.items(), key=lambda kv: str(kv[0]))
        ]
    }


def _axes_fft() -> dict[str, list]:
    """convolve / correlate: a dtype axis AND a keyword axis (mode), which is
    the class batch 15 named as still uncovered. hist2d's 0.75x loss was a
    threshold on the wrong axis of exactly this kind."""
    from pyoverdrive.fastpaths import fftconvolve

    def _fft(dtype, mode, work):
        spec = fftconvolve.SUPPORTED[dtype]
        m = spec.min_len
        # the shipped gate models 'valid' as (hi - lo + 1) * lo, not n * m,
        # because a near-equal-length valid call is nearly free on stock. Size
        # the row from the MODE'S OWN work model or the cell lands under the
        # floor and proves nothing.
        n = (m + max(0, -(-int(work) // m) - 1) if mode == "valid"
             else -(-int(work) // m))
        n = max(m, n)

        def make():
            rng = np.random.default_rng(_ROW_SEED + 3)
            if np.dtype(dtype).kind == "f":
                a = (rng.random(n) + 1.0).astype(dtype)
                v = (rng.random(spec.min_len) + 1.0).astype(dtype)
            else:
                a = _ints(rng, n, dtype, span=100)
                v = _ints(rng, spec.min_len, dtype, span=100)
            return (a, v), {"mode": mode}
        return make

    axes: dict[str, list] = {}
    for path in ("fftconvolve", "fftcorrelate"):
        entries = []
        for dt in sorted(fftconvolve.SUPPORTED, key=str):
            entries.append((np.dtype(dt).name,
                            _fft(dt, "full", fftconvolve._MODE_WORK_FLOOR["full"])))
        base = np.dtype(np.float64) if path == "fftconvolve" else np.dtype(np.int64)
        for mode, work in sorted(fftconvolve._MODE_WORK_FLOOR.items()):
            entries.append((f"mode-{mode}", _fft(base, mode, work)))
        axes[path] = entries
    return axes


def _axes_searchsorted() -> dict[str, list]:
    from pyoverdrive.fastpaths import searchsorted_sortqueries

    def _searchsorted(dtype, floor):
        def make():
            rng = np.random.default_rng(_ROW_SEED + 4)
            n = max(int(floor), searchsorted_sortqueries._HAYSTACK_FLOOR)
            if np.dtype(dtype).kind in "iu":
                x = np.sort(_ints(rng, n, dtype))
                v = _ints(rng, n, dtype)  # random order: passes the disorder gate
            else:
                x = np.sort(rng.standard_normal(n).astype(dtype))
                v = rng.standard_normal(n).astype(dtype)
            return (x, v), {}
        return make

    return {
        "searchsorted_sortqueries": [
            (np.dtype(dt).name, _searchsorted(dt, floor))
            for dt, floor in sorted(searchsorted_sortqueries.SUPPORTED.items(), key=str)
        ]
    }


def _axes_int_matmul() -> dict[str, list]:
    from pyoverdrive.fastpaths import matmul_int_blas

    def _int_matmul(dtype):
        def make():
            rng = np.random.default_rng(_ROW_SEED + 5)
            n = matmul_int_blas.MIN_DIM  # the row's OWN edge, not twice it
            x = _ints(rng, n * n, dtype, span=1_000).reshape(n, n)
            y = _ints(rng, n * n, dtype, span=1_000).reshape(n, n)
            return (x, y), {}
        return make

    axes: dict[str, list] = {}
    for path in ("matmul_int_blas", "dot_int_blas"):
        axes[path] = [(np.dtype(dt).name, _int_matmul(dt))
                      for dt in sorted(matmul_int_blas._BOUNDS, key=str)]
    return axes


def _axes_split_complex() -> dict[str, list]:
    from pyoverdrive.fastpaths import matmul_split_complex

    def _split_complex(cdt, rdt, axis=None, offset=0, fixed_m=None):
        def make():
            rng = np.random.default_rng(_ROW_SEED + 6)
            # every one of the three at its OWN limit: the widest complex
            # operand the gate admits against the narrowest product it will
            # take. A cell in the comfortable middle tests no edge at all.
            m = matmul_split_complex.M_MAX if fixed_m is None else fixed_m
            n = matmul_split_complex.N_MIN
            q = matmul_split_complex.Q_MIN
            # Coherent dimensions keep the contraction valid. Generic array
            # scaling changes only one operand and produces invalid-input
            # skips, which provide no evidence about these shape boundaries.
            m += offset if axis == "m" else 0
            n += offset if axis == "n" else 0
            q += offset if axis == "q" else 0
            c = (rng.uniform(0.5, 1.5, size=(m, n))
                 + 1j * rng.uniform(0.5, 1.5, size=(m, n))).astype(cdt)
            r = rng.uniform(0.5, 1.5, size=(n, q)).astype(rdt)
            return (c, r), {}
        return make

    entries = []
    for cdt, rdt in sorted(matmul_split_complex._PAIRS.items(), key=str):
        label = np.dtype(cdt).name
        entries.append((label, _split_complex(cdt, rdt)))
        for axis in ("m", "n", "q"):
            for offset in (-1, 1):
                entries.append((f"{label}-{axis}{offset:+d}",
                                _split_complex(cdt, rdt, axis, offset)))
        # Replay the row-count grid independently of any later gate change.
        # NumPy 2.3.0 lost at m=255 while m=256 won: measure a candidate
        # narrower regime and its neighbors before changing M_MAX. Select
        # just these fresh-process cells with --rows --only '#complex128-rows-'.
        for m in (1, 16, 32, 63, 64, 65, 128, 192, 255, 256):
            entries.append((f"{label}-rows-m{m}",
                            _split_complex(cdt, rdt, fixed_m=m)))
        for axis in ("n", "q"):
            for offset in (-1, 1):
                entries.append((f"{label}-rows-m64-{axis}{offset:+d}",
                                _split_complex(cdt, rdt, axis, offset, fixed_m=64)))
    return {"matmul_split_complex": entries}


def _axes_hist2d() -> dict[str, list]:
    from pyoverdrive.fastpaths import hist2d_uniform

    def _samples(offset):
        def make():
            rng = np.random.default_rng(29)  # same distribution as canonical
            n = hist2d_uniform.SAMPLES_MIN + offset
            return (rng.normal(size=n), rng.normal(size=n)), {
                "bins": [40, 40], "range": [[-3.0, 3.0], [-3.0, 3.0]]}
        return make

    return {"hist2d_uniform": [
        (f"samples{offset:+d}", _samples(offset)) for offset in (-1, 0, 1)]}


def _axes_relayout() -> dict[str, list]:
    from pyoverdrive.fastpaths import relayout_blocked

    def _relayout(dtype, floor):
        def make():
            # +1 off the exact square root, for the same reason the selfcheck
            # fixture is: a power-of-two side is this path's best case, not
            # its typical one, and a row cell that lands there would report
            # 3.58x for a row whose honest worst is 1.44x.
            n = int(int(floor) ** 0.5) + 1
            rng = np.random.default_rng(_ROW_SEED + 7)
            a = rng.standard_normal((n, n))
            if np.dtype(dtype).kind in "iu":
                a = (a * 1000).astype(dtype)
            else:
                a = a.astype(dtype)
            return (a.T,), {}  # F-contiguous view, as the fixture does
        return make

    return {
        "relayout_blocked": [
            (np.dtype(dt).name, _relayout(dt, floor))
            for dt, floor in sorted(relayout_blocked.SUPPORTED.items(), key=str)
        ]
    }


def _axes_unique_sort() -> dict[str, list]:
    """SCALE-AWARE. A maker that takes `scale` is asked to BUILD the bigger
    input rather than let _scaled resample the small one, so the number of
    distinct values grows with n the way a real caller's array does.
    Without it the size axis holds cardinality pinned at the floor's, and
    for these three paths cardinality is the axis that decides the answer."""
    from pyoverdrive.fastpaths import unique_sort

    def _unique(dtype, floor, kwargs):
        def make(scale: int = 1):
            rng = np.random.default_rng(_ROW_SEED + 8)
            return (_ints(rng, int(floor) * scale, dtype),), dict(kwargs)
        return make

    axes: dict[str, list] = {}
    for path, kw in (("unique_sort", {}), ("unique_values_sort", {})):
        axes[path] = [(np.dtype(dt).name, _unique(dt, floor, kw))
                      for dt, floor in sorted(unique_sort._THRESHOLDS.items(), key=str)]
    return axes


def _axes_intersect() -> dict[str, list]:
    from pyoverdrive.fastpaths import intersect_sorted

    def _intersect(dtype, floor):
        def make(scale: int = 1):
            rng = np.random.default_rng(_ROW_SEED + 9)
            half = max(2, -(-int(floor) // 2)) * scale
            return (_ints(rng, half, dtype), _ints(rng, half, dtype)), {}
        return make

    return {
        "intersect_sorted": [
            (np.dtype(dt).name, _intersect(dt, floor))
            for dt, floor in sorted(intersect_sorted._THRESHOLDS.items(), key=str)
        ]
    }


def _axes_char_view() -> dict[str, list]:
    """char views: 'U' and 'S' are different int views of different widths,
    and the fixtures are all 'U'."""
    from pyoverdrive.fastpaths import char_view

    def _char(kind, n, kwargs):
        def make():
            rng = np.random.default_rng(_ROW_SEED + 10)
            alphabet = np.array(list("ASDFGHJKLZ"), dtype=f"{kind}1")
            return (alphabet[rng.integers(0, 10, size=int(n))],), dict(kwargs)
        return make

    # unique's floor depends on WHICH RETURN was asked for - counts, index or
    # inverse, or plain - and each branch has its own measured number. That is
    # a gate axis as much as any dtype is, and only the counts branch had a
    # fixture. The counts floor is the only one keyed by 'U'/'S'; the other
    # two are single constants that still need one cell per kind, because the
    # int view underneath them is 4 bytes wide for 'U' and 1 for 'S'.
    unique_branches = [("counts", {"return_counts": True}, None),
                       ("index", {"return_index": True}, char_view.UNIQUE_IDXINV_FLOOR),
                       ("plain", {}, char_view.UNIQUE_PLAIN_FLOOR)]
    return {
        "unique_char_view": [
            (kind if branch == "counts" else f"{kind}-{branch}",
             _char(kind, floor if floor is not None
                   else char_view.UNIQUE_COUNTS_FLOOR[kind], kwargs))
            for kind in sorted(char_view.UNIQUE_COUNTS_FLOOR)
            for branch, kwargs, floor in unique_branches
        ],
        "sort_char_view": [
            (kind, _char(kind, char_view.SORT_FLOOR, {}))
            for kind in sorted(char_view.UNIQUE_COUNTS_FLOOR)
        ],
    }


def _ROW_AXES() -> dict[str, list]:
    """path name -> [(row label, maker)], read from the live shipped tables.

    Split into one helper per path family, each an independent block that
    only adds its own keys - the assembly order below is the same order
    those blocks ran in before the split, so the merged dict is identical.
    """
    axes: dict[str, list] = {}
    axes.update(_axes_linalg())
    axes.update(_axes_inv())
    axes.update(_axes_cholesky_qr())
    axes.update(_axes_svd_family())
    axes.update(_axes_isclose())
    axes.update(_axes_fft())
    axes.update(_axes_searchsorted())
    axes.update(_axes_int_matmul())
    axes.update(_axes_split_complex())
    axes.update(_axes_hist2d())
    axes.update(_axes_relayout())
    axes.update(_axes_unique_sort())
    axes.update(_axes_intersect())
    axes.update(_axes_char_view())
    return axes


# Tables that are NOT dispatch gates, each with the reason it is exempt. The
# audit below fails on anything that is neither claimed by an axis above nor
# named here, so a new gate table cannot slip in unswept.
_EXEMPT_TABLES = {
    "char_view._INT_VIEW": "mechanism, keyed by the U/S axis unique_char_view already sweeps",
    "fftconvolve._MODE_WORK_FLOOR": "claimed as fftconvolve/fftcorrelate's mode axis",
    "inv_small_batch._FOLD_FROM": "internal fold crossover, reached through the det/slogdet d rows",
    "intersect_sorted._THRESHOLDS": "claimed as intersect_sorted's dtype axis",
    "linalg_small_batch._WINDOWS": "claimed as det/slogdet/solve's (kind, d) axis",
    "matmul_int_blas._BOUNDS": "claimed as matmul_int_blas/dot_int_blas's dtype axis",
    "nanreduce_scan._FLOORS": "one registered path per key; each already has its own cell",
    "nanreduce_scan._PLAIN": "stock function map, not a threshold",
    "parallel_binary.SHIPPED": "pristine pre-calibration copy of SUPPORTED, not a live gate",
    "parallel_binary.SUPPORTED": "already one cell per dtype via the @ axis",
    "parallel_ufunc.SHIPPED": "pristine pre-calibration copy of SUPPORTED, not a live gate",
    "parallel_ufunc.SUPPORTED": "already one cell per dtype via the @ axis",
    "searchsorted_extreme_key._BOUNDS_CACHE": "runtime cache, not a threshold",
    "svd_small_batch.SVDVALS_SIGMA_RATIO_MIN": "claimed as pinv/svdvals/norm2's d axis",
    "unique_sort._THRESHOLDS": "claimed as unique_sort/unique_values_sort's dtype axis",
}

_CLAIMED_TABLES = {
    "char_view.UNIQUE_COUNTS_FLOOR", "cholesky_small_batch._WINDOWS",
    "fftconvolve.SUPPORTED", "inv_small_batch._FLOORS", "isclose_fused._CAPS",
    "matmul_split_complex._PAIRS", "qr_small_batch._FLOORS",
    "relayout_blocked.SUPPORTED", "searchsorted_sortqueries.SUPPORTED",
}


def _table_audit() -> list[str]:
    """Every multi-row table in the package, minus the ones an axis covers.

    A coverage instrument that quietly skips what it does not understand
    reads exactly like one that found nothing, so this is loud by design: an
    unclaimed table is an error, not a note.
    """
    import importlib
    import pkgutil

    from pyoverdrive import fastpaths

    unclaimed = []
    for mod_info in pkgutil.iter_modules(fastpaths.__path__):
        mod = importlib.import_module(f"pyoverdrive.fastpaths.{mod_info.name}")
        for attr, val in vars(mod).items():
            if attr.startswith("__") or not isinstance(val, dict) or len(val) < 2:
                continue
            if attr in ("_PROVENANCE",):
                continue
            qual = f"{mod_info.name}.{attr}"
            if qual in _EXEMPT_TABLES or qual in _CLAIMED_TABLES:
                continue
            unclaimed.append(f"{qual} ({len(val)} rows)")
    return sorted(unclaimed)


def _row_cells() -> dict:
    """cell name -> maker, one per shipped calibration row."""
    return {f"{path}#{label}": maker
            for path, entries in _ROW_AXES().items()
            for label, maker in entries}


def _takes_scale(make) -> bool:
    """Does this maker know how to build its input at a different size?

    Asked of the maker rather than kept in a list beside it, so a maker
    that gains the parameter is used without a second edit somewhere else -
    the same reason the row cells are derived from the shipped tables.
    """
    try:
        return "scale" in inspect.signature(make).parameters
    except (TypeError, ValueError):
        return False


def _inputs_for(cell: str):
    """(path name, maker) for a cell, expanding a "path@dtype" cell onto the
    row's own floor rather than whatever the selfcheck input would pick, and a
    "path#row" cell onto that shipped table row's own edge."""
    cell = re.split(r"[*/><%]", cell, maxsplit=1)[0]
    if "#" in cell:
        return cell.split("#", 1)[0], _row_cells().get(cell)
    if "@" not in cell:
        return cell, D._selfcheck_inputs().get(cell)
    name, dtype_name = cell.split("@", 1)
    from pyoverdrive.fastpaths import parallel_binary, parallel_ufunc

    op = name[len("pyrallel_"):]
    for mod, builder in ((parallel_ufunc, D._inputs_ufunc),
                         (parallel_binary, D._inputs_binary)):
        row = mod.SUPPORTED.get(op)
        if row is None:
            continue
        want = {d: n for d, n in row.items() if np.dtype(d).name == dtype_name}
        if want:
            return name, builder(op, want)
    return name, None


def _op_of(name: str) -> str | None:
    """The numpy op a registered path serves, or None if it is not a FastPath
    (the vectorize ClassPath has no entry here)."""
    for lst in GEARBOX._paths.values():
        for p in (lst if isinstance(lst, list) else [lst]):
            if p.name == name:
                return p.op
    return None


def _cell_variants(cell: str, call_args: tuple, make) -> tuple[list | None, str | None]:
    """(variants, skip_reason) for a cell's dispatch-input variants, expanding
    a %d/%u (revalue) or */></> (rescale/reshape) marker when the cell name
    carries one. `skip_reason` is None on success."""
    variants = [call_args]
    value_marker = re.search(r"%([du])$", cell)
    if value_marker:
        variants = _revalued(call_args, value_marker.group(1))
        if not variants:
            return None, "SKIP not-valued"
    marker = re.search(r"([*/><])(\d+)$", cell)
    if marker:
        kind, val = marker.group(1), int(marker.group(2))
        if kind in "*/":
            # PREFER REBUILDING over resampling. A maker that accepts a
            # scale knows how to draw its own input at any size, so the
            # cell gets cardinality that grows with n the way a caller's
            # array does. _scaled can only reuse the values it was handed,
            # which pins cardinality to the canonical fixture's and is
            # exactly the entanglement that made the size axis report on a
            # distribution it never intended to test.
            scaled = None
            if kind == "*" and _takes_scale(make):
                # The scale-aware unique/intersect makers grow every operand
                # linearly. Reject their predicted total before regeneration.
                predicted = sum(int(a.size) for a in call_args
                                if isinstance(a, np.ndarray)) * val
                if predicted > MAX_ELEMENTS:
                    return None, "SKIP too-big"
                try:
                    rebuilt, _ = make(scale=val)
                except Exception:  # noqa: BLE001
                    rebuilt = None
                if rebuilt is not None:
                    total = sum(a.size for a in rebuilt
                                if isinstance(a, np.ndarray))
                    if total > MAX_ELEMENTS:
                        return None, "SKIP too-big"
                    scaled = rebuilt
            if scaled is None:
                scaled = _scaled(call_args,
                                 val if kind == "*" else 1,
                                 val if kind == "/" else 1)
            if scaled is None:
                return None, "SKIP too-big"
            variants = [scaled]
        else:
            try:
                variants = _reshaped(call_args, val, kind == ">")
            except OverflowError:
                return None, "SKIP too-big"
            if not variants:
                return None, "SKIP not-shaped"
    return variants, None


def _input_description(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "strides": list(value.strides), "size": value.size}
    if isinstance(value, (list, tuple)):
        return [_input_description(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _dispatch_measure(name: str, op: str, variants: list, call_kwargs: dict,
                      evidence=None, *, rounds: int = 9, sample_seconds: float = 0.02) -> str:
    """Try each input variant against the live dispatch predicate and, for
    the first one that actually dispatches to `name`, return its ratio."""
    dispatched = False
    invalid = 0
    for variant, cargs in enumerate(variants):
        # DOES THE OP EVEN ACCEPT THIS? A reshaped variant the op cannot run
        # used to report as "no-dispatch", which is the report a well-gated
        # path produces - so a sweep whose cells silently failed to construct
        # would read exactly like a sweep that found nothing. They are now
        # different words. (Found by the 1-D pair axis: elementwise operands
        # are ratio-locked, and moving their ratio is a broadcast error, not
        # a refusal.)
        try:
            _consume(GEARBOX.stock_fn(op)(*cargs, **call_kwargs))
        except Exception:
            invalid += 1
            continue
        try:
            chosen, reason = GEARBOX.decide(op, cargs, call_kwargs)
            if evidence is not None:
                evidence.update(op=op, path=name, variant=variant,
                                dispatch={"chosen": chosen, "reason": reason})
            if chosen != name:
                continue
        except Exception:
            continue
        dispatched = True
        if evidence is None:
            ratio = _measure(op, cargs, call_kwargs, rounds=rounds, sample_seconds=sample_seconds)
        else:
            path = next(p for p in GEARBOX._paths[op] if p.name == name)
            evidence["inputs"] = _input_description(cargs)
            evidence["kwargs"] = {k: _input_description(v) for k, v in call_kwargs.items()}
            ratio = _measure(op, cargs, call_kwargs, rounds=rounds, evidence=evidence,
                             sample_seconds=sample_seconds,
                             comparison_mode=path.provenance.get("comparison_mode"))
        if ratio is None:
            continue
        return f"RATIO {ratio:.4f} {op}"
    if dispatched:
        return "SKIP unmeasurable"
    return "SKIP invalid-input" if invalid == len(variants) else "SKIP no-dispatch"


def _measure_one(cell: str, fast_under: float | None = None, evidence=None,
                 *, rounds: int = 9, sample_seconds: float = 0.02) -> str:
    """Measure a single cell. Runs as its own process; prints one line."""
    if fast_under is not None and cpuclass.probe_us() > fast_under:
        return "SLOW-CORE"
    pyoverdrive.enable()
    name, make = _inputs_for(cell)
    if make is None:
        return "SKIP no-input"
    try:
        call_args, call_kwargs = make()
    except Exception as exc:  # noqa: BLE001
        return f"SKIP input-error {exc!r}"

    variants, skip = _cell_variants(cell, call_args, make)
    if skip is not None:
        return skip

    paths = [
        p
        for lst in GEARBOX._paths.values()
        for p in (lst if isinstance(lst, list) else [lst])
        if p.name == name
    ]
    if not paths:
        return "SKIP unknown"
    op = paths[0].op

    return _dispatch_measure(name, op, variants, call_kwargs, evidence=evidence,
                             rounds=rounds, sample_seconds=sample_seconds)


def _pair_is_free(base: str, a: tuple, kwargs: dict) -> bool:
    """Are the two 1-D operands independent, or locked to equal length?

    Elementwise operands are locked - moving their ratio is a broadcast
    error, not a shape. Asked with a TINY pair rather than the real one,
    because the answer is about broadcast rules and the real operands
    can be 80M elements.
    """
    op = _op_of(base)
    if op is None:
        return False
    idx = [i for i, x in enumerate(a)
           if isinstance(x, np.ndarray) and x.ndim == 1]
    i, j = idx
    probe = list(a)
    probe[i] = np.sort(a[i][:8]) if a[i].size >= 8 else a[i]
    probe[j] = np.sort(a[j][:2]) if a[j].size >= 2 else a[j]
    try:
        _consume(GEARBOX.stock_fn(op)(*probe, **kwargs))
        return True
    except Exception:
        return False


def _has_aspect(cell: str, shaped_ok: dict[str, bool]) -> bool:
    """Does this cell have an aspect ratio that can move?

    Needs either a 2-D-or-deeper input (rows against columns) or a pair of
    same-dtype 1-D operands (one against the other). `shaped_ok` memoizes
    the answer per base cell across calls from the same sweep.
    """
    base = re.split(r"[*/><%]", cell, maxsplit=1)[0]
    if base not in shaped_ok:
        try:
            a, kwargs = _inputs_for(base)[1]()
            arrays = [x for x in a if isinstance(x, np.ndarray)]
            ones = [x for x in arrays if x.ndim == 1]
            shaped_ok[base] = bool(
                any(x.ndim >= 2 for x in arrays)
                or (len(ones) == 2 and ones[0].dtype == ones[1].dtype
                    and _pair_is_free(base, a, kwargs)))
        except Exception:
            shaped_ok[base] = False
    return shaped_ok[base]


def _cells_file_selection(args):
    if not getattr(args, "cells_file", None):
        return None
    if not hasattr(args, "_cells_file_selection"):
        raw = Path(args.cells_file).read_bytes()
        args._cells_file_selection = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "requested": [line.strip() for line in raw.decode("utf-8-sig").splitlines() if line.strip()],
        }
    return args._cells_file_selection


def _select_cells(args, live: set) -> tuple[list[str], int | None]:
    """Build the cell name list for this run: the canonical axis plus
    whichever of --sizes/--shapes/--rows/--values were asked for, filtered to
    live paths, audited when --rows is set, and narrowed by --only.

    Returns (names, exit_code); exit_code is not None only when the table
    audit found gate tables no axis claims, and the run must abort before
    measuring anything.
    """
    names = [c for c in cells(SIZE_MULTS if args.sizes else (),
                              SIZE_DIVS if args.sizes else (),
                              ASPECT_FACTORS if args.shapes else (),
                              rows=args.rows, values=args.values)
             if re.split(r"[@#*/><%]", c)[0] in live]

    # A row cell only means something if every gate table is either swept or
    # explicitly excused. Anything else and this tool would report full
    # coverage of a set it silently narrowed.
    if args.rows:
        unclaimed = _table_audit()
        if unclaimed:
            print("UNCOVERED GATE TABLES - no --rows axis claims these, and "
                  "none is listed in _EXEMPT_TABLES with a reason:")
            for t in unclaimed:
                print(f"  !! {t}")
            return names, 1

    # A shape cell needs an aspect ratio to move: either a 2-D-or-deeper
    # input (rows against columns) or a pair of same-dtype 1-D operands (one
    # against the other). Filtering the rest here saves a subprocess each,
    # and the count printed below stays a count of cells that could judge.
    shaped_ok: dict[str, bool] = {}
    names = [c for c in names if (">" not in c and "<" not in c)
             or _has_aspect(c, shaped_ok)]
    if args.only:
        names = [c for c in names if any(s in c for s in args.only)]
        print(f"--only {args.only}: {len(names)} cells - THIS IS A PROBE, not "
              f"a suite run; a green here says nothing about the rest")
    try:
        exact = _cells_file_selection(args)
    except (OSError, ValueError) as exc:
        print(f"NOT verified: cannot read --cells-file: {exc}")
        pyoverdrive.disable()
        return [], 1
    if exact is not None:
        requested = set(exact["requested"])
        unknown = requested.difference(names)
        if not requested or unknown:
            reason = (f"unknown cells for the current axes/filters: {', '.join(sorted(unknown))}"
                      if unknown else "--cells-file requests no cells")
            print(f"NOT verified: {reason}")
            pyoverdrive.disable()
            return [], 1
        names = [name for name in names if name in requested]
        print(f"--cells-file: {len(names)} exact cells selected; coverage is limited to this list")
    pyoverdrive.disable()
    return names, None


def _child_parts(proc, name: str) -> list[str]:
    """Validate the child protocol before interpreting a result as a skip."""
    output = (proc.stdout or "").strip()
    detail = (proc.stderr or output).strip()
    if proc.returncode:
        raise RuntimeError(f"cell {name} exited {proc.returncode}: {detail}")
    parts = (output.splitlines()[-1:] or [""])[0].split()
    if parts == ["SLOW-CORE"] or (len(parts) >= 2 and parts[0] == "SKIP"):
        return parts
    if len(parts) == 3 and parts[0] == "RATIO":
        try:
            ratio = float(parts[1])
        except ValueError:
            pass
        else:
            if math.isfinite(ratio) and ratio > 0:
                return parts
    raise RuntimeError(f"cell {name} returned no valid measurement or skip: {detail}")


def _validate_child_record(record, proc, name):
    parts = _child_parts(proc, name)
    if record.get("result") != " ".join(parts):
        raise ValueError("sidecar result does not match child stdout")
    if parts[0] != "RATIO":
        return

    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    correctness = record.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("passed") is not True:
        raise ValueError("ratio lacks passing correctness evidence")
    dispatch = record.get("dispatch")
    path = record.get("path")
    if (not isinstance(path, str) or not isinstance(dispatch, dict)
            or dispatch.get("chosen") != path
            or dispatch.get("fallback_observed") is not False
            or parts[2] not in (record.get("op"), path)):
        raise ValueError("ratio lacks matching dispatch evidence")
    rounds, iterations = record.get("rounds"), record.get("iterations_per_sample")
    if type(rounds) is not int or rounds < 1 or type(iterations) is not int or iterations < 1:
        raise ValueError("ratio lacks valid sample counts")
    medians = []
    for family in ("stock", "patched"):
        samples = record.get(f"{family}_seconds")
        median = record.get(f"{family}_median_seconds")
        if (not isinstance(samples, list) or len(samples) != rounds
                or not all(positive(sample) for sample in samples)):
            raise ValueError(f"ratio lacks valid {family} timing samples")
        measured = sorted(samples)[rounds // 2]
        if not positive(median) or not math.isclose(median, measured, rel_tol=1e-12):
            raise ValueError(f"{family} median does not match timing samples")
        validate_timing_stability(samples, family)
        medians.append(median)
    ratio = record.get("ratio")
    if (not positive(ratio) or not math.isclose(ratio, medians[0] / medians[1], rel_tol=1e-12)
            or float(parts[1]) != float(f"{ratio:.4f}")):
        raise ValueError("ratio does not match timing samples and stdout")


def _collect_child_evidence(args, path, name, phase, attempt, proc):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        # Python's JSON decoder accepts NaN/Infinity; the persisted artifact
        # must remain strict JSON even when a failed child wrote bad data.
        json.dumps(record, allow_nan=False)
        if record.get("cell") != name:
            raise ValueError("sidecar cell does not match")
        if not proc.returncode:
            _validate_child_record(record, proc, name)
    except RuntimeError as exc:
        # The strict JSON and cell identity checks already passed. Preserve
        # raw samples from an unstable measurement, with an explicit failure.
        record["error"] = f"unverified child evidence: {exc}"
    except (OSError, ValueError, AttributeError) as exc:
        record = {"cell": name, "error": f"missing/invalid child evidence: {exc}"}
    record.update(phase=phase, attempt=attempt, returncode=proc.returncode,
                  stderr=proc.stderr or "")
    args._evidence["cells"].append(record)
    if not proc.returncode and ("error" in record or "result" not in record):
        raise RuntimeError(f"cell {name}: {record.get('error', 'missing result evidence')}")


def _measure_one_cell(name: str, args, cutoff: float | None,
                      phase: str = "initial") -> tuple[list[str], list[str]]:
    """Run `name` in its own process, retrying while it draws a slow core.

    Returns (cmd, parts); `parts` is the validated last output line split
    on whitespace. A failed child aborts, including during core retries.
    """
    cmd = [sys.executable, str(Path(__file__)), "--one", name]
    cmd += ["--rounds", str(getattr(args, "rounds", 9)),
            "--sample-seconds", str(getattr(args, "sample_seconds", 0.02))]
    if cutoff is not None:
        cmd += ["--fast-under", f"{cutoff:.3f}"]
    for attempt in range(max(1, args.retries)):
        child_cmd = list(cmd)
        evidence_path = None
        if getattr(args, "_evidence", None) is not None:
            evidence_path = Path(args._evidence_dir) / f"{len(args._evidence['cells'])}.json"
            child_cmd += ["--evidence-cell", str(evidence_path)]
        proc = subprocess.run(child_cmd, capture_output=True, text=True,
                              cwd=str(REPO))
        if evidence_path is not None:
            _collect_child_evidence(args, evidence_path, name, phase, attempt + 1, proc)
        parts = _child_parts(proc, name)
        if parts != ["SLOW-CORE"]:
            break
    return cmd, parts


def _measure_all(names: list[str], args,
                 cutoff: float | None) -> tuple[list[tuple[str, str, float]], int, int]:
    """Measure every cell in its own process; return (losses, judged, skipped)."""
    losses: list[tuple[str, str, float]] = []
    judged = skipped = 0
    for name in names:
        _, parts = _measure_one_cell(name, args, cutoff)
        if not parts or parts[0] != "RATIO":
            skipped += 1
            if args.verbose:
                print(f"  {'skip':>8s}  {name:28s} {' '.join(parts[1:])}")
            continue
        judged += 1
        ratio, op = float(parts[1]), parts[2]
        if ratio < args.min:
            # MAKE THE RED REPRODUCE. A sweep this wide will throw the odd
            # unlucky cell, and a detector that cries wolf gets ignored -
            # which is how the real one then gets missed. A loss is only
            # reported when a second, independent process agrees; the worse
            # of the two is what gets printed.
            _, reparts = _measure_one_cell(name, args, cutoff, phase="confirmation")
            if reparts[0] != "RATIO":
                judged -= 1
                skipped += 1
                if args.verbose:
                    print(f"  {'skip':>8s}  {name:28s} confirmation {' '.join(reparts)}")
                continue
            confirmed = float(reparts[1])
            if confirmed < args.min:
                ratio = min(ratio, confirmed)
                losses.append((name, op, ratio))
            elif args.verbose:
                print(f"  {ratio:6.2f}x  {name:28s} {op}  (did not reproduce; "
                      f"second reading {confirmed:.4f})")
            if confirmed >= args.min:
                ratio = confirmed
        if args.verbose:
            # Preserve the child's precision for the cross-version parser;
            # rounding 0.9999x to 1.00x would hide a confirmed loss.
            print(f"  {ratio:8.4f}x  {name:28s} {op}")
    return losses, judged, skipped


def _report(losses: list[tuple[str, str, float]], judged: int, skipped: int, args) -> int:
    print(f"\njudged {judged} dispatching paths in their own processes, "
          f"skipped {skipped} (no canonical input, or it does not dispatch)")
    if not judged:
        print("NOT verified: no dispatching cell was measured")
        return 1
    if losses:
        print(f"PESSIMIZATION: {len(losses)} path(s) below {args.min:g}x")
        for name, op, ratio in sorted(losses, key=lambda r: r[2]):
            print(f"  !! {ratio:5.2f}x  {name}  ({op})")
        return 1
    print(f"no dispatching path is below {args.min:g}x")
    return 0


def _fingerprint():
    from lab.dyno.fingerprint import machine_fingerprint
    return machine_fingerprint()


def _foreign_load():
    from lab.dyno.load import cpu_busy_fraction
    return cpu_busy_fraction()


def _calibration_metadata():
    from pyoverdrive import calibration

    return {"saved": copy.deepcopy(calibration.load()),
            "override_configured": bool(os.environ.get("PYOVERDRIVE_CALIBRATION"))}


def _source_revision():
    # The content manifest is authoritative; HEAD is only optional context.
    # Remote evidence commonly runs from an uncommitted source zip, no Git.
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=str(REPO))
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _source_manifest(root=REPO):
    """Hash exact source bytes, independent of checkout location or Git."""
    root = Path(root)
    paths = [*root.glob("src/pyoverdrive/**/*.py"), *root.glob("lab/dyno/**/*.py")]
    paths += [root / "tools/verify_no_pessimization.py", root / "pyproject.toml"]
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(paths) if path.is_file()}
    manifest = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(manifest).hexdigest(), "files": files}


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _run_child(args):
    evidence = {"cell": args.one} if args.evidence_cell else None
    try:
        result = _measure_one(args.one, args.fast_under, evidence=evidence,
                              rounds=args.rounds, sample_seconds=args.sample_seconds)
        if evidence is not None:
            evidence["result"] = result
        print(result)
        return 0
    except BaseException as exc:
        if evidence is not None:
            evidence["error"] = repr(exc)
        raise
    finally:
        if evidence is not None:
            _write_json(args.evidence_cell, evidence)


def _run_with_evidence(args):
    if not args.json and not args.require_quiet:
        return _run_sweep(args)
    evidence = {
        "schema_version": 1, "tool": "verify_no_pessimization",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": _fingerprint(), "source_revision": _source_revision(),
        "source": _source_manifest(),
        "calibration": _calibration_metadata(),
        "selection": {key: getattr(args, key) for key in ("only", "rows", "sizes", "shapes", "values")},
        "settings": {"min": args.min, "retries": args.retries,
                     "rounds": args.rounds, "sample_seconds": args.sample_seconds,
                     "any_core": args.any_core, "require_quiet": args.require_quiet,
                     "environment": {key: os.environ.get(key) for key in (
                         "PYOVERDRIVE_THREADS", "PYOVERDRIVE_DISABLE", "OMP_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "conditions": {"quiet_threshold": 0.20, "cpu_busy_before": _foreign_load()},
        "cells": [], "completed": False,
    }
    exact = _cells_file_selection(args)
    if exact is not None:
        evidence["selection"]["cells_file"] = exact
    code = 1
    try:
        before = evidence["conditions"]["cpu_busy_before"]
        if args.require_quiet and (before is None or before > 0.20):
            evidence["error"] = f"quiet run refused: foreign CPU load is {before!r}"
            print("NOT verified: " + evidence["error"])
        else:
            with tempfile.TemporaryDirectory(prefix="pyoverdrive-evidence-") as scratch:
                args._evidence = evidence if args.json else None
                args._evidence_dir = scratch
                code = _run_sweep(args)
                evidence["completed"] = "summary" in evidence if args.json else True
    except BaseException as exc:
        evidence["error"] = repr(exc)
        print(f"NOT verified: {exc}", file=sys.stderr)
        if not isinstance(exc, Exception):
            raise
    finally:
        conditions = evidence["conditions"]
        conditions["cpu_busy_after"] = _foreign_load()
        loads = [conditions[key] for key in ("cpu_busy_before", "cpu_busy_after")]
        conditions["contended"] = any(value is not None and value > 0.20 for value in loads)
        conditions["load_known"] = all(value is not None for value in loads)
        if args.require_quiet and (conditions["contended"] or not conditions["load_known"]):
            code = 1
            print("NOT verified: quiet conditions were not met before and after the run")
        evidence["exit_code"] = code
        if args.json:
            _write_json(args.json, evidence)
    return code


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=1.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--any-core", action="store_true",
                    help="accept measurements taken on a slow core too; on a "
                         "hybrid CPU that can hide a threaded pessimization")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=9,
                    help="interleaved timing rounds per measured cell (default: 9)")
    ap.add_argument("--sample-seconds", type=float, default=0.02,
                    help="target seconds per stock timing sample; both sides use the same iteration count (default: 0.02)")
    ap.add_argument("--sizes", action="store_true",
                    help="also judge each cell at 3x, 10x, 30x and 100x its "
                         "canonical size. Every canonical input sits near the "
                         "BOTTOM of what its path accepts, so upward is the "
                         "axis nothing was sampling - and three shipped losses "
                         "were hiding up there.")
    ap.add_argument("--shapes", action="store_true",
                    help="also judge each 2-D-or-deeper cell with its aspect "
                         "ratio moved 4x and 16x in both directions at ~constant "
                         "volume. Scaling moves volume; this moves SHAPE, which "
                         "is the axis np.inner's 0.38x corner lived on.")
    ap.add_argument("--rows", action="store_true",
                    help="also judge one cell per SHIPPED CALIBRATION ROW - "
                         "every dimension, dtype and mode a path's gate table "
                         "names, at that row's own floor and cap. The sweep "
                         "otherwise judges whichever row the path's single "
                         "fixture happens to land on, and every small-matrix "
                         "linalg fixture is d=3.")
    ap.add_argument("--values", action="store_true",
                    help="also judge each cell with its INTEGER operands "
                         "redrawn at both ends of the cardinality axis - 16 "
                         "distinct values, and the dtype's whole range. The "
                         "unique and intersect families have measured "
                         "cardinality sensitivity written in their own "
                         "comments and never had a cell for it.")
    ap.add_argument("--only", action="append", default=[],
                    help="restrict the run to cells whose name contains this "
                         "(repeatable). For probing one path's window after a "
                         "red - the whole sweep is minutes, one path is "
                         "seconds. It NARROWS coverage, so a green under "
                         "--only is never a green for the suite.")
    ap.add_argument("--json", metavar="PATH", help="write fingerprinted raw timing and correctness evidence")
    ap.add_argument("--cells-file", metavar="PATH",
                    help="select exact newline-separated cell names after the normal axes/filters; reject unknown names")
    ap.add_argument("--require-quiet", action="store_true",
                    help="refuse unknown or >20%% foreign CPU load before/after the run")
    ap.add_argument("--evidence-cell", help=argparse.SUPPRESS)
    ap.add_argument("--fast-under", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--one", help=argparse.SUPPRESS)  # internal: one path, own process
    args = ap.parse_args(argv[1:])
    if args.rounds < 1 or not math.isfinite(args.sample_seconds) or args.sample_seconds <= 0:
        ap.error("rounds and sample-seconds must be positive finite values")

    if args.one:
        return _run_child(args)
    return _run_with_evidence(args)


def _run_sweep(args):
    # EVERY PATH GETS A FRESH PROCESS. Measuring 69 paths in one process
    # produced a red that reproduced four times and was still wrong:
    # pyrallel_tan reported 0.89x in the sweep and 1.15x in isolation, in
    # every configuration tried (only-tan enabled, everything enabled, pool
    # already warm). Sixty-eight other paths' allocations get to the
    # threaded ones through cache and allocator state, and no amount of
    # re-measuring inside that process escapes it. A per-path subprocess is
    # slower and is the only way the number means anything.
    pyoverdrive.enable()
    live = {
        p.name
        for lst in GEARBOX._paths.values()
        for p in (lst if isinstance(lst, list) else [lst])
        if p.enabled
    }
    names, code = _select_cells(args, live)
    evidence = getattr(args, "_evidence", None)
    if evidence is not None:
        evidence["selected_cells"] = names
    if code is not None:
        pyoverdrive.disable()
        return code

    classes = cpuclass.classify()
    # WHICH NUMPY produced a verdict is part of the verdict. batch 16 found
    # searchsorted's int64 row winning 1.9-4.0x on numpy 2.4.5 and losing
    # 0.26-0.74x on 2.5.2 - same box, same code, same day. A green with no
    # version beside it cannot be compared with the next one.
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}")
    print(cpuclass.describe(classes))
    cutoff = None if args.any_core else cpuclass.fast_cutoff(classes)
    if evidence is not None:
        evidence["cpu_classes"] = classes
        evidence["fast_core_cutoff_us"] = cutoff
    if cutoff is not None:
        print(f"measuring only on the fast class (probe <= {cutoff:.0f} us), "
              f"up to {args.retries} re-draws per path")

    losses, judged, skipped = _measure_all(names, args, cutoff)
    if evidence is not None:
        evidence["summary"] = {"losses": losses, "judged": judged, "skipped": skipped}
    return _report(losses, judged, skipped, args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
