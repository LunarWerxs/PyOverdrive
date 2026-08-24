"""Fast path: large two-operand numpy.einsum through numpy's own optimize=True.

Provenance (OPP-000018): numpy/numpy#22604 reports default np.einsum ~20x
slower than torch/tf/jax on a batched-matmul-shaped contraction
('thd,Thd->thT'), because optimize=False runs the naive C loop while every
other library routes through BLAS. Upstream largely fixed the OPTIMIZED
path (PR #23513) but deliberately kept optimize=False as the default to
protect tiny contractions (dgasmith: len-3 inner products; measured here:
optimize=True is 0.15-0.18x, i.e. 5-7x SLOWER, at that size). Dyno
reproduced 17.6-45.0x on the reported shapes
(benchmarks/results/OPP-000018/).

This path does NOT reimplement any contraction. It routes an admissible
default-arguments call to stock np.einsum(..., optimize=True): numpy's own
planner and BLAS decomposition, upstream-tested. What PyOverdrive adds is
the SIZE GATE the maintainers could not: dispatch only where measurement
says the planning overhead is repaid.

Correctness contract:
- Applies only to the subscripts-string form einsum(subs, a, b) with
  exactly two operands, both plain ndarrays of the same float64/float32
  dtype, and NO keyword arguments (out, dtype, order, casting, and an
  explicit optimize= all force stock: the user chose them).
- The subscripts must parse cleanly (alphabetic labels, one comma, label
  counts matching each operand's ndim); anything else stays on stock so
  stock also raises its own errors for malformed calls. Ellipsis forms
  (batch 9): admitted when BOTH operands carry '...' with equal ellipsis
  shapes and at least one real ellipsis dimension - '...ij,...jk->...ik'
  and its implicit-output form - always in the projected regime.
  Unequal ellipsis shapes (numpy broadcasts them), '...' in only one
  operand, and explicit outputs without '...' stay on stock: legal
  numpy, but unmeasured or subtle shape classes.
- Two regimes with separate calibrated floors (EINSUM-CAL battery,
  benchmarks/results/EINSUM-CAL/):
  * projected output (the result keeps at least one index): matmul-shaped
    and batched contractions, min(a.size, b.size) >= 10_000. Measured
    3.4-27.4x float64 from the floor up (0.72-0.79x just below it).
  * scalar output ('i,i->', 'ij,ij->' and implicit forms reducing every
    index): BLAS only pays at min size >= 1_000_000 (1.5-2.1x measured;
    0.28-0.48x at 10_000).
- Results are numerically equal, not bit-identical: optimize=True sums in
  a different order. Near-cancelling float32 outputs can differ by
  O(eps32 * L) absolutely, and the measured default path is FARTHER from
  the float64 truth than the optimized one at such elements (5.11x vs
  0.65x relative at 'thd' 300x1x300), so the difference is accumulation
  noise on both sides, not a fast-path defect; the differential battery
  uses absolute-scaled tolerance accordingly.

Chain regime (OPP-000049, numpy/numpy#11714 class): a SECOND path,
einsum_optimize_chain, covers three-or-more-operand contractions, where
optimize=False runs one fused loop over the union of every index (so
ij,jk,kl->il costs O(i*j*k*l) against the optimized route's two BLAS
matmuls; 249x measured at n=64 on the dev box). Same clean-subscripts
and same-dtype rules as the two-operand path, including the batch-9
ellipsis admission (every operand carries '...', equal ellipsis
shapes; the batch dimensions multiply into the naive volume because
the naive loop iterates them too, and the ellipsis spelling gates
against its own measured CHAIN_ELLIPSIS_VOLUME_FLOOR); extent
consistency and output-label validity are pre-checked so malformed
calls raise from stock; the gate is the naive loop volume (product of
distinct label extents) against
CHAIN_VOLUME_FLOOR, or CHAIN_SCALAR_VOLUME_FLOOR for
scalar-output chains (BLAS pays off later on full reductions, same as
the two-operand path's split). Separate kill switch:
einsum_optimize_chain.

Comparison mode: numeric (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=einsum_optimize or
pyoverdrive.disable_path("einsum_optimize").
"""

from __future__ import annotations

import math

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

# CALIBRATION (fp 8f8198d9abab, benchmarks/results/EINSUM-CAL/, 2026-08-23;
# float64 sweep at 16-18% load, float32 rerun contended 33-48% and
# consistent): floors are the measured crossovers, identical for both
# dtypes. At the projected-output floor: matmul 3.98x f64 / 4.14x f32,
# thd 3.40x f64 / 1.43x f32 (contended), bmm already 2.0-2.2x one step
# below it. At the scalar-output floor: inner 2.06x f64 / 1.86x f32,
# frob 1.53x f64 / 1.96x f32.
PROJECTED_FLOOR = 10_000
SCALAR_FLOOR = 1_000_000

_DTYPES = (np.dtype(np.float64), np.dtype(np.float32))


def _ellipsis_term(term: str, arr):
    """(labels, left-label count, ellipsis shape) for one operand term
    carrying exactly one '...' and no stray dots, or None. Requires at
    least one real ellipsis dimension: the zero-dim case is just the
    non-ellipsis spelling of the same call, already handled there."""
    i = term.find("...")
    if i < 0 or "." in term[:i] or "." in term[i + 3 :]:
        return None
    labels = term[:i] + term[i + 3 :]
    if labels and not labels.isalpha():
        return None
    ell_nd = arr.ndim - len(labels)
    if ell_nd < 1:
        return None
    return labels, i, arr.shape[i : i + ell_nd]


def _ellipsis_out(out: str, extents) -> bool:
    """Validate an explicit '...'-carrying output term against the input
    label set; anything else (including '->' with no ellipsis, where
    numpy's own broadcast-dimension rules get subtle) stays on stock."""
    j = out.find("...")
    if j < 0 or "." in out[:j] or "." in out[j + 3 :]:
        return False
    olabels = out[:j] + out[j + 3 :]
    if olabels and not olabels.isalpha():
        return False
    return len(set(olabels)) == len(olabels) and set(olabels) <= set(extents)


def _ellipsis_ok(s: str, a, b) -> bool:
    """Admit a two-operand ellipsis form ('...ij,...jk->...ik' class,
    batch 9): both operands carry '...', ellipsis shapes equal (numpy
    broadcasts unequal ones - legal, but an unmeasured shape class, so
    it stays on stock). Always the projected regime: with at least one
    real ellipsis dimension the output is never a scalar."""
    lhs, arrow, out = s.partition("->")
    ops = lhs.split(",")
    if len(ops) != 2:
        return False
    pa = _ellipsis_term(ops[0], a)
    pb = _ellipsis_term(ops[1], b)
    if pa is None or pb is None or pa[2] != pb[2]:
        return False
    extents: dict[str, int] = {}
    for (labels, left, eshape), arr in ((pa, a), (pb, b)):
        lshape = arr.shape[:left] + arr.shape[left + len(eshape) :]
        for c, n in zip(labels, lshape):
            if extents.setdefault(c, n) != n:
                return False  # mismatched extent: stock raises its own error
    if arrow and not _ellipsis_out(out, extents):
        return False
    return True


def _parse(subs: str):
    """Return True (scalar output) / False (projected) for a clean
    two-operand subscript string, or None when the call must stay on stock."""
    s = subs.replace(" ", "")
    if "." in s:
        return None  # ellipsis forms go through _ellipsis_ok instead
    lhs, arrow, out = s.partition("->")
    ops = lhs.split(",")
    if len(ops) != 2:
        return None
    labels = ops[0] + ops[1] + out
    if labels and not labels.isalpha():
        return None
    if arrow:
        return out == ""
    counts: dict[str, int] = {}
    for c in ops[0] + ops[1]:
        counts[c] = counts.get(c, 0) + 1
    return all(v > 1 for v in counts.values())


def _applicable(args: tuple, kwargs: dict) -> bool:
    if kwargs or len(args) != 3:
        return False
    subs, a, b = args
    if type(subs) is not str or type(a) is not np.ndarray or type(b) is not np.ndarray:
        return False
    if a.dtype != b.dtype or a.dtype not in _DTYPES:
        return False
    lo = a.size if a.size < b.size else b.size
    if lo < PROJECTED_FLOOR:  # cheapest refusal before any string work
        return False
    s = subs.replace(" ", "")
    if "." in s:
        return _ellipsis_ok(s, a, b)  # projected regime; lo already gated
    scalar = _parse(subs)
    if scalar is None:
        return False
    ops = s.partition("->")[0].split(",")
    if len(ops[0]) != a.ndim or len(ops[1]) != b.ndim:
        return False
    return lo >= (SCALAR_FLOOR if scalar else PROJECTED_FLOOR)


def _run(subs, *operands):
    res = GEARBOX.stock_fn("numpy.einsum")(subs, *operands, optimize=True)
    if type(res) is np.ndarray and res.ndim == 0:
        # stock einsum returns a numpy scalar for scalar-output subscripts;
        # the optimize=True route returns a 0-d array. Match stock.
        return res[()]
    return res


# --- chain regime: three or more operands (OPP-000049) -----------------
#
# With optimize=False, numpy runs a SINGLE fused C loop over the union of
# every index in the contraction, so a chain like ij,jk,kl->il costs
# O(i*j*k*l) where the optimized route pays two BLAS matmuls. The honest
# gate is therefore the naive loop volume (product of the distinct input
# labels' extents), not any one operand's size.
#
# CALIBRATION (fp 9bbe7063c555, idle box, 0-1% load, numpy 2.5.2,
# benchmarks/results/BATCH7-CAL/, ij,jk,kl->il grid): 0.78x at volume
# 20_736 (n=12), 1.92x at 65_536 (n=16), 7.40x at 331_776 (n=24),
# 20.4x at 1.05M, 178x f64 / 235x f32 at 16.8M (n=64); 4-operand cell
# 234x at 3.2M. Scalar output pays off later, exactly as it did for the
# two-operand path: the measured scalar chain cell wins 6.31x at volume
# 262_144, unmeasured below - hence its own higher floor at that cell.
CHAIN_VOLUME_FLOOR = 65_536
CHAIN_SCALAR_VOLUME_FLOOR = 262_144
# The ellipsis spelling crosses 1.3x later than the label spelling (the
# planner and kernels see the batched shape): BATCH9-CAL idle-box cells
# 0.62x at volume 20_736, then 1.29x/1.49x across runs at 65_536 - a
# straddling cell, so by the bistable-cell rule the floor moves one
# measured notch up, to the first cell clearing 1.3x in every run:
# 2.20x idle / 2.28x dev at 76_832, 3.38x at 131_072, 163x at 67.1M.
CHAIN_ELLIPSIS_VOLUME_FLOOR = 76_832


def _chain_volume_ellipsis(s: str, operands: tuple):
    """(naive-loop volume, CHAIN_ELLIPSIS_VOLUME_FLOOR) for an ellipsis
    chain where every operand carries '...' with equal ellipsis shapes
    ('...ij,...jk,...kl' class, batch 9), or None. The naive C loop
    iterates the batch dimensions too, so they multiply into the volume;
    with at least one real ellipsis dimension the output is never a
    scalar, but the ellipsis spelling carries its own, higher floor."""
    lhs, arrow, out = s.partition("->")
    ops = lhs.split(",")
    if len(ops) < 3 or len(ops) != len(operands):
        return None
    extents: dict[str, int] = {}
    ell_shape = None
    for term, arr in zip(ops, operands):
        parsed = _ellipsis_term(term, arr)
        if parsed is None:
            return None
        labels, left, eshape = parsed
        if ell_shape is None:
            ell_shape = eshape
        elif eshape != ell_shape:
            return None  # ellipsis broadcasting: unmeasured, stock
        lshape = arr.shape[:left] + arr.shape[left + len(eshape) :]
        for c, n in zip(labels, lshape):
            if extents.setdefault(c, n) != n:
                return None  # mismatched extent: stock raises its own error
    if arrow and not _ellipsis_out(out, extents):
        return None
    vol = math.prod(ell_shape)
    for n in extents.values():
        vol *= n
    return vol, CHAIN_ELLIPSIS_VOLUME_FLOOR


def _chain_volume(subs: str, operands: tuple):
    """(naive-loop volume, the floor it gates against) for a clean
    >=3-operand subscript string, or None when the call must stay on
    stock."""
    s = subs.replace(" ", "")
    if "." in s:
        return _chain_volume_ellipsis(s, operands)
    lhs, arrow, out = s.partition("->")
    ops = lhs.split(",")
    if len(ops) < 3 or len(ops) != len(operands):
        return None
    labels = lhs.replace(",", "") + out
    if not labels or not labels.isalpha():
        return None
    extents: dict[str, int] = {}
    for term, arr in zip(ops, operands):
        if len(term) != arr.ndim:
            return None
        for c, n in zip(term, arr.shape):
            if extents.setdefault(c, n) != n:
                return None  # mismatched extent: stock raises its own error
    if arrow:
        if len(set(out)) != len(out) or not set(out) <= extents.keys():
            return None  # malformed output: stock raises its own error
        scalar = out == ""
    else:
        counts: dict[str, int] = {}
        for c in lhs.replace(",", ""):
            counts[c] = counts.get(c, 0) + 1
        scalar = all(v > 1 for v in counts.values())
    vol = 1
    for n in extents.values():
        vol *= n
    return vol, (CHAIN_SCALAR_VOLUME_FLOOR if scalar else CHAIN_VOLUME_FLOOR)


def _applicable_chain(args: tuple, kwargs: dict) -> bool:
    if kwargs or len(args) < 4:
        return False
    subs, *operands = args
    if type(subs) is not str:
        return False
    first = operands[0]
    if type(first) is not np.ndarray or first.dtype not in _DTYPES:
        return False
    for o in operands[1:]:
        if type(o) is not np.ndarray or o.dtype != first.dtype:
            return False
    parsed = _chain_volume(subs, tuple(operands))
    if parsed is None:
        return False
    vol, floor = parsed
    return vol >= floor


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="einsum_optimize",
            op="numpy.einsum",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000018",
                "source": "https://github.com/numpy/numpy/issues/22604",
                "license": "routes to numpy's own optimize=True machinery; no third-party code",
                "comparison_mode": "numeric (summation order differs; absolute-scaled for float32)",
            },
        )
    )
    gearbox.register(
        FastPath(
            name="einsum_optimize_chain",
            op="numpy.einsum",
            applicable=_applicable_chain,
            run=_run,
            provenance={
                "opportunity": "OPP-000049",
                "source": "https://github.com/numpy/numpy/issues/11714",
                "license": "routes to numpy's own optimize=True machinery; no third-party code",
                "comparison_mode": "numeric (summation order differs; absolute-scaled for float32)",
            },
        )
    )
