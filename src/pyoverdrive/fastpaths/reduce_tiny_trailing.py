"""Fast paths: numpy.mean / numpy.sum reductions that keep only a tiny
trailing axis, rerouted to one full stock reduction per kept column.

Provenance (OPP-000026): numpy/numpy#8480 - np.ufunc.reduce always
prefers memory-order iteration, which wins when the kept trailing axis is
large but loses badly when it is tiny (the inner loop shrinks to k
elements and per-iteration overhead dominates; seberg's analysis in the
thread). The reporter's own fix - one full reduction per kept slice - is
what ships here: reshape(-1, k) (a view for C-order input) and one stock
call per column, each an axis=None reduction this predicate refuses by
construction, so the route cannot recurse.

Measured regime (MEANSUM-CAL + OPP-000026 batteries, fp 9bbe7063c555,
idle box, 0-1% load, numpy 2.5.2; complete op x dtype x k grid at the
floor):
- wins at k in {2,3,4,5} from rows >= 10_000 for float64 and float32,
  mean and sum alike: worst measured cell 2.20x (mean f64 k=5 at 500k
  rows), best 14.4x (sum f32 k=2 at 100k rows); the 3-D headline
  (1000, 1000, 3) mean is 3.92x.
- regime edges, all measured: k=1 is a wash (0.97x); k=6 thins to 1.44x
  and k=7 to ~1.0x (excluded); rows=1000 straddles the min-win at k=3/4
  (1.25x/0.99x), hence the uniform 10_000 floor even though k=2 still
  wins 1.69x there; F-order input loses (0.98x, and 0.07x for a copying
  reshape), hence the C-contiguity requirement.

Correctness contract:
- Applies only to mean(a, axis)/sum(a, axis) where a is a plain
  C-contiguous float64/float32 ndarray with ndim >= 2, the axis argument
  reduces exactly all axes except the last (ndim=2: axis 0/-2 or a
  tuple naming it; ndim>=3: a tuple covering every leading axis),
  a.shape[-1] in [2, 5], rows >= 10_000, and no other arguments.
  dtype/out/keepdims/where/initial, other dtypes, other axes, integer
  arrays, and subclasses all stay on stock.
- The reroute changes summation order versus stock's traversal, so the
  result is numerically equal, not bit-identical: both routes are
  pairwise summations of the same elements, and every battery cell
  passed allclose at rtol 1e-9 (float64) / 1e-3 (float32) with the
  headline 3e6-element reduction included. NaN/inf propagation and the
  result dtype (input dtype, per numpy's float rules) are unchanged.

Comparison mode: numeric (spec section 9). Kill switches:
PYOVERDRIVE_DISABLE=mean_tiny_trailing / sum_tiny_trailing, or
pyoverdrive.disable_path(...).
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath

_DTYPES = frozenset((np.dtype(np.float64), np.dtype(np.float32)))
K_MIN, K_MAX = 2, 5
ROWS_MIN = 10_000


def _reduces_all_but_last(axis, ndim: int) -> bool:
    leading = frozenset(range(ndim - 1))
    if isinstance(axis, (tuple, list)):
        seen = set()
        for ax in axis:
            if isinstance(ax, bool) or not isinstance(ax, (int, np.integer)):
                return False
            ax = int(ax)
            if not -ndim <= ax < ndim:
                return False
            seen.add(ax % ndim)
        return seen == leading
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        return False
    ax = int(axis)
    return ndim == 2 and -ndim <= ax < ndim and ax % ndim == 0


def _applicable(args: tuple, kwargs: dict) -> bool:
    if not 1 <= len(args) <= 2:
        return False
    if set(kwargs) - {"axis"}:
        return False
    if len(args) == 2 and "axis" in kwargs:
        return False  # duplicate axis: stock raises TypeError
    axis = args[1] if len(args) == 2 else kwargs.get("axis", None)
    a = args[0]
    if type(a) is not np.ndarray or a.dtype not in _DTYPES:
        return False
    if a.ndim < 2 or not a.flags.c_contiguous:
        return False
    k = a.shape[-1]
    if not K_MIN <= k <= K_MAX:
        return False
    if a.size // k < ROWS_MIN:
        return False
    return _reduces_all_but_last(axis, a.ndim)


def _make_run(op: str):
    def _run(a, axis=None):
        stock = GEARBOX.stock_fn(op)  # full reductions; never the patched name
        k = a.shape[-1]
        flat = a.reshape(-1, k)  # a view: the predicate required C-order
        out = np.empty(k, dtype=a.dtype)
        for c in range(k):
            out[c] = stock(flat[:, c])
        return out

    return _run


def register(gearbox) -> None:
    provenance = {
        "opportunity": "OPP-000026",
        "source": "https://github.com/numpy/numpy/issues/8480",
        "license": "reporter's own per-slice route from the public issue body",
        "comparison_mode": "numeric",
    }
    for name, op in (
        ("mean_tiny_trailing", "numpy.mean"),
        ("sum_tiny_trailing", "numpy.sum"),
    ):
        gearbox.register(
            FastPath(
                name=name,
                op=op,
                applicable=_applicable,
                run=_make_run(op),
                provenance=dict(provenance),
            )
        )
