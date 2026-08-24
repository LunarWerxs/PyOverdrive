"""Fast path: numpy.roll on small 1-D arrays via concatenate of two slices.

Provenance (OPP-000032): numpy/numpy#10848 - np.roll is implemented in
Python and carries ~3-4 us of fixed argument-normalization and
index-machinery overhead per call, which dominates on small arrays;
the reporter's own candidate np.concatenate([d[-s:], d[:-s]]) does the
identical data movement without it. seberg confirmed the mechanism
("the overhead is only the python call overhead") and closed the issue;
the overhead is still there on numpy 2.5.

Measured (OPP-000032 + BATCH4-CAL batteries, fp 9bbe7063c555, idle box,
0% load): 5.3-5.9x at n in [8, 99], 4.6-5.2x at 1000, 2.4-4.7x at
10_000, across int64/float64/int32/float32/bool alike (the saving is
fixed machinery, not element work). At 100_000 the margin is gone
(bimodal 0.13-1.55x) and at 1e6 it is a wash, hence the size cap.
shift=0 routes to a plain copy: 11.3x at n=99 down to 2.9x at 10_000.

Correctness contract:
- Applies only to roll(a, shift) with axis absent or None, where a is a
  plain 1-D ndarray of a measured dtype (int64/float64/int32/float32/
  bool), 1 <= a.size <= SIZE_CAP, and shift is a Python/numpy integer
  (bool excluded). Multi-axis rolls, tuple shifts, n-D input, axis
  given, size 0, and other dtypes stay on stock.
- shift is normalized modulo n exactly as stock does (negative and
  oversized shifts included); shift % n == 0 returns a copy (np.roll
  ALWAYS returns a copy - never the input; see OPP-000031 for why the
  aliasing shortcut is forbidden). Output is bit-identical, dtype and
  contiguity included.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=roll_concat_1d or
pyoverdrive.disable_path("roll_concat_1d").
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath

_DTYPES = frozenset(
    np.dtype(t) for t in (np.int64, np.float64, np.int32, np.float32, np.bool_)
)
SIZE_CAP = 10_000


def _applicable(args: tuple, kwargs: dict) -> bool:
    if not 1 <= len(args) <= 2:
        return False
    if set(kwargs) - {"shift", "axis"}:
        return False
    if kwargs.get("axis") is not None:
        return False
    if len(args) == 2:
        if "shift" in kwargs:
            return False  # duplicate: stock raises TypeError
        shift = args[1]
    elif "shift" in kwargs:
        shift = kwargs["shift"]
    else:
        return False  # stock raises: shift is required
    if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)):
        return False
    a = args[0]
    if type(a) is not np.ndarray or a.ndim != 1 or a.dtype not in _DTYPES:
        return False
    return 1 <= a.size <= SIZE_CAP


def _run(a, shift, axis=None):
    s = int(shift) % a.size
    if s == 0:
        return a.copy()
    return np.concatenate((a[-s:], a[:-s]))


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="roll_concat_1d",
            op="numpy.roll",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000032",
                "source": "https://github.com/numpy/numpy/issues/10848",
                "license": "reporter's own pure-numpy snippet from the public issue body",
                "comparison_mode": "bit-identical",
            },
        )
    )
