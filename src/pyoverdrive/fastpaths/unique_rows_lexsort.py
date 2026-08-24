"""Fast path: numpy.unique(a, axis=0) on integer rows via lexsort.

Provenance (OPP-000040): numpy/numpy#11136 - unique(axis=0) is
needlessly slow (thread spans 2018-2026; 1.96-3x claims). The raw
void-view trick the thread favors is DEAD for a transparent layer: a
pre-battery probe showed stock returns rows in NUMERIC lexicographic
order while little-endian void memcmp does not (negative ints). The
shipped route is np.lexsort over the columns (last column as the least
significant key), a gather, and an adjacent-difference row mask -
numeric lexicographic by construction, so every battery cell was
bit-identical, negative-salted rows and counts included.

Measured (OPP-000040 battery, fp 9bbe7063c555, idle box, 0-1% load):
int64 low-cardinality 4.4-5.0x (k=2), 2.5-3.1x (k=4), 1.84x (k=8);
high-cardinality 2.46x (k=2) / 1.40x (k=4); int32 4.0x;
return_counts 4.88x. Every measured cell from n=1000 up clears the
min-win, hence the floor; k > 8 and other dtypes are unmeasured and
stay on stock.

Correctness contract:
- Applies only to unique(a, axis=0) (axis by keyword or fifth
  positional) where a is a plain 2-D int64/int32 ndarray with
  2 <= columns <= 8 and rows >= 1000, and the only other argument is
  return_counts. return_index/return_inverse are refused (index
  selection among duplicate rows follows stock's internal sort
  stability, which this route does not replicate); equal_nan/sorted
  and other dtypes stay on stock. Single-column 2-D belongs to
  unique_axis0_column (OPP-000014); 1-D belongs to unique_sort.
- Output rows ascend numerically-lexicographically, counts align:
  bit-identical to stock.

Comparison mode: bit-identical (spec section 9). Kill switch:
PYOVERDRIVE_DISABLE=unique_rows_lexsort or
pyoverdrive.disable_path("unique_rows_lexsort").

Implementation note: calls np.lexsort and elementwise compares only -
never np.unique (a patched name; the OPP-000000 recursion law).
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath

_DTYPES = frozenset((np.dtype(np.int64), np.dtype(np.int32)))
ROWS_MIN = 1_000
K_MIN, K_MAX = 2, 8


def _normalize(args: tuple, kwargs: dict):
    if not args:
        return None
    allowed = {"return_index", "return_inverse", "return_counts", "axis", "equal_nan"}
    if set(kwargs) - allowed:
        return None
    if kwargs.get("return_index", False) or kwargs.get("return_inverse", False):
        return None
    if "equal_nan" in kwargs:
        return None
    rc = kwargs.get("return_counts", False)
    axis = kwargs.get("axis", None)
    if len(args) > 1:
        # positional (ar, return_index, return_inverse, return_counts, axis)
        if len(args) > 5:
            return None
        pos = list(args[1:]) + [None] * (5 - len(args))
        if pos[0] or pos[1]:
            return None
        if pos[2] is not None:
            if "return_counts" in kwargs:
                return None
            rc = pos[2]
        if pos[3] is not None:
            if "axis" in kwargs:
                return None
            axis = pos[3]
    if axis != 0:
        return None
    if not isinstance(rc, (bool, np.bool_)):
        return None
    return args[0], bool(rc)


def _applicable(args: tuple, kwargs: dict) -> bool:
    norm = _normalize(args, kwargs)
    if norm is None:
        return False
    a, _rc = norm
    if type(a) is not np.ndarray or a.ndim != 2 or a.dtype not in _DTYPES:
        return False
    return K_MIN <= a.shape[1] <= K_MAX and a.shape[0] >= ROWS_MIN


def _run(ar, *pos_flags, return_counts=False, axis=0, **_):
    if pos_flags and len(pos_flags) >= 3 and pos_flags[2]:
        return_counts = True
    n, k = ar.shape
    order = np.lexsort(tuple(ar[:, j] for j in range(k - 1, -1, -1)))
    srt = ar[order]
    mask = np.empty(n, dtype=bool)
    mask[0] = True
    np.any(srt[1:] != srt[:-1], axis=1, out=mask[1:])
    if not return_counts:
        return srt[mask]
    idx = np.flatnonzero(mask)
    counts = np.diff(np.append(idx, n))
    return srt[mask], counts


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="unique_rows_lexsort",
            op="numpy.unique",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000040",
                "source": "https://github.com/numpy/numpy/issues/11136",
                "license": "lexsort + adjacent-difference, standard technique; no third-party code",
                "comparison_mode": "bit-identical",
            },
        )
    )
