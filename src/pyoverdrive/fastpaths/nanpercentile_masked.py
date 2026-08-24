"""Fast path: numpy.nanpercentile - the shipped nanquantile_masked route
with numpy's own q/100 scaling.

Provenance (OPP-000013, sibling surface): numpy implements nanpercentile
as nanquantile at q/100 (the same true_divide(q, 100) this route
performs), so it inherits the identical per-slice Python-loop collapse
numpy/numpy#16575 documents. The NANPERCENTILE-CAL battery measures this
path's OWN cells against stock np.nanpercentile at a subset of the
FASTNANQ-CAL grid; floors and the admission rule are shared with the
parent because the post-scaling arithmetic is the parent's, verbatim.

Correctness contract: identical to nanquantile_masked's, with q a single
Python int/float in [0, 100] instead of [0, 1]. Same refusals (out,
keepdims, overwrite_input, method, weights, q sequences, non-f64, 1-D,
axis=None), same all-NaN-slice RuntimeWarning, same numeric comparison
mode (ULP-scale: the same interpolation arithmetic as stock).

Kill switch: PYOVERDRIVE_DISABLE=nanpercentile_masked or
pyoverdrive.disable_path("nanpercentile_masked").
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import FastPath
from . import nanquantile_masked as _nq

_F64 = np.dtype(np.float64)


def _applicable(args: tuple, kwargs: dict) -> bool:
    norm = _nq._normalize(args, kwargs)
    if norm is None:
        return False
    a, q, axis = norm
    if type(a) is not np.ndarray or a.dtype != _F64:
        return False
    if a.ndim < 2 or a.size < _nq.SIZE_FLOOR:
        return False
    if isinstance(axis, bool) or not isinstance(axis, int):
        return False
    if not -a.ndim <= axis < a.ndim:
        return False
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        return False
    if not 0.0 <= q <= 100.0:
        return False
    reduced = a.shape[axis]
    return reduced <= _nq._REDUCED_LEN_CAP or a.size >= reduced * reduced


def _run(a: np.ndarray, q, axis=None) -> np.ndarray:
    return _nq._run(a, np.true_divide(q, 100), axis)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="nanpercentile_masked",
            op="numpy.nanpercentile",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000013",
                "source": "https://github.com/numpy/numpy/issues/16575",
                "license": "shared with nanquantile_masked; no third-party code",
                "comparison_mode": "numeric (float64, ULP-scale: same interpolation arithmetic as stock)",
            },
        )
    )
