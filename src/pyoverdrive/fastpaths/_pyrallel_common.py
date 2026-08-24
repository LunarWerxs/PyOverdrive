"""Shared pieces of the PyRallel fast-path families (unary and binary).

THREAD_SCHEDULE is keyed on BYTES, not elements: float32 kernels run ~2x
faster per element than float64 on the calibrated machine, so a float32
array needs ~2x the elements to carry the same fixed thread overhead; one
byte-keyed schedule fits both dtypes where an element-keyed one could not.
For binary ops the bytes counted are the OUTPUT's (one operand's worth), the
same yardstick the unary battery used. Derived by
lab/cli/calibrate_pyrallel.py from the PYRALLEL-CAL evidence.
"""

from __future__ import annotations

import numpy as np

from ..parallel import pyrallel

THREAD_SCHEDULE: tuple[tuple[int, int], ...] = (
    (24 * 1024 * 1024, 16),  # >= 24 MiB: 3M float64 / 6M float32
    (8 * 1024 * 1024, 8),    # >= 8 MiB: 1M float64 / 2M float32
    (768 * 1024, 4),         # >= 768 KiB: ~100k float64 / ~200k float32
)


def threads_for(nbytes: int) -> int:
    """Thread count for an output of ``nbytes``; 1 below the schedule floor
    (unreachable through dispatch, since every calibrated threshold sits at
    or above the floor; kept defensive for direct callers)."""
    for min_bytes, t in THREAD_SCHEDULE:
        if nbytes >= min_bytes:
            return t
    return 1


def out_ok(o, shape: tuple, dtype: np.dtype) -> bool:
    """``out=`` is accepted only as an exact-shape, exact-dtype, writeable,
    C-contiguous plain ndarray: no cast, no broadcast, no strided write."""
    return (
        type(o) is np.ndarray
        and o.shape == shape
        and o.dtype == dtype
        and o.flags.c_contiguous
        and o.flags.writeable
    )


def core_ready() -> bool:
    """PyRallel can dispatch at all: pool width >= 2 and the caller's FP
    error mode can be mirrored into workers (costs ~1 us; call it last)."""
    return pyrallel.available() and pyrallel.errstate_parallel_safe()
