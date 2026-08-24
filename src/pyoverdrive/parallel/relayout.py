"""Cache-blocked, threaded 2-D relayout (transpose-and-materialize) on the
PyRallel pool.

OPP-000004 (numpy/numpy#21655): materializing a C-order copy of a
transposed 2-D array (``np.ascontiguousarray(a.T)``) walks one operand with
a large stride, so stock's single-threaded strided copy is cache-hostile on
big matrices. Tiling the copy into blocks that fit L1/L2 and spreading the
blocks across threads is the issue reporter's own fix; adapted here to the
persistent pool (the reproducer's per-call executor still measured 2.1x at
4096x4096 float32; the pool removes that startup cost).

Contract: ``x`` is a plain 2-D F-contiguous (not C-contiguous) ndarray;
returns a fresh C-contiguous array equal to ``x`` element for element
(a copy is bit-identical by definition). Each block is a NumPy slice
assignment, i.e. stock's own strided copy kernel on a cache-sized window;
this module adds no arithmetic.
"""

from __future__ import annotations

import numpy as np

from .pyrallel import _pool, _wait_all, max_threads


def _copy_blocks(src_t: np.ndarray, dst: np.ndarray, blocks: list[tuple[int, int]], tile: int) -> None:
    n, m = dst.shape
    for i0, j0 in blocks:
        i1 = min(i0 + tile, n)
        j1 = min(j0 + tile, m)
        # dst[i, j] = x[i, j] = src_t[j, i]; src_t is the C-contiguous view
        dst[i0:i1, j0:j1] = src_t[j0:j1, i0:i1].T


def blocked_transpose_copy(x: np.ndarray, tile: int = 128, threads: int = 8) -> np.ndarray:
    """C-order copy of the F-contiguous 2-D ``x`` via tiled, threaded block copies."""
    n, m = x.shape
    src_t = x.T  # C-contiguous view, shape (m, n)
    dst = np.empty((n, m), dtype=x.dtype, order="C")
    blocks = [(i, j) for i in range(0, n, tile) for j in range(0, m, tile)]
    threads = min(threads, max_threads(), len(blocks))
    if threads < 2:
        _copy_blocks(src_t, dst, blocks, tile)
        return dst
    # contiguous runs of blocks per task: each task walks a band of rows, so
    # its destination writes stay local and the per-task submit cost is paid
    # `threads` times, not once per block
    per = -(-len(blocks) // threads)
    ex = _pool()
    futures = [
        ex.submit(_copy_blocks, src_t, dst, blocks[k : k + per], tile)
        for k in range(0, len(blocks), per)
    ]
    _wait_all(futures)
    for f in futures:
        f.result()
    return dst
