"""OPP-000004: cache-blocked plus threaded relayout (order conversion) baseline.

Source: https://github.com/numpy/numpy/issues/21655

Fingerprints the cost of numpy's stock relayout path,
``np.ascontiguousarray(a.T)`` (transpose a C-order 2-D array and materialize
the result in C order), across square sizes and both common float dtypes.
This is baseline-focused: the point is a fingerprinted cost surface for a
future native tiled kernel, not a claim about a specific candidate.

One candidate is included because the issue thread names a concrete
pure-Python alternative: a cache-blocked tiling of the same numpy slice
assignment, optionally spread across threads via
``concurrent.futures.ThreadPoolExecutor`` (the reporter's own
``fast_relayout_array2d``). It is adapted here to the transpose-copy shape
measured by the baseline. It is not a native kernel and is not expected to
reliably beat numpy at every size; it exists to show whether the issue's own
Python-level approach holds up outside its original benchmark harness.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 0x000004
SMOKE = "--smoke" in sys.argv

SIZES = [128] if SMOKE else [512, 1024, 2048, 4096]
DTYPES = ["float32", "float64"]

TILE = 128
MAX_WORKERS = 4


def tiled_transpose_copy(a: np.ndarray, tile: int = TILE, max_workers: int = 1) -> np.ndarray:
    """Cache-blocked, optionally threaded transpose-and-materialize.

    Adapts the issue's block-tiling idea (numpy slice assignment split into
    L1/L2-sized blocks, optionally parallel) to producing a C-order copy of
    a.T, matching the semantics the baseline measures.
    """
    n, m = a.shape
    result = np.empty((m, n), dtype=a.dtype, order="C")

    def process(block, a=a, result=result):
        i0, j0 = block
        i1 = min(i0 + tile, n)
        j1 = min(j0 + tile, m)
        result[j0:j1, i0:i1] = a[i0:i1, j0:j1].T

    blocks = [(i, j) for i in range(0, n, tile) for j in range(0, m, tile)]
    if max_workers <= 1:
        for b in blocks:
            process(b)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(process, blocks))
    return result


def main() -> None:
    suite = BenchSuite("OPP-000004", "relayout: ascontiguousarray(a.T) baseline surface")
    rng = np.random.default_rng(SEED)

    for dtype in DTYPES:
        for n in SIZES:
            a = rng.standard_normal((n, n)).astype(dtype, order="C")
            samples = 3 if SMOKE else (5 if n >= 2048 else 11)

            suite.measure(
                case=f"{dtype}_n{n}",
                params={"dtype": dtype, "n": n},
                baseline=("numpy.ascontiguousarray(a.T)", lambda a=a: np.ascontiguousarray(a.T)),
                candidates={
                    "cache_blocked_threaded": (
                        lambda a=a: tiled_transpose_copy(a, tile=TILE, max_workers=MAX_WORKERS)
                    ),
                },
                check=np.array_equal,
                samples=samples,
            )

    if not SMOKE:
        suite.save()


if __name__ == "__main__":
    main()
