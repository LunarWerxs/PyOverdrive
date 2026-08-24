"""relayout_blocked calibration: tile size x threads x dtype x shape.

Extends the OPP-000004 baseline surface (which measured one candidate with a
per-call executor: 1.1-2.1x at 4096x4096) with the SHIPPED mechanism on the
persistent PyRallel pool, sweeping the two knobs that matter for a blocked
transpose copy: tile edge (64/128/256 elements) and thread count (4/8/16),
over float64 / float32 / int64 and square plus tall/wide shapes.

Check is bit-identity (it is a copy). Result JSON:
benchmarks/results/RELAYOUT-CAL/. Feeds the tile/threads/threshold literals
in src/pyoverdrive/fastpaths/relayout_blocked.py.
Run: .venv/Scripts/python benchmarks/micro/bench_relayout_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite
from pyoverdrive.parallel.relayout import blocked_transpose_copy

SMOKE = "--smoke" in sys.argv

if SMOKE:
    SHAPES = [(256, 256)]
    TILES = [128]
    THREADS = [2]
    DTYPES = [np.float64]
    SAMPLES = 3
else:
    SHAPES = [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (8192, 1024), (1024, 8192)]
    TILES = [64, 128, 256]
    THREADS = [4, 8, 16]
    DTYPES = [np.float64, np.float32, np.int64]
    SAMPLES = 7


def bit_identical(c, b):
    return c.dtype == b.dtype and c.shape == b.shape and c.flags.c_contiguous and np.array_equal(c, b)


suite = BenchSuite("RELAYOUT-CAL", "blocked threaded transpose copy vs np.ascontiguousarray(a.T)")
rng = np.random.default_rng(21655)

for dtype in DTYPES:
    dt = np.dtype(dtype).name
    for shape in SHAPES:
        n, m = shape
        if np.dtype(dtype).kind == "f":
            base = rng.standard_normal((m, n)).astype(dtype, order="C")
        else:
            base = rng.integers(-1000, 1000, size=(m, n), dtype=dtype)
        x = base.T  # F-contiguous (n, m) view: the shape the fast path accepts
        assert x.flags.f_contiguous and not x.flags.c_contiguous
        samples = SAMPLES if n * m <= 2048 * 2048 else max(3, SAMPLES - 3)
        suite.measure(
            case=f"{dt}_{n}x{m}",
            params={"dtype": dt, "n": n, "m": m, "elements": n * m},
            baseline=("numpy.ascontiguousarray", lambda x=x: np.ascontiguousarray(x)),
            candidates={
                f"blocked_t{tile}_{t}thr": (
                    lambda x=x, tile=tile, t=t: blocked_transpose_copy(x, tile=tile, threads=t)
                )
                for tile in TILES
                for t in THREADS
            },
            check=bit_identical,
            samples=samples,
        )

if not SMOKE:
    suite.save()
