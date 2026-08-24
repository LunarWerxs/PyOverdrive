"""OPP-000008 reproducer: threaded chunked dispatch for ufuncs.

NumPy issue numpy/numpy#8208 (2016, closed same day as out-of-scope) reported
a ~3x speedup running np.sin across 4 threads on a 1e7-element float64 array,
using a ThreadPoolExecutor and out= slices per chunk. NumPy's C ufunc loops
release the GIL for compute-bound kernels above a size threshold, so manual
chunked threading can genuinely scale even though the interpreter itself does
not run Python bytecode in parallel.

This reproducer:
  - times a plain np.sin(x) call as the baseline (per assignment; the
    original issue's baseline used out=x in place, which this does not);
  - times a threaded_chunks candidate that splits x into N contiguous
    chunks and dispatches each to np.sin(chunk, out=out_chunk) on a
    pre-created ThreadPoolExecutor (created once at module setup, never
    inside a timed call), for thread counts [4, 8, 16];
  - repeats the same threaded_chunks harness on np.add at the largest size
    as an explicit memory-bound negative control: np.add is bandwidth
    limited, not compute limited, so it should show little or no scaling
    even though the mechanism (GIL release, chunking) is identical.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SEED = 20161024

if SMOKE:
    SIZES = [10_000]
    THREAD_COUNTS = [2]
    SAMPLES_SMALL = 3
    SAMPLES_LARGE = 3
else:
    SIZES = [100_000, 1_000_000, 10_000_000]
    THREAD_COUNTS = [4, 8, 16]
    SAMPLES_SMALL = 11  # sizes below 1e7, single call well under 100ms
    SAMPLES_LARGE = 5  # n=1e7: single call can exceed 100ms

# Executors are created ONCE here, at module setup, and reused across every
# case and every candidate below; none are created inside a timed lambda.
EXECUTORS = {t: ThreadPoolExecutor(max_workers=t) for t in THREAD_COUNTS}


def threaded_chunks(op, arrays, out, threads):
    """Split `arrays` and `out` into `threads` contiguous chunks and run
    `op(*chunk_args, out=chunk_out)` for each chunk on the shared executor."""
    n = out.shape[0]
    bounds = np.linspace(0, n, threads + 1).astype(np.int64)
    ex = EXECUTORS[threads]
    futures = [
        ex.submit(op, *(a[bounds[i]:bounds[i + 1]] for a in arrays), out=out[bounds[i]:bounds[i + 1]])
        for i in range(threads)
    ]
    for f in futures:
        f.result()
    return out


suite = BenchSuite("OPP-000008", "threaded chunked ufunc dispatch: np.sin vs np.add contrast")

rng = np.random.default_rng(SEED)

for n in SIZES:
    x = np.linspace(0.0, 2 * np.pi, n)
    out_sin = np.empty_like(x)
    samples = SAMPLES_LARGE if n >= 10_000_000 else SAMPLES_SMALL

    suite.measure(
        case=f"sin_float64_n{n}",
        params={"dtype": "float64", "n": n, "op": "sin"},
        baseline=("numpy.sin", lambda x=x: np.sin(x)),
        candidates={
            f"threaded_chunks_{t}t": (
                lambda t=t, x=x, out=out_sin: threaded_chunks(np.sin, (x,), out, t)
            )
            for t in THREAD_COUNTS
        },
        check=np.allclose,
        samples=samples,
    )

# Memory-bound contrast case: np.add at the largest swept size only. Expected
# finding is NO scaling here, unlike np.sin above, since addition is
# bandwidth limited rather than compute limited.
n_contrast = SIZES[-1]
a = rng.random(n_contrast)
b = rng.random(n_contrast)
out_add = np.empty_like(a)
samples_contrast = SAMPLES_LARGE if n_contrast >= 10_000_000 else SAMPLES_SMALL

suite.measure(
    case=f"add_float64_n{n_contrast}",
    params={"dtype": "float64", "n": n_contrast, "op": "add"},
    baseline=("numpy.add", lambda a=a, b=b: np.add(a, b)),
    candidates={
        f"threaded_chunks_{t}t": (
            lambda t=t, a=a, b=b, out=out_add: threaded_chunks(np.add, (a, b), out, t)
        )
        for t in THREAD_COUNTS
    },
    check=np.allclose,
    samples=samples_contrast,
)

if not SMOKE:
    suite.save()
