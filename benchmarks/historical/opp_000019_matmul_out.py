"""OPP-000019: np.matmul(out=) cold-page contention with threaded OpenBLAS.

numpy/numpy#18669 (bmerry, 2021-03-23, still open, no linked PR) claims
np.matmul(a, b, out=c) is much slower than np.dot(a, b, out=c) when `c` is a
freshly allocated, never-touched output buffer, because OpenBLAS's worker
threads all page-fault into the cold buffer concurrently and contend on
kernel locks; np.dot avoids this only as a side effect of unconditionally
memset-ing its output before the GEMM call. This reproducer measures:

  - baseline: np.matmul(a[i], b[i], out=c[i]) looped per batch index over a
    freshly allocated `c`, matching the issue's own "matmul_loop" mode
    (report step 1, baseline call).
  - candidate "prefaulted": the identical matmul loop, but `c` has every one
    of its memory pages touched once (a single byte written per page, per
    seberg's own suggestion in the thread to avoid a full memset) right
    before the loop starts.
  - candidate "dot": the same loop with np.dot(a[i], b[i], out=c[i]) instead
    of np.matmul, as the report's "reference point" (report step 1).

Deviation from the report's exact sizing (documented per the task instructions,
since the literal numbers are infeasible under this repo's time/memory bounds):
the reporter's own repro is a batch of 1000 matmuls of shape 1024x32 @ 32x1024
complex64, i.e. a single output buffer of 1000 x 1024 x 1024 x 8 bytes ~= 8 GiB.
That is too large to allocate repeatedly (this reproducer allocates a FRESH
`c` on every single timed call -- see below for why) and far too slow to fit
in a bounded battery. This reproducer keeps the reporter's exact per-matrix
shape (1024x32 @ 32x1024) but shrinks the batch to 2 (for the shape actually
swept across both thread settings) and 8 (for a larger-batch-only tier), and
adds a "small" tier (128x32 @ 32x128, batch 4) below the reporter's shape, per
report step 1's "at least one smaller and one larger batch/matrix size."
Sizing was chosen conservatively from FLOP-rate estimates, not measured on a
reference machine, precisely because the non-smoke battery must stay well
under the ~90s budget on a fast machine even in the worst case.

Why every candidate allocates its OWN fresh `out` buffer inside the timed
callable, instead of the harness measuring one shared preallocated buffer:
Dyno's BenchSuite times a callable across many warmup + sampled calls (see
lab/dyno/__init__.py's _time_variant). If `c` were allocated once outside the
lambda and reused across those calls, only the very first call in the whole
run would ever see a genuinely cold, unfaulted buffer -- every later call
(warmup or timed) would reuse already-faulted pages, and the effect this
issue is about would be invisible after warmup. So each timed call allocates
its own `c` via np.empty(...) fresh, does the batch loop, and lets it be
garbage-collected before the next call. Buffer sizes here (>= 128 KiB) are
kept above numpy's small-allocation cache and glibc's mmap threshold, so a
freed buffer is actually unmapped and the next np.empty(...) genuinely gets
fresh, zero-fill-on-demand pages from the OS rather than a recycled, already-
resident chunk -- this is what makes fn() repeatable across warmup/samples
while still reproducing "freshly allocated, not yet page-faulted" on every
call. Only the setup (rng, operand arrays a/b) happens once and is never
regenerated per call; the out buffer is the only thing deliberately
reallocated on every invocation, and that reallocation happens inside the
timed region by design, not as accidental setup leakage.

check: np.array_equal (bit-identity), not allclose. Justification: with
beta=0, C is write-only in the underlying GEMM call (BLAS docs, quoted by
seberg in the thread), so prefaulting or not prefaulting `c` cannot change a
single output bit, only when its pages get faulted in -- the "prefaulted"
candidate makes the exact same np.matmul call as the baseline. np.dot and
np.matmul both dispatch 2-D, non-broadcast, contiguous matrix products to the
same underlying cblas_?gemm routine with identical alpha/beta/operand
layout (bmerry's own comment-out-the-memset experiment showed dot and matmul
converge to the same speed once that's controlled for, implying the same
GEMM path underneath); there is no operand reordering between them, so exact
equality is the honest check here rather than a loosened tolerance.

Per report step 3, the thread-count dependence claim (seberg: the gap
shrank, but did not vanish, under OMP_NUM_THREADS=1) is tested by wrapping
one shape tier in threadpoolctl.threadpool_limits(limits=1, user_api="blas")
if threadpoolctl is importable; if not, that dimension is skipped with a
printed note (per the task's optional-dependency instruction) and only the
default thread count is measured. The 1-thread sweep is applied only to the
"reporter" tier, not to all three, to keep the full battery bounded -- a
single-thread run of the larger batch tiers would be several times slower
than their default-thread run and risks blowing the ~90s budget for a
dimension the report treats as secondary evidence, not the primary claim.

This script measures stock numpy only; it never imports pyoverdrive. The
"prefaulted" candidate calls public np.matmul directly because prefaulting
is a caller-side technique applied to a stock, unpatched matmul call -- it is
not a reimplementation of matmul, so there is no private/internal route to
substitute it for.
"""

import sys
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

try:
    import threadpoolctl

    HAVE_THREADPOOLCTL = True
except ImportError:
    HAVE_THREADPOOLCTL = False
    print(
        "[opp-000019] threadpoolctl not installed; skipping the 1-thread BLAS "
        "sweep (report step 3). Only the default thread count will be measured."
    )

SEED = 18669
SMOKE = "--smoke" in sys.argv
PAGE = 4096


def make_operands(rng, batch, m, k, n, dtype):
    """Seeded, C-contiguous random operands; generated once per case, never
    inside a timed callable."""
    if np.issubdtype(dtype, np.complexfloating):
        a = (rng.standard_normal((batch, m, k)) + 1j * rng.standard_normal((batch, m, k))).astype(dtype)
        b = (rng.standard_normal((batch, k, n)) + 1j * rng.standard_normal((batch, k, n))).astype(dtype)
    else:
        a = rng.standard_normal((batch, m, k)).astype(dtype)
        b = rng.standard_normal((batch, k, n)).astype(dtype)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def prefault(out):
    """Touch one byte per memory page of `out` so every page's fault is
    resolved now, not during the timed matmul loop (seberg's own suggestion:
    a per-page touch, not a full memset)."""
    flat = out.reshape(-1).view(np.uint8)
    if flat.size == 0:
        return
    flat[::PAGE] = 0
    flat[-1] = 0  # make sure a trailing partial page is touched too


def matmul_loop(a, b, batch, out_shape, dtype):
    out = np.empty(out_shape, dtype=dtype)
    for i in range(batch):
        np.matmul(a[i], b[i], out=out[i])
    return out


def matmul_loop_prefaulted(a, b, batch, out_shape, dtype):
    out = np.empty(out_shape, dtype=dtype)
    prefault(out)
    for i in range(batch):
        np.matmul(a[i], b[i], out=out[i])
    return out


def dot_loop(a, b, batch, out_shape, dtype):
    out = np.empty(out_shape, dtype=dtype)
    for i in range(batch):
        np.dot(a[i], b[i], out=out[i])
    return out


# (label, batch, M, K, N, samples). K matches the reporter's own shape (32)
# throughout, since only batch/M/N vary the buffer size and FLOP count here.
if SMOKE:
    SHAPE_TIERS = [("smoke", 2, 64, 16, 64, 3)]
    DTYPES = [np.complex64, np.float64]
    THREAD_LABELS_BY_TIER = {"smoke": ["default"]}
else:
    SHAPE_TIERS = [
        ("small", 4, 128, 32, 128, 9),
        ("reporter", 2, 1024, 32, 1024, 5),
        ("large_batch", 8, 1024, 32, 1024, 5),
    ]
    DTYPES = [np.complex64, np.float64]
    THREAD_LABELS_BY_TIER = {
        "small": ["default"],
        # Only the reporter's own shape gets the thread-count sweep (report
        # step 3); the batch-heavier tier stays default-only to bound total
        # wall time -- see module docstring.
        "reporter": ["default"] + (["threads1"] if HAVE_THREADPOOLCTL else []),
        "large_batch": ["default"],
    }

suite = BenchSuite("OPP-000019", "np.matmul(out=) cold-page contention vs prefaulted vs np.dot")

rng = np.random.default_rng(SEED)

for tier_label, batch, m, k, n, samples in SHAPE_TIERS:
    out_shape = (batch, m, n)
    for dtype in DTYPES:
        dtype_name = np.dtype(dtype).name
        a, b = make_operands(rng, batch, m, k, n, dtype)
        for thread_label in THREAD_LABELS_BY_TIER[tier_label]:
            ctx = (
                threadpoolctl.threadpool_limits(limits=1, user_api="blas")
                if thread_label == "threads1"
                else nullcontext()
            )
            with ctx:
                suite.measure(
                    case=f"{dtype_name}_{tier_label}_{thread_label}",
                    params={
                        "dtype": dtype_name,
                        "batch": batch,
                        "M": m,
                        "K": k,
                        "N": n,
                        "threads": thread_label,
                    },
                    baseline=(
                        "numpy.matmul_loop(out=cold)",
                        lambda a=a, b=b, batch=batch, out_shape=out_shape, dtype=dtype: matmul_loop(
                            a, b, batch, out_shape, dtype
                        ),
                    ),
                    candidates={
                        "prefaulted": lambda a=a, b=b, batch=batch, out_shape=out_shape, dtype=dtype: matmul_loop_prefaulted(
                            a, b, batch, out_shape, dtype
                        ),
                        "dot": lambda a=a, b=b, batch=batch, out_shape=out_shape, dtype=dtype: dot_loop(
                            a, b, batch, out_shape, dtype
                        ),
                    },
                    check=np.array_equal,
                    samples=samples,
                )

if not SMOKE:
    suite.save()
