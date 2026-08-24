"""Batch-10 calibration battery: qr_small_batch (OPP-000053).

Settles the constants the batch-10 module leaves provisional (pending
this idle-box run, fp 9bbe7063c555):

- BATCH_MIN: the dev-box probe put the 1.3x crossing below batch 300
  for both d (2x2 2.54x, 3x3 1.56x at 300); the n=100 cells here
  locate the idle-box crossing so the floor lands one measured notch
  above it under the two-machine law.
- CHUNK: candidates drive the module's own _qr2_chunk/_qr3_chunk at
  1024 vs 4096 across the grid, mirroring _run exactly. The cholesky
  precedent (BATCH9-CAL) found a narrow chunk-4096 resonance around
  batch 10_000 on both machines, so that neighborhood is gridded.
- mode='r' cells prove the Q-assembly saving is real on the idle box.

Result JSON: benchmarks/results/BATCH10-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_batch10_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from lab.dyno import BenchSuite

from pyoverdrive.fastpaths.qr_small_batch import (
    BATCH_MIN,
    CHUNK,
    _qr2_chunk,
    _qr3_chunk,
)

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite(
    "BATCH10-CAL",
    "qr_small_batch: unrolled Householder 2x2/3x3 vs per-matrix LAPACK",
)
rng = np.random.default_rng(20260825)


def qr_close(rtol):
    def _chk(c, b):
        cq, cr = (c if isinstance(c, tuple) else (None, c))
        bq, br = (b if isinstance(b, tuple) else (None, b))
        scale = max(1.0, float(np.abs(br).max()))
        if not bool(np.allclose(cr, br, rtol=rtol, atol=rtol * scale)):
            return False
        if bq is None:
            return cq is None
        return bool(np.allclose(cq, bq, rtol=rtol, atol=rtol))

    return _chk


def qr_candidate(a, chunk, want_q):
    # mirrors the module's _run (same allocations, same chunk loop, same
    # band-tripper split-and-recombine), with the chunk size as the
    # calibration knob. Random Gaussian stacks trip the QR_RTOL band
    # with probability ~1e-5 per matrix, so the large-n cells here
    # genuinely exercise the split arm - which is the honest shipped
    # cost, a handful of per-matrix stock calls scattered back.
    d = a.shape[-1]
    factor = _qr2_chunk if d == 2 else _qr3_chunk
    r = np.empty(a.shape, dtype=a.dtype)
    q = np.empty(a.shape, dtype=a.dtype) if want_q else None
    src = a.reshape(-1, d, d)
    rdst = r.reshape(-1, d, d)
    qdst = q.reshape(-1, d, d) if want_q else None
    n = src.shape[0]
    bad_idx = []
    for s in range(0, n, chunk):
        bad = factor(
            src[s : s + chunk],
            None if qdst is None else qdst[s : s + chunk],
            rdst[s : s + chunk],
        )
        if bad is not None:
            bad_idx.append(s + np.flatnonzero(bad))
    if bad_idx:
        idx = np.concatenate(bad_idx)
        sub = src[idx]
        if want_q:
            res = np.linalg.qr(sub)
            qdst[idx] = res.Q
            rdst[idx] = res.R
        else:
            rdst[idx] = np.linalg.qr(sub, mode="r")
    return (q, r) if want_q else r


def stock_call(a, want_q):
    if want_q:
        res = np.linalg.qr(a)
        return (res.Q, res.R)
    return np.linalg.qr(a, mode="r")


QR_NS = (
    [1_000, 10_000]
    if SMOKE
    else [100, 300, 1_000, 3_000, 10_000, 16_384, 30_000, 100_000, 1_000_000]
)

for d in (2, 3):
    for n in QR_NS:
        a = np.ascontiguousarray(rng.standard_normal((n, d, d)))
        for want_q in (True, False):
            label = "reduced" if want_q else "r"
            suite.measure(
                case=f"qr_{d}x{d}_n{n}_{label}",
                params={
                    "d": d,
                    "n": n,
                    "mode": label,
                    "batch_min": BATCH_MIN,
                    "module_chunk": CHUNK,
                },
                baseline=(
                    "numpy.linalg.qr",
                    lambda a=a, w=want_q: stock_call(a, w),
                ),
                candidates={
                    "householder_chunk1024": lambda a=a, w=want_q: qr_candidate(a, 1024, w),
                    "householder_chunk4096": lambda a=a, w=want_q: qr_candidate(a, 4096, w),
                },
                check=qr_close(1e-9),
                samples=SAMPLES,
            )

if not SMOKE:
    suite.save()
