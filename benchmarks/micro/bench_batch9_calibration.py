"""Batch-9 calibration battery: fused-chunked cholesky, eigvalsh
split-and-recombine, einsum ellipsis admission.

Each block settles a constant the batch-9 rewrites left provisional
(pending this idle-box run, fp 9bbe7063c555):

- CHOLESKY_SMALL_BATCH (OPP-000047, fused rewrite): the guard now rides
  the factorization pass in CHUNK-sized blocks, so the candidate here IS
  the module's _run loop, driven at chunk 1024 vs 4096 to pick CHUNK.
  The grid spans the old window (floor 1000, 3x3 cap 3000), the old
  route's weak/bistable cells (2x2 20_000 at 1.29x; 3x3 5000 bistable),
  and the formerly-forfeited large-batch regime up to 1M. The window
  decision (keep the 3x3 cap lifted, or restore it) comes from these
  cells under the two-machine law.
- EIGVALSH_3X3 (OPP-000048, split-and-recombine): candidate is the
  module's own _run (guards, split, scatter included). Cells sweep the
  degenerate fraction 0..0.5 at n=10_000/100_000; DEGEN_FRAC_MAX
  (provisional 0.25) is set from where the split stops clearing the
  1.3x bar against one batched stock call.
- EINSUM ellipsis (OPP-000018/000049 extension): candidates route to
  optimize=True exactly as _run does. Cells witness the two-operand
  ellipsis form at/below PROJECTED_FLOOR and the ellipsis chain
  at/below CHAIN_VOLUME_FLOOR, checking the non-ellipsis floors
  transfer to the ellipsis spelling (same planner, same kernels; only
  the parse differs).

Result JSON: benchmarks/results/BATCH9-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_batch9_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from lab.dyno import BenchSuite

from pyoverdrive.fastpaths.cholesky_small_batch import (
    _WINDOWS as CHOL_WINDOWS,
    _chunk_for,
    _factor2,
    _factor3,
)
from pyoverdrive.fastpaths.eigvalsh_3x3 import (
    DEGEN_FRAC_MAX,
    DEGENERACY_MIN,
    _run as _eig_run,
)
from pyoverdrive.fastpaths.einsum_optimize import (
    CHAIN_VOLUME_FLOOR,
    PROJECTED_FLOOR,
)

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite(
    "BATCH9-CAL",
    "fused-chunked cholesky, eigvalsh split-and-recombine, einsum ellipsis",
)
rng = np.random.default_rng(20260824)


def scaled_close(rtol):
    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        if c.dtype != b.dtype or c.shape != b.shape:
            return False
        scale = max(1.0, float(np.abs(b).max())) if b.size else 1.0
        return bool(np.allclose(c, b, rtol=rtol, atol=rtol * scale))

    return _chk


# --- 1. cholesky: fused factor-and-guard, chunk grid ------------------------


def spd_batch(n, d, seed):
    local = np.random.default_rng(seed)
    m = local.standard_normal((n, d, d))
    return np.ascontiguousarray(m @ np.swapaxes(m, -1, -2) + d * np.eye(d))


def chol_candidate(a, chunk):
    # mirrors the module's _run exactly (same allocation, same reshape,
    # same per-chunk fused factor+guard), with the chunk size as the
    # calibration knob; a guard refusal cannot happen on these SPD cells
    d = a.shape[-1]
    factor = _factor2 if d == 2 else _factor3
    out = np.empty(a.shape, dtype=a.dtype)
    src = a.reshape(-1, d, d)
    dst = out.reshape(-1, d, d)
    n = src.shape[0]
    for s in range(0, n, chunk):
        if not factor(src[s : s + chunk], dst[s : s + chunk]):
            raise AssertionError("guard refused an SPD calibration cell")
    return out


CHOL_NS = {
    2: [1_000, 20_000]
    if SMOKE
    else [300, 1_000, 3_000, 5_000, 10_000, 20_000, 100_000, 1_000_000],
    3: [1_000, 5_000]
    if SMOKE
    else [300, 1_000, 3_000, 5_000, 10_000, 16_384, 20_000, 30_000, 100_000, 1_000_000],
}

for d, ns in CHOL_NS.items():
    for n in ns:
        a = spd_batch(n, d, seed=17 * d + n)
        suite.measure(
            case=f"chol_{d}x{d}_n{n}",
            params={
                "d": d,
                "n": n,
                "window": list(CHOL_WINDOWS[d]),
                "module_chunk": _chunk_for(d, n),
            },
            baseline=("numpy.linalg.cholesky", lambda a=a: np.linalg.cholesky(a)),
            candidates={
                "fused_chunk1024": lambda a=a: chol_candidate(a, 1024),
                "fused_chunk4096": lambda a=a: chol_candidate(a, 4096),
            },
            check=scaled_close(1e-9),
            samples=SAMPLES,
        )

# --- 2. eigvalsh 3x3: split-and-recombine over degenerate fractions --------

DEGEN = np.diag([1.0, 1.0, 5.0])  # exactly repeated pair: 1 - r^2 == 0


def sym_batch(n, frac, seed):
    local = np.random.default_rng(seed)
    m = local.standard_normal((n, 3, 3))
    a = np.ascontiguousarray((m + np.swapaxes(m, -1, -2)) / 2.0)
    nbad = int(round(frac * n))
    if nbad:
        a[local.choice(n, nbad, replace=False)] = DEGEN
    return a


EIG_CELLS = (
    [(10_000, 0.0), (10_000, 0.01)]
    if SMOKE
    else [
        (10_000, 0.0),
        (10_000, 0.001),
        (10_000, 0.01),
        (10_000, 0.1),
        (10_000, 0.25),
        (10_000, 0.5),
        (100_000, 0.0),
        (100_000, 0.01),
        (100_000, 0.1),
    ]
)

for n, frac in EIG_CELLS:
    a = sym_batch(n, frac, seed=n + int(frac * 1e6))
    suite.measure(
        case=f"eig3_n{n}_degen{frac:g}",
        params={
            "n": n,
            "degen_frac": frac,
            "degen_frac_max": DEGEN_FRAC_MAX,
            "degeneracy_min": DEGENERACY_MIN,
        },
        baseline=("numpy.linalg.eigvalsh", lambda a=a: np.linalg.eigvalsh(a)),
        candidates={"trig_split": lambda a=a: _eig_run(a)},
        check=scaled_close(1e-9),
        samples=SAMPLES,
    )

# --- 3. einsum ellipsis: floors transfer to the ellipsis spelling ----------


def stacks(ell, n, count, seed):
    local = np.random.default_rng(seed)
    return [local.standard_normal(ell + (n, n)) for _ in range(count)]


EINSUM_CELLS = (
    [("e2_ellipsis_B64_n32", "...ij,...jk->...ik", (64,), 32, 2)]
    if SMOKE
    else [
        # two-operand: min-operand-size gate (PROJECTED_FLOOR = 10_000)
        ("e2_ellipsis_B64_n32", "...ij,...jk->...ik", (64,), 32, 2),
        ("e2_ellipsis_floor_B10_n32", "...ij,...jk->...ik", (10,), 32, 2),
        ("e2_ellipsis_below_B8_n16", "...ij,...jk->...ik", (8,), 16, 2),
        ("e2_ellipsis_implicit_B64_n32", "...ij,...jk", (64,), 32, 2),
        # chain: the ellipsis spelling crosses 1.3x later than the label
        # spelling (planner works on the batched shape), so these cells
        # locate its own floor between the label floor and 262_144
        ("chain_ellipsis_vol20736", "...ij,...jk,...kl->...il", (1,), 12, 3),
        ("chain_ellipsis_vol65536", "...ij,...jk,...kl->...il", (1,), 16, 3),
        ("chain_ellipsis_vol76832", "...ij,...jk,...kl->...il", (2,), 14, 3),
        ("chain_ellipsis_vol131072", "...ij,...jk,...kl->...il", (2,), 16, 3),
        ("chain_ellipsis_vol196608", "...ij,...jk,...kl->...il", (3,), 16, 3),
        ("chain_ellipsis_vol262144", "...ij,...jk,...kl->...il", (4,), 16, 3),
        ("chain_ellipsis_B64_n32", "...ij,...jk,...kl->...il", (64,), 32, 3),
    ]
)

for case, subs, ell, n, count in EINSUM_CELLS:
    ops = stacks(ell, n, count, seed=n * count + ell[0])
    vol = int(np.prod(ell)) * n ** (count + 1)
    suite.measure(
        case=case,
        params={
            "subs": subs,
            "ell": list(ell),
            "n": n,
            "min_size": min(o.size for o in ops),
            "naive_volume": vol,
            "projected_floor": PROJECTED_FLOOR,
            "chain_volume_floor": CHAIN_VOLUME_FLOOR,
        },
        baseline=("numpy.einsum", lambda subs=subs, ops=ops: np.einsum(subs, *ops)),
        candidates={
            "optimize_true": lambda subs=subs, ops=ops: np.einsum(subs, *ops, optimize=True),
        },
        check=scaled_close(1e-9),
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
