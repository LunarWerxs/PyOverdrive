"""Batch-7 calibration battery: cholesky_small_batch, eigvalsh_3x3_trig,
einsum_optimize_chain, and the nan_to_num_where kwargs extension.

Each block chases the floor(s) each shipped module's docstring still marks
provisional (pending this idle-box run, fp 9bbe7063c555):

- CHOLESKY closed form (d=2/3): dev-box probe measured 1.20x at batch 100
  rising to 9.95x at 10_000 (d=2), and 0.74x at 100 rising to ~2x at
  1_000-100_000 (d=3). Candidate pays the pivot-guard pass (_pivots +
  the PIVOT_RTOL bound check), the same cost _applicable pays, exactly as
  linalg_small_batch's own calibration did for its det-vs-scale guard.
- EIGVALSH 3x3 trig: dev-box probe measured 1.41x at batch 100 rising to
  7.06x at 1000; candidate pays the isfinite scan. Plus one ill-conditioned
  witness (eigenvalue spread 1e12, QR-orthogonal construction) mirroring
  the 2x2 battery's OPP-000030 witness cell: a check failure there is a
  regime finding, recorded either way, not a battery bug.
- EINSUM chain (>=3 operands): the module docstring cites 249x at n=64
  (volume 16.8M) on ij,jk,kl->il; this sweeps the volume floor plus a
  4-operand cell and a scalar-output cell near the provisional floor.
  Candidate pays the _chain_volume predicate, mirroring _applicable_chain.
- NAN_TO_NUM kwargs extension: SIZE_FLOOR=10_000 is already shipped from
  BATCH6-CAL's default-args regime; this sweeps the three kwargs regimes
  (nan= override, nan+posinf mix, clean-with-overrides) from below that
  floor (n=3000) through 1e6 to confirm the kwargs path crosses at the
  same floor. Candidate is the module's _run with kwargs, bit-identical
  by contract.

Result JSON: benchmarks/results/BATCH7-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_batch7_calibration.py [--smoke]
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
    _pivots as _chol_pivots,
    _run as _chol_run,
    PIVOT_RTOL,
)
from pyoverdrive.fastpaths.eigvalsh_3x3 import _run as _eig_run, BATCH_MIN
from pyoverdrive.fastpaths.einsum_optimize import _chain_volume, CHAIN_VOLUME_FLOOR
from pyoverdrive.fastpaths.nan_to_num_where import _run as _ntn_run, SIZE_FLOOR

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite(
    "BATCH7-CAL",
    "cholesky/eigvalsh_3x3 closed forms, einsum chain, nan_to_num kwargs",
)
rng = np.random.default_rng(20260824)


def exact(c, b):
    c = np.asarray(c)
    b = np.asarray(b)
    return c.shape == b.shape and c.dtype == b.dtype and bool(
        np.array_equal(c, b, equal_nan=True)
    )


def close(rtol):
    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        return c.shape == b.shape and c.dtype == b.dtype and bool(
            np.allclose(c, b, rtol=rtol, atol=0.0, equal_nan=True)
        )

    return _chk


def values_close(rtol):
    # eigvalsh precedent (bench_batch4_calibration.py / OPP-000030): scaled
    # per-batch-element by the largest eigenvalue magnitude, not a plain rtol.
    def check(cand, base):
        if cand.dtype != base.dtype or cand.shape != base.shape:
            return False
        scale = np.maximum(1e-30, np.abs(base).max(axis=-1, keepdims=True))
        return bool((np.abs(cand - base) <= rtol * scale).all())

    return check


def chain_check(c, b):
    # EINSUM-CAL precedent: absolute-scaled, not per-element relative, since
    # summation order differs and near-zero output elements are dominated by
    # accumulation noise on both routes at float32.
    c = np.asarray(c)
    b = np.asarray(b)
    if c.shape != b.shape:
        return False
    scale = max(1.0, float(np.abs(b).max())) if b.size else 1.0
    if b.dtype == np.float32:
        return bool(np.allclose(c, b, rtol=1e-3, atol=1e-4 * scale))
    return bool(np.allclose(c, b, rtol=1e-9, atol=1e-9 * scale))


# --- 1. cholesky closed form: pivot-guard pass + _run -----------------------

def spd_batch(nb, d):
    m = rng.standard_normal((nb, d, d))
    return np.ascontiguousarray(m @ np.swapaxes(m, -1, -2) + d * np.eye(d))


def chol_candidate(a):
    pivots, scale = _chol_pivots(a)
    bound = PIVOT_RTOL * np.maximum(scale, 1e-100)
    assert all(bool((p >= bound).all()) for p in pivots), "pivot guard refused SPD witness"
    return _chol_run(a)


# 2000/3000 sit inside the 3x3 window whose edges the module docstring
# cites; 20000 documents the Intel wash that motivates the 3x3 cap
CHOL_BATCHES = [30, 100] if SMOKE else [30, 100, 300, 1_000, 2_000, 3_000, 5_000, 20_000]

for d in (2, 3):
    for nb in CHOL_BATCHES:
        a = spd_batch(nb, d)
        suite.measure(
            case=f"cholesky_{d}x{d}_batch{nb}",
            params={"d": d, "batch": nb, "window": list(CHOL_WINDOWS[d])},
            baseline=("numpy.linalg.cholesky", lambda a=a: np.linalg.cholesky(a)),
            candidates={"closed_form_guarded": lambda a=a: chol_candidate(a)},
            check=close(1e-9),
            samples=SAMPLES,
        )

# --- 2. eigvalsh 3x3 trig: isfinite scan + _run ------------------------------

def sym_batch3(n, dtype=np.float64):
    a = rng.uniform(-1.0, 1.0, size=(n, 3, 3)).astype(dtype)
    return np.ascontiguousarray(a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(3, dtype=dtype))


def eig_candidate(a):
    assert bool(np.isfinite(a).all()), "isfinite guard refused finite witness"
    return _eig_run(a)


EIG_BATCHES = [30, 100] if SMOKE else [30, 100, 300, 1_000, 10_000, 100_000]

for dtype, rtol in ((np.float64, 1e-9), (np.float32, 1e-3)):
    dt = np.dtype(dtype).name
    for n in EIG_BATCHES:
        a = sym_batch3(n, dtype)
        suite.measure(
            case=f"eigvalsh_3x3_batch{n}_{dt}",
            params={"batch": n, "dtype": dt, "batch_min": BATCH_MIN},
            baseline=("numpy.linalg.eigvalsh", lambda a=a: np.linalg.eigvalsh(a)),
            candidates={"closed_form_guarded": lambda a=a: eig_candidate(a)},
            check=values_close(rtol),
            samples=SAMPLES,
        )

if not SMOKE:
    # ill-conditioned witness: eigenvalue spread ~1e12 via a QR-orthogonal
    # construction (mirrors OPP-000030's 2x2 witness); a check failure here
    # is a regime finding, recorded either way, not a battery bug.
    n = 10_000
    q = rng.uniform(-1.0, 1.0, size=(n, 3, 3))
    qq, _ = np.linalg.qr(q)
    w_big = np.stack([np.full(n, 1e-12), np.full(n, 1.0), np.full(n, 1.0)], axis=-1)
    a_ill = np.ascontiguousarray(qq @ (w_big[..., None] * np.swapaxes(qq, -1, -2)))
    a_ill = 0.5 * (a_ill + np.swapaxes(a_ill, -1, -2))
    suite.measure(
        case=f"eigvalsh_3x3_batch{n}_illconditioned",
        params={"batch": n, "dtype": "float64", "condition": 1e12},
        baseline=("numpy.linalg.eigvalsh", lambda a=a_ill: np.linalg.eigvalsh(a)),
        candidates={"closed_form_guarded": lambda a=a_ill: eig_candidate(a)},
        # This spectrum has a coalesced pair (1, 1), so since the
        # DEGENERACY_MIN bail landed, _run detects 1 - r^2 < 1e-12 mid-run
        # and falls back to stock for the whole stack: the cell now measures
        # the bail's wasted-compute overhead (the honest losing direction),
        # and the check is stock-vs-stock at the full contract tolerance. A
        # correctness failure here would mean the bail stopped firing.
        check=values_close(1e-9),
        samples=SAMPLES,
    )

# --- 3. einsum chain: >=3 operands, volume gate ------------------------------

def chain_candidate(subs, *operands):
    _chain_volume(subs, operands)  # pay the predicate cost, mirroring dispatch
    res = np.einsum(subs, *operands, optimize=True)
    if type(res) is np.ndarray and res.ndim == 0:
        return res[()]
    return res


CHAIN_NS = [8, 12] if SMOKE else [8, 12, 16, 24, 32, 64]
CHAIN_SUBS = "ij,jk,kl->il"

for n in CHAIN_NS:
    a = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float64)
    b = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float64)
    c = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float64)
    suite.measure(
        case=f"chain_ijjkkl_n{n}_float64",
        params={
            "pattern": CHAIN_SUBS, "n": n, "dtype": "float64",
            "volume": n**4, "volume_floor": CHAIN_VOLUME_FLOOR,
        },
        baseline=("numpy.einsum", lambda s=CHAIN_SUBS, a=a, b=b, c=c: np.einsum(s, a, b, c)),
        candidates={
            "optimize_chain": lambda s=CHAIN_SUBS, a=a, b=b, c=c: chain_candidate(s, a, b, c)
        },
        check=chain_check,
        samples=SAMPLES,
    )

if not SMOKE:
    # one float32 cell at the largest n
    n = 64
    a32 = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float32)
    b32 = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float32)
    c32 = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float32)
    suite.measure(
        case=f"chain_ijjkkl_n{n}_float32",
        params={"pattern": CHAIN_SUBS, "n": n, "dtype": "float32", "volume": n**4},
        baseline=("numpy.einsum", lambda s=CHAIN_SUBS, a=a32, b=b32, c=c32: np.einsum(s, a, b, c)),
        candidates={
            "optimize_chain": lambda s=CHAIN_SUBS, a=a32, b=b32, c=c32: chain_candidate(s, a, b, c)
        },
        check=chain_check,
        samples=SAMPLES,
    )

    # one 4-operand cell, volume comfortably above the floor
    n4 = 20
    subs4 = "ij,jk,kl,lm->im"
    ops4 = [rng.uniform(-1.0, 1.0, size=(n4, n4)).astype(np.float64) for _ in range(4)]
    suite.measure(
        case=f"chain_4op_ijjkklm_n{n4}_float64",
        params={"pattern": subs4, "n": n4, "dtype": "float64", "volume": n4**5},
        baseline=("numpy.einsum", lambda s=subs4, ops=ops4: np.einsum(s, *ops)),
        candidates={"optimize_chain": lambda s=subs4, ops=ops4: chain_candidate(s, *ops)},
        check=chain_check,
        samples=SAMPLES,
    )

    # one scalar-output cell: volume 64^3 = 262_144 is exactly
    # CHAIN_SCALAR_VOLUME_FLOOR, the measured cell that floor comes from
    n5 = 64
    subs5 = "ij,jk,ki->"
    ops5 = [rng.uniform(-1.0, 1.0, size=(n5, n5)).astype(np.float64) for _ in range(3)]
    suite.measure(
        case=f"chain_scalar_ijjkki_n{n5}_float64",
        params={"pattern": subs5, "n": n5, "dtype": "float64", "volume": n5**3},
        baseline=("numpy.einsum", lambda s=subs5, ops=ops5: np.einsum(s, *ops)),
        candidates={"optimize_chain": lambda s=subs5, ops=ops5: chain_candidate(s, *ops)},
        check=chain_check,
        samples=SAMPLES,
    )

# --- 4. nan_to_num kwargs: nan=/posinf=/neginf= overrides --------------------

NTN_KW_CASES = [
    ("nan_1pct", {"nan": 1.5}, 0.01, 0.0),
    ("nan_inf_mix", {"nan": 1.5, "posinf": 100.0}, 0.01, 0.01),
    ("clean_overrides", {"nan": 1.5, "posinf": 100.0, "neginf": -100.0}, 0.0, 0.0),
]
NTN_KW_NS = [3_000, 10_000] if SMOKE else [3_000, 10_000, 100_000, 1_000_000]

for label, kwargs, nan_frac, inf_frac in NTN_KW_CASES:
    for n in NTN_KW_NS:
        z = rng.standard_normal(n)
        if nan_frac:
            z[rng.random(n) < nan_frac] = np.nan
        if inf_frac:
            z[rng.random(n) < inf_frac] = np.inf
            z[rng.random(n) < inf_frac] = -np.inf
        suite.measure(
            case=f"nan_to_num_kw_{label}_n{n}",
            params={
                "n": n, "kwargs": kwargs, "nan_frac": nan_frac, "inf_frac": inf_frac,
                "size_floor": SIZE_FLOOR,
            },
            baseline=("nan_to_num", lambda z=z, kw=kwargs: np.nan_to_num(z, **kw)),
            candidates={"where_route_kwargs": lambda z=z, kw=kwargs: _ntn_run(z, **kw)},
            check=exact,
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
