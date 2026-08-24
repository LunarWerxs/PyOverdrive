"""Batch-4 regime calibration: the floors the OPP-000029/30/32/34 batteries
left open.

Each block chases one predicate edge exposed by the clean-box reproducer
runs (benchmarks/results/OPP-0000{29,30,32,34}/9bbe7063c555.json):

- EIGVALSH floor: closed form won 2.51x at batch 100 and lost 0.46x at
  batch 1; probe 10 and 30. Plus one float32 cell (unmeasured dtype) and
  one complex128 Hermitian cell is deliberately NOT probed (formula
  differs; out of scope until a record demands it).
- MATMUL regime: split_gemm lost most shapes on Intel but won 3.1x when
  C is short-and-wide against a large R ((64,2000)@(2000,2000)), where
  stock's complex upcast of R dominates. Sweep m x (n=q) and a q-only
  axis to see if "m small, R large" is a crisp, citable regime.
- ROLL edges: concat route measured only n >= 99 and int64/float64;
  probe n 8/32, the dtypes int32/float32/bool, and the shift=0 copy
  route at small n.
- ARGMAX rows floor: blocked transpose route won at 10000x10000 (2.5x)
  and 100000x100 (1.9x) but lost at 1000x1000; probe 3000/5000 rows and
  a (10000, 1000) cell, plus the NaN-salted correctness/timing cell the
  reproducer only ran for the non-blocked route.

Result JSON: benchmarks/results/BATCH4-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_batch4_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
rng = np.random.default_rng(20260824)

suite = BenchSuite("BATCH4-CAL", "regime edges for the batch-4 candidates")

SAMPLES = 3 if SMOKE else 9


# --- eigvalsh closed form ----------------------------------------------------

def eigvalsh2x2(a):
    a00 = a[..., 0, 0]
    a10 = a[..., 1, 0]
    a11 = a[..., 1, 1]
    mid = 0.5 * (a00 + a11)
    disc = np.sqrt((0.5 * (a00 - a11)) ** 2 + a10 * a10)
    return np.stack([mid - disc, mid + disc], axis=-1)


def sym_batch(n, dtype=np.float64):
    a = rng.uniform(-1.0, 1.0, size=(n, 2, 2)).astype(dtype)
    return np.ascontiguousarray(a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(2, dtype=dtype))


def values_close(rtol):
    def check(cand, base):
        if cand.dtype != base.dtype or cand.shape != base.shape:
            return False
        scale = np.maximum(1e-30, np.abs(base).max(axis=-1, keepdims=True))
        return bool((np.abs(cand - base) <= rtol * scale).all())

    return check


EIG_BATCHES = [100] if SMOKE else [10, 30, 100]
for n in EIG_BATCHES:
    a = sym_batch(n)
    suite.measure(
        case=f"eigvalsh_2x2_batch{n}_float64",
        params={"family": "eigvalsh", "batch": n, "dtype": "float64"},
        baseline=("numpy.linalg.eigvalsh", lambda a=a: np.linalg.eigvalsh(a)),
        candidates={"closed_form_values": lambda a=a: eigvalsh2x2(a)},
        check=values_close(1e-9),
        samples=SAMPLES if SMOKE else 15,
    )
if not SMOKE:
    a32 = sym_batch(10_000, np.float32)
    suite.measure(
        case="eigvalsh_2x2_batch10000_float32",
        params={"family": "eigvalsh", "batch": 10_000, "dtype": "float32"},
        baseline=("numpy.linalg.eigvalsh", lambda a=a32: np.linalg.eigvalsh(a32)),
        candidates={"closed_form_values": lambda a=a32: eigvalsh2x2(a32)},
        check=values_close(1e-3),
        samples=SAMPLES,
    )


# --- matmul split regime -----------------------------------------------------

def split_gemm(c, r):
    out = np.empty((c.shape[0], r.shape[1]), dtype=c.dtype)
    np.matmul(c.real, r, out=out.real)
    np.matmul(c.imag, r, out=out.imag)
    return out


def gemm_check(rtol):
    def check(cand, base):
        if cand.dtype != base.dtype or cand.shape != base.shape:
            return False
        scale = max(1.0, float(np.abs(base).max()))
        return bool(np.allclose(cand, base, rtol=rtol, atol=rtol * scale))

    return check


if not SMOKE:
    MM_CELLS = [(m, n, n) for m in (16, 64, 256) for n in (1000, 2000, 4000)]
    MM_CELLS += [(64, 2000, 500), (64, 2000, 1000)]
    for m, n, q in MM_CELLS:
        for label, cdt, rdt, rtol in (
            ("complex128", np.complex128, np.float64, 1e-12),
            ("complex64", np.complex64, np.float32, 1e-5),
        ):
            c = (
                rng.uniform(0.5, 1.5, size=(m, n)) + 1j * rng.uniform(0.5, 1.5, size=(m, n))
            ).astype(cdt)
            r = rng.uniform(0.5, 1.5, size=(n, q)).astype(rdt)
            suite.measure(
                case=f"matmul_C{m}x{n}_R{n}x{q}_{label}",
                params={"family": "matmul", "m": m, "n": n, "q": q, "dtype": label},
                baseline=("numpy.matmul", lambda c=c, r=r: np.matmul(c, r)),
                candidates={"split_gemm": lambda c=c, r=r: split_gemm(c, r)},
                check=gemm_check(rtol),
                samples=7,
            )


# --- roll edges ----------------------------------------------------------------

def roll_concat(d, s):
    s = s % d.size
    if s == 0:
        return d.copy(order="K")
    return np.concatenate((d[-s:], d[:-s]))


def exact(cand, base):
    return cand.dtype == base.dtype and cand.shape == base.shape and bool(
        np.array_equal(cand, base)
    )


ROLL_DTYPES = {
    "int64": lambda n: rng.integers(-(2**40), 2**40, size=n, dtype=np.int64),
    "float64": lambda n: rng.random(n),
    "int32": lambda n: rng.integers(-(2**30), 2**30, size=n, dtype=np.int32),
    "float32": lambda n: rng.random(n, dtype=np.float32),
    "bool": lambda n: rng.integers(0, 2, size=n).astype(bool),
}
if SMOKE:
    ROLL_CELLS = [("int64", 99, 1)]
else:
    ROLL_CELLS = [(d, n, 1) for d in ("int64", "float64") for n in (8, 32)]
    ROLL_CELLS += [(d, n, 1) for d in ("int32", "float32", "bool") for n in (99, 1_000, 10_000)]
    ROLL_CELLS += [(d, n, 0) for d in ("int64", "float64") for n in (99, 1_000, 10_000)]
for label, n, s in ROLL_CELLS:
    d = ROLL_DTYPES[label](n)
    suite.measure(
        case=f"roll_1d_n{n}_{label}_shift{s}",
        params={"family": "roll", "n": n, "dtype": label, "shift": s},
        baseline=("numpy.roll", lambda d=d, s=s: np.roll(d, s)),
        candidates={"concat_slices": lambda d=d, s=s: roll_concat(d, s)},
        check=exact,
        samples=SAMPLES if SMOKE else 15,
    )


# --- argmax blocked route: rows floor + NaN cell -------------------------------

def blocked_relayout_argmax(a, block=128):
    rows, cols = a.shape
    out = np.empty((cols, rows), dtype=a.dtype)
    for j0 in range(0, cols, block):
        j1 = min(j0 + block, cols)
        for i0 in range(0, rows, block):
            i1 = min(i0 + block, rows)
            out[j0:j1, i0:i1] = a[i0:i1, j0:j1].T
    return np.argmax(out, axis=1)


if not SMOKE:
    for rows, cols in ((3_000, 3_000), (5_000, 5_000), (10_000, 1_000), (5_000, 20_000)):
        a = rng.random(size=(rows, cols))
        suite.measure(
            case=f"argmax_axis0_{rows}x{cols}_float64",
            params={"family": "argmax", "rows": rows, "cols": cols, "dtype": "float64"},
            baseline=("numpy.argmax", lambda a=a: np.argmax(a, axis=0)),
            candidates={"blocked_relayout_argmax": lambda a=a: blocked_relayout_argmax(a)},
            check=exact,
            samples=7,
        )
    a = rng.random(size=(20_000, 200))
    nan_rows = rng.integers(0, 20_000, size=200)
    a[nan_rows, np.arange(200)] = np.nan
    suite.measure(
        case="argmax_axis0_20000x200_nan_salted_blocked",
        params={"family": "argmax", "rows": 20_000, "cols": 200, "nan": True},
        baseline=("numpy.argmax", lambda a=a: np.argmax(a, axis=0)),
        candidates={"blocked_relayout_argmax": lambda a=a: blocked_relayout_argmax(a)},
        check=exact,
        samples=9,
    )

if not SMOKE:
    suite.save()
