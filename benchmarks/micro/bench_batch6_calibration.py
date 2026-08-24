"""Batch-6 calibration battery: nan-family wrappers, exact-BLAS integer
matmul, and small-batch det/solve closed forms.

Triage probes on the dev box (2026-08-24, loaded) measured: nanmean
1.9-13.3x via isnan-scan-then-mean, nansum 8.8x, nanstd/nanvar 1.8-1.9x,
nanargmax 14-21x, nanmedian 2-D 2.7x, int64/int32 matmul 6.7-14.8x via
exact float64 BLAS (bit-equal within the 2^53 accumulation bound), det
(5000,3,3) 19.8x and Cramer solve 6.25x closed-form (relerr ~3e-13).
This battery measures the regime edges on a clean box so floors come
from measured cells, per house law.

Candidate routes mirror what the modules will ship, including every
guard cost (the isnan scan, the abs-max bound checks, the casts).
Wasted-scan cells (NaNs present, scan then stock fallback) measure the
guard's overhead in the losing direction.

Result JSON: benchmarks/results/BATCH6-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_batch6_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("BATCH6-CAL", "nan-family, int-BLAS matmul, small-batch det/solve")
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


# --- 1. nan-reductions: isnan-scan then plain reduction ---------------------

def scan_or(fast, fallback, a, **kw):
    # the shipped shape: one cheap NaN probe, plain op if clean, stock if not
    if np.isnan(np.min(a)):
        return fallback(a, **kw)
    return fast(a, **kw)


NAN_RED_CASES = [
    ("nanmean_1d", 100, None, np.nanmean, np.mean, 0.0),
    ("nanmean_1d", 300, None, np.nanmean, np.mean, 0.0),
    ("nanmean_1d", 1_000, None, np.nanmean, np.mean, 0.0),
    ("nanmean_1d", 10_000, None, np.nanmean, np.mean, 0.0),
    ("nanmean_1d", 100_000, None, np.nanmean, np.mean, 0.0),
    ("nanmean_2d", (1000, 100), 1, np.nanmean, np.mean, 0.0),
    ("nanmean_2d", (100, 1000), 1, np.nanmean, np.mean, 0.0),
    ("nanmean_1d_wasted", 100_000, None, np.nanmean, np.mean, 0.01),
    ("nansum_1d", 3_000, None, np.nansum, np.sum, 0.0),
    ("nansum_1d", 10_000, None, np.nansum, np.sum, 0.0),
    ("nansum_1d", 100_000, None, np.nansum, np.sum, 0.0),
    ("nanstd_1d", 3_000, None, np.nanstd, np.std, 0.0),
    ("nanstd_2d", (100, 100), 1, np.nanstd, np.std, 0.0),
    ("nanstd_2d", (1000, 100), 1, np.nanstd, np.std, 0.0),
    ("nanvar_1d", 3_000, None, np.nanvar, np.var, 0.0),
    ("nanvar_2d", (100, 100), 1, np.nanvar, np.var, 0.0),
    ("nanvar_2d", (1000, 100), 1, np.nanvar, np.var, 0.0),
]
if SMOKE:
    NAN_RED_CASES = NAN_RED_CASES[:2]

for label, shape, axis, nanfn, fastfn, nan_frac in NAN_RED_CASES:
    a = rng.standard_normal(shape)
    if nan_frac:
        a[rng.random(np.shape(a)) < nan_frac] = np.nan
    kw = {} if axis is None else {"axis": axis}
    n = a.size
    suite.measure(
        case=f"{label}_n{n}",
        params={"shape": list(np.shape(a)), "axis": axis, "nan_frac": nan_frac},
        baseline=(nanfn.__name__, lambda a=a, kw=kw, f=nanfn: f(a, **kw)),
        candidates={
            "scan_route": lambda a=a, kw=kw, f=fastfn, g=nanfn: scan_or(f, g, a, **kw)
        },
        check=exact,
        samples=SAMPLES,
    )

# --- 2. nanargmax / nanargmin ----------------------------------------------

NANARG_CASES = [
    ("nanargmax_1d", 300, None, 0.0),
    ("nanargmax_1d", 1_000, None, 0.0),
    ("nanargmax_1d", 10_000, None, 0.0),
    ("nanargmax_1d", 100_000, None, 0.0),
    ("nanargmax_2d", (1000, 1000), 1, 0.0),
    ("nanargmax_1d_wasted", 100_000, None, 0.01),
]
if SMOKE:
    NANARG_CASES = NANARG_CASES[:1]

for label, shape, axis, nan_frac in NANARG_CASES:
    a = rng.standard_normal(shape)
    if nan_frac:
        a[rng.random(np.shape(a)) < nan_frac] = np.nan
    kw = {} if axis is None else {"axis": axis}
    suite.measure(
        case=f"{label}_n{a.size}",
        params={"shape": list(np.shape(a)), "axis": axis, "nan_frac": nan_frac},
        baseline=("nanargmax", lambda a=a, kw=kw: np.nanargmax(a, **kw)),
        candidates={
            "scan_route": lambda a=a, kw=kw: scan_or(np.argmax, np.nanargmax, a, **kw)
        },
        check=exact,
        samples=SAMPLES,
    )

# --- 3. nanmedian 2-D many-slice -------------------------------------------

NANMED_CASES = [
    ((500, 500), 1, 0.0),
    ((2000, 200), 1, 0.0),
    ((100, 2000), 1, 0.0),
    ((1000, 1000), 1, 0.0),
    ((200, 10_000), 1, 0.0),
    ((1000, 1000), 1, 0.01),
]
if SMOKE:
    NANMED_CASES = NANMED_CASES[:1]

for shape, axis, nan_frac in NANMED_CASES:
    a = rng.standard_normal(shape)
    if nan_frac:
        a[rng.random(shape) < nan_frac] = np.nan
    tag = "wasted" if nan_frac else "clean"
    suite.measure(
        case=f"nanmedian_2d_{tag}_{shape[0]}x{shape[1]}",
        params={"shape": list(shape), "axis": axis, "nan_frac": nan_frac},
        baseline=("nanmedian", lambda a=a, axis=axis: np.nanmedian(a, axis=axis)),
        candidates={
            "scan_route": lambda a=a, axis=axis: scan_or(
                np.median, np.nanmedian, a, axis=axis
            )
        },
        check=exact,
        samples=SAMPLES,
    )

# --- 4. integer matmul via exact float64 BLAS ------------------------------

def int_blas_matmul(x, y, bound):
    # the shipped shape, guards included in the timing:
    # exactness bound: k * max|x| * max|y| < bound means every f64
    # partial sum is an exactly-representable integer
    mx = np.max(np.abs(x)) if x.size else 0
    my = np.max(np.abs(y)) if y.size else 0
    k = x.shape[-1]
    if k * float(mx) * float(my) >= bound:
        return x @ y
    r = x.astype(np.float64) @ y.astype(np.float64)
    return r.astype(x.dtype)


INTMM_CASES = [50, 100, 200, 400] if SMOKE else [30, 50, 100, 200, 400, 800]

for dt, bound in ((np.int64, 2.0**53), (np.int32, 2.0**31)):
    for n in INTMM_CASES:
        x = rng.integers(-1000, 1000, (n, n)).astype(dt)
        y = rng.integers(-1000, 1000, (n, n)).astype(dt)
        suite.measure(
            case=f"matmul_{np.dtype(dt).name}_{n}",
            params={"dtype": np.dtype(dt).name, "n": n},
            baseline=("matmul", lambda x=x, y=y: x @ y),
            candidates={
                "f64_blas": lambda x=x, y=y, b=bound: int_blas_matmul(x, y, b)
            },
            check=exact,
            samples=SAMPLES,
        )
        if SMOKE:
            break

# --- 4b. nan_to_num full-semantics where-route -----------------------------

def ntn_route(a):
    # full stock semantics for the float64 default-args regime: NaN -> 0.0,
    # +inf -> f64 max, -inf -> f64 min, always a copy
    out = a.copy()
    np.copyto(out, 0.0, where=np.isnan(out))
    isinf = np.isinf(out)
    if isinf.any():
        info = np.finfo(out.dtype)
        np.copyto(out, info.max, where=isinf & (out > 0))
        np.copyto(out, info.min, where=isinf & (out < 0))
    return out


NTN_CASES = [
    ("nan_only", 1_000_000, 0.01, 0.0),
    ("nan_inf", 1_000_000, 0.01, 0.005),
    ("clean", 1_000_000, 0.0, 0.0),
    ("nan_only_small", 10_000, 0.01, 0.0),
]
if SMOKE:
    NTN_CASES = NTN_CASES[:1]

for tag, n, nan_frac, inf_frac in NTN_CASES:
    z = rng.standard_normal(n)
    if nan_frac:
        z[rng.random(n) < nan_frac] = np.nan
    if inf_frac:
        z[rng.random(n) < inf_frac] = np.inf
        z[rng.random(n) < inf_frac] = -np.inf
    suite.measure(
        case=f"nan_to_num_{tag}_n{n}",
        params={"n": n, "nan_frac": nan_frac, "inf_frac": inf_frac},
        baseline=("nan_to_num", lambda z=z: np.nan_to_num(z)),
        candidates={"where_route": lambda z=z: ntn_route(z)},
        check=exact,
        samples=SAMPLES,
    )

# --- 5. det / solve small-batch closed forms -------------------------------

def det3_closed(m):
    a, b, c = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    d, e, f = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    g, h, i = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def det2_closed(m):
    return m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0]


def cramer3(m, b):
    a11, a12, a13 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    a21, a22, a23 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    a31, a32, a33 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    b1, b2, b3 = b[:, 0, 0], b[:, 1, 0], b[:, 2, 0]
    c11 = a22 * a33 - a23 * a32
    c12 = a23 * a31 - a21 * a33
    c13 = a21 * a32 - a22 * a31
    det = a11 * c11 + a12 * c12 + a13 * c13
    x1 = b1 * c11 + b2 * (a13 * a32 - a12 * a33) + b3 * (a12 * a23 - a13 * a22)
    x2 = b1 * c12 + b2 * (a11 * a33 - a13 * a31) + b3 * (a13 * a21 - a11 * a23)
    x3 = b1 * c13 + b2 * (a12 * a31 - a11 * a32) + b3 * (a11 * a22 - a12 * a21)
    out = np.empty_like(b)
    out[:, 0, 0] = x1 / det
    out[:, 1, 0] = x2 / det
    out[:, 2, 0] = x3 / det
    return out


def cramer2(m, b):
    a11, a12 = m[:, 0, 0], m[:, 0, 1]
    a21, a22 = m[:, 1, 0], m[:, 1, 1]
    b1, b2 = b[:, 0, 0], b[:, 1, 0]
    det = a11 * a22 - a12 * a21
    out = np.empty_like(b)
    out[:, 0, 0] = (b1 * a22 - b2 * a12) / det
    out[:, 1, 0] = (a11 * b2 - a21 * b1) / det
    return out


def slogdet_closed(detfn):
    def _run(m):
        det = detfn(m)
        return np.sign(det), np.log(np.abs(det))

    return _run


def slogdet_check(c, b):
    return (
        np.array_equal(c[0], b[0])
        and bool(np.allclose(c[1], b[1], rtol=1e-9, atol=1e-12, equal_nan=True))
    )


BATCH_NS = [100, 1_000] if SMOKE else [30, 100, 300, 1_000, 5_000, 20_000]

for d, detfn, solfn in ((3, det3_closed, cramer3), (2, det2_closed, cramer2)):
    for nb in BATCH_NS:
        m = rng.standard_normal((nb, d, d))
        b = rng.standard_normal((nb, d, 1))
        suite.measure(
            case=f"det_{d}x{d}_batch{nb}",
            params={"d": d, "batch": nb},
            baseline=("linalg.det", lambda m=m: np.linalg.det(m)),
            candidates={"closed_form": lambda m=m, f=detfn: f(m)},
            check=close(1e-9),
            samples=SAMPLES,
        )
        suite.measure(
            case=f"solve_{d}x{d}_batch{nb}",
            params={"d": d, "batch": nb},
            baseline=("linalg.solve", lambda m=m, b=b: np.linalg.solve(m, b)),
            candidates={"cramer": lambda m=m, b=b, f=solfn: f(m, b)},
            check=close(1e-8),
            samples=SAMPLES,
        )
        suite.measure(
            case=f"slogdet_{d}x{d}_batch{nb}",
            params={"d": d, "batch": nb},
            baseline=("linalg.slogdet", lambda m=m: np.linalg.slogdet(m)),
            candidates={"closed_form": lambda m=m, f=detfn: slogdet_closed(f)(m)},
            check=slogdet_check,
            samples=SAMPLES,
        )
        if SMOKE:
            break

if not SMOKE:
    suite.save()
