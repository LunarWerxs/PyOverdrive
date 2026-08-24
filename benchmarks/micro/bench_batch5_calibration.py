"""Batch-5 regime calibration: the floors and ceilings the OPP-000035..040
batteries left open.

- INV: batch floor between 100 (1.39x straddle) and 1000 (5.24x); probe
  300 and 500. Condition ceiling: the cond-1e10 witness FAILED the
  rtol-1e-9 scaled check; ladder cond {1e3, 1e6, 1e8} finds where the
  adjugate stops clearing it, which sets the predicate's det-ratio
  guard honestly.
- MEDIAN: cap between 1001 (2.15x) and 10000 (1.26x); probe 2000, 3000,
  5000.
- HIST2D: floor between 10x10 bins (1.09x) and 100x100 (1.65x); probe
  30x30 and 50x50 at 1e6 samples.
- ISIN-OBJECT: guarded floor between 105 combined (1.32x straddle) and
  1100 (7.3x); probe 300 and 600 combined.

Result JSON: benchmarks/results/BATCH5-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_batch5_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
rng = np.random.default_rng(20260825)

suite = BenchSuite("BATCH5-CAL", "regime edges for the batch-5 candidates")
SAMPLES = 3 if SMOKE else 11


# --- inv: batch floor + condition ceiling -----------------------------------

def adj_inv_3x3(a):
    m00 = a[..., 0, 0]; m01 = a[..., 0, 1]; m02 = a[..., 0, 2]
    m10 = a[..., 1, 0]; m11 = a[..., 1, 1]; m12 = a[..., 1, 2]
    m20 = a[..., 2, 0]; m21 = a[..., 2, 1]; m22 = a[..., 2, 2]
    c00 = m11 * m22 - m12 * m21
    c10 = m12 * m20 - m10 * m22
    c20 = m10 * m21 - m11 * m20
    det = m00 * c00 + m01 * c10 + m02 * c20
    inv_det = 1.0 / det
    out = np.empty_like(a)
    out[..., 0, 0] = c00 * inv_det
    out[..., 0, 1] = (m02 * m21 - m01 * m22) * inv_det
    out[..., 0, 2] = (m01 * m12 - m02 * m11) * inv_det
    out[..., 1, 0] = c10 * inv_det
    out[..., 1, 1] = (m00 * m22 - m02 * m20) * inv_det
    out[..., 1, 2] = (m02 * m10 - m00 * m12) * inv_det
    out[..., 2, 0] = c20 * inv_det
    out[..., 2, 1] = (m01 * m20 - m00 * m21) * inv_det
    out[..., 2, 2] = (m00 * m11 - m01 * m10) * inv_det
    return out


def inv_check(cand, base):
    if cand.dtype != base.dtype or cand.shape != base.shape:
        return False
    scale = np.abs(base).max(axis=(-2, -1), keepdims=True)
    return bool((np.abs(cand - base) <= 1e-9 * np.maximum(1.0, scale)).all())


def spd_batch(n):
    a = rng.uniform(-1.0, 1.0, size=(n, 3, 3))
    return np.ascontiguousarray(a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(3))


def cond_batch(n, cond):
    q, _ = np.linalg.qr(rng.uniform(-1.0, 1.0, size=(n, 3, 3)))
    w = np.stack([np.full(n, 1.0 / cond), np.full(n, 0.03), np.full(n, 1.0)], axis=-1)
    a = q @ (w[..., None] * np.swapaxes(q, -1, -2))
    return np.ascontiguousarray(0.5 * (a + np.swapaxes(a, -1, -2)))


INV_BATCHES = [500] if SMOKE else [300, 500]
for n in INV_BATCHES:
    a = spd_batch(n)
    suite.measure(
        case=f"inv_3x3_batch{n}_floorprobe",
        params={"family": "inv", "batch": n},
        baseline=("numpy.linalg.inv", lambda a=a: np.linalg.inv(a)),
        candidates={"adjugate": lambda a=a: adj_inv_3x3(a)},
        check=inv_check,
        samples=SAMPLES,
    )
if not SMOKE:
    for cond in (1e3, 1e6, 1e8):
        a = cond_batch(10_000, cond)
        suite.measure(
            case=f"inv_3x3_batch10000_cond{cond:.0e}",
            params={"family": "inv", "batch": 10_000, "condition": cond},
            baseline=("numpy.linalg.inv", lambda a=a: np.linalg.inv(a)),
            candidates={"adjugate": lambda a=a: adj_inv_3x3(a)},
            check=inv_check,
            samples=7,
        )


# --- median: cap probe -------------------------------------------------------

def partition_median(x):
    n = x.size
    k = n // 2
    if n % 2:
        p = np.partition(x, (k, n - 1))
        if np.isnan(p[-1]):
            return np.float64(np.nan)
        return p[k]
    p = np.partition(x, (k - 1, k, n - 1))
    if np.isnan(p[-1]):
        return np.float64(np.nan)
    return np.float64(0.5 * (p[k - 1] + p[k]))


def med_exact(cand, base):
    c, b = np.asarray(cand), np.asarray(base)
    return c.dtype == b.dtype and bool(np.array_equal(c, b, equal_nan=True))


MED_SIZES = [2_001] if SMOKE else [2_001, 3_001, 5_001, 2_000, 5_000]
for n in MED_SIZES:
    x = rng.random(n)
    suite.measure(
        case=f"median_1d_n{n}_capprobe",
        params={"family": "median", "n": n},
        baseline=("numpy.median", lambda x=x: np.median(x)),
        candidates={"partition_median": lambda x=x: partition_median(x)},
        check=med_exact,
        samples=SAMPLES if SMOKE else 15,
    )


# --- hist2d: bin floor -------------------------------------------------------

def _uniform_indices(x, lo, hi, nbins, edges):
    idx = np.floor((x - lo) * (nbins / (hi - lo))).astype(np.intp)
    np.clip(idx, 0, nbins - 1, out=idx)
    idx[x < edges[idx]] -= 1
    idx2 = idx + 1
    np.clip(idx2, 0, nbins, out=idx2)
    idx[x >= edges[idx2]] += 1
    np.clip(idx, 0, nbins - 1, out=idx)
    keep = (x >= edges[0]) & (x <= edges[-1])
    return idx, keep


def hist2d_uniform(x, y, bins, range_):
    nx, ny = bins
    (xlo, xhi), (ylo, yhi) = range_
    ex = np.linspace(xlo, xhi, nx + 1)
    ey = np.linspace(ylo, yhi, ny + 1)
    ix, keepx = _uniform_indices(x, xlo, xhi, nx, ex)
    iy, keepy = _uniform_indices(y, ylo, yhi, ny, ey)
    keep = keepx & keepy
    h = np.bincount(ix[keep] * ny + iy[keep], minlength=nx * ny).reshape(nx, ny)
    return h.astype(np.float64), ex, ey


def h2_check(cand, base):
    return (
        cand[0].dtype == base[0].dtype
        and np.array_equal(cand[0], base[0])
        and np.array_equal(cand[1], base[1])
        and np.array_equal(cand[2], base[2])
    )


if not SMOKE:
    xm = rng.normal(0.0, 1.0, size=1_000_000)
    ym = rng.normal(0.0, 1.0, size=1_000_000)
    for b in (30, 50):
        suite.measure(
            case=f"hist2d_n1000000_bins{b}x{b}_floorprobe",
            params={"family": "hist2d", "bins": [b, b]},
            baseline=(
                "numpy.histogram2d",
                lambda x=xm, y=ym, b=b: np.histogram2d(x, y, bins=[b, b], range=[[-3, 3], [-3, 3]]),
            ),
            candidates={
                "direct_index": lambda x=xm, y=ym, b=b: hist2d_uniform(
                    x, y, (b, b), ((-3.0, 3.0), (-3.0, 3.0))
                )
            },
            check=h2_check,
            samples=7,
        )


# --- isin object: guarded floor ----------------------------------------------

def _is_nanlike(x):
    try:
        return x != x
    except Exception:
        return False


def set_route_guarded(element, test):
    el = element.tolist()
    te = test.tolist()
    if any(_is_nanlike(x) for x in te) or any(_is_nanlike(x) for x in el):
        raise AssertionError("unexpected NaN-like")
    lookup = set(te)
    return np.fromiter((s in lookup for s in el), dtype=bool, count=element.size)


def iso_exact(cand, base):
    return cand.dtype == base.dtype and bool(np.array_equal(cand, base))


if not SMOKE:
    VOCAB = [f"key_{i:05d}" for i in range(2_000)]
    for n, m in ((280, 20), (550, 50)):
        el = np.array([VOCAB[i] for i in rng.integers(0, len(VOCAB), size=n)], dtype=object)
        te = np.array(
            [VOCAB[i] for i in rng.choice(len(VOCAB), size=m, replace=False)], dtype=object
        )
        suite.measure(
            case=f"isin_object_str_n{n}_m{m}_floorprobe",
            params={"family": "isin_object", "n": n, "m": m, "combined": n + m},
            baseline=("numpy.isin", lambda e=el, t=te: np.isin(e, t)),
            candidates={"set_route_nan_guarded": lambda e=el, t=te: set_route_guarded(e, t)},
            check=iso_exact,
            samples=15,
        )

if not SMOKE:
    suite.save()
