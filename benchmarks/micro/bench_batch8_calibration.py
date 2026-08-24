"""Batch-8 calibration battery: hist1d_uniform, interp_uniform_grid, and
take_index_assign.

Each block chases the floor/window each shipped module's docstring still
marks provisional (pending this idle-box run, fp 9bbe7063c555):

- HIST1D_UNIFORM (OPP-000052): SIZE_WINDOW=(1000, 100_000) provisional
  from the idle-box probe (2.27x at n=1000, 2.17x at 3000, 1.86x at
  10_000, 4.41x at 30_000, 1.72x at 100_000, 0.98x at 300_000; bins=50).
  This sweeps below the floor (n=300) and above the ceiling (n=300_000)
  plus a bins sweep at n=10_000 (10/50/1000), one weighted cell, and one
  edge-salted cell (mirroring hist2d_uniform's differential-test
  salting) checked exact on both counts and edges. Candidate is the
  module's own _run: the predicate guards are cheap metadata, so no
  guard cost is mirrored separately.
- INTERP_UNIFORM_GRID (OPP-000050): NQ_MIN=10_000 provisional from the
  idle-box probe (5.83x at nq=10_000/grid 1000, 1.95x at 100_000/1000,
  3.10x at 1M/10_000, 1.38x at 1M/100; 1.06x at nq=1000, below the bar).
  This sweeps (nq, grid) pairs below and above the floor, one built from
  an arange grid rather than linspace. Candidate pays the _applicable
  guard cost (the diff-uniformity and isfinite scans) plus _run,
  mirroring dispatch exactly.
- TAKE_INDEX_ASSIGN (OPP-000051): SIZE_MIN=10_000 provisional from the
  idle-box probe (2.11x at 10_000 gathered elements, 1.70x at 100_000,
  1.39x at 1M, 1.35x at 10M; bit-identical). This sweeps gathered sizes
  below and above the floor for float64, plus one int64 cell. Candidate
  is the bare out[...] = a[indices] assignment (no guard: the module's
  own predicate is cheap metadata), matching the module's out= contract.

Result JSON: benchmarks/results/BATCH8-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_batch8_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from lab.dyno import BenchSuite

# hist1d_uniform was WITHDRAWN after this battery's first full run (the
# edge-correction the bit-identity contract requires costs the entire
# margin: 0.72-1.27x everywhere, see docs/research/batch8-shortlist.md).
# The route is inlined here so the decline evidence stays reproducible.
BINS_MIN = 2
SIZE_WINDOW = (1_000, 100_000)


def _hist_run(a, bins=10, range=None, density=None, weights=None):
    nbins = int(bins)
    lo, hi = float(range[0]), float(range[1])
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.floor((a - lo) * (nbins / (hi - lo))).astype(np.intp)
    np.clip(idx, 0, nbins - 1, out=idx)
    idx[a < edges[idx]] -= 1
    idx2 = idx + 1
    np.clip(idx2, 0, nbins, out=idx2)
    idx[a >= edges[idx2]] += 1
    np.clip(idx, 0, nbins - 1, out=idx)
    keep = (a >= edges[0]) & (a <= edges[-1])
    w = None if weights is None else weights[keep]
    h = np.bincount(idx[keep], weights=w, minlength=nbins)
    if weights is not None:
        h = h.astype(np.float64, copy=False)
    return h, edges


from pyoverdrive.fastpaths.interp_uniform_grid import (
    _run as _interp_run,
    NQ_MIN,
    UNIFORM_RTOL,
)
from pyoverdrive.fastpaths.take_index_assign import SIZE_MIN

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite(
    "BATCH8-CAL",
    "hist1d_uniform, interp_uniform_grid, take_index_assign direct-index paths",
)
rng = np.random.default_rng(20260824)


def exact(c, b):
    c = np.asarray(c)
    b = np.asarray(b)
    return c.shape == b.shape and c.dtype == b.dtype and bool(
        np.array_equal(c, b, equal_nan=True)
    )


def scaled_close(rtol):
    # absolute tolerance scaled by the batch's own magnitude, not a plain
    # per-element rtol - interp output crosses zero, so a relative bound
    # blows up near the crossing while telling us nothing useful there
    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        if c.dtype != b.dtype or c.shape != b.shape:
            return False
        scale = max(1.0, float(np.abs(b).max())) if b.size else 1.0
        return bool(np.allclose(c, b, rtol=rtol, atol=rtol * scale))

    return _chk


# --- 1. hist1d_uniform: direct-index binning + edge correction --------------

HIST_LO, HIST_HI = -3.0, 3.0


def hist_exact(cand, base):
    ch, ce = cand
    bh, be = base
    return (
        type(ch) is np.ndarray and type(bh) is np.ndarray
        and type(ce) is np.ndarray and type(be) is np.ndarray
        and ch.dtype == bh.dtype and ce.dtype == be.dtype
        and ch.shape == bh.shape and ce.shape == be.shape
        and bool(np.array_equal(ch, bh))
        and bool(np.array_equal(ce, be))
    )


def hist_sample(n, seed):
    return rng.uniform(HIST_LO, HIST_HI, size=n).astype(np.float64)


def hist_edge_salted(n, nbins, seed):
    # mirrors hist2d_uniform's differential-test salting (OPP-000038
    # precedent): values landed exactly on bin edges, at the two outer
    # edges, and just outside the range, to exercise the floor-index
    # edge-correction compares in _run (idx[a < edges[idx]] -=1, etc.)
    edges = np.linspace(HIST_LO, HIST_HI, nbins + 1)
    n_group = n // 4
    on_edges = edges[rng.integers(0, len(edges), size=n_group)]
    endpoints = np.array([HIST_LO, HIST_HI] * (n_group // 2))
    just_outside = np.array(
        [np.nextafter(HIST_LO, -np.inf), np.nextafter(HIST_HI, np.inf)] * (n_group // 2)
    )
    n_random = n - on_edges.size - endpoints.size - just_outside.size
    randoms = rng.uniform(HIST_LO, HIST_HI, size=max(n_random, 0))
    return np.concatenate([on_edges, endpoints, just_outside, randoms]).astype(np.float64)


HIST_NS = [1_000, 10_000] if SMOKE else [300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000]

for n in HIST_NS:
    bin_choices = (10, 50, 1_000) if n == 10_000 else (50,)
    a = hist_sample(n, seed=n)
    for nbins in bin_choices:
        suite.measure(
            case=f"hist1d_n{n}_bins{nbins}",
            params={"n": n, "bins": nbins, "window": list(SIZE_WINDOW), "bins_min": BINS_MIN},
            baseline=(
                "numpy.histogram",
                lambda a=a, nbins=nbins: np.histogram(a, bins=nbins, range=(HIST_LO, HIST_HI)),
            ),
            candidates={
                "direct_index": lambda a=a, nbins=nbins: _hist_run(
                    a, bins=nbins, range=(HIST_LO, HIST_HI)
                ),
            },
            check=hist_exact,
            samples=SAMPLES,
        )

if not SMOKE:
    n_w, nbins_w = 10_000, 50
    a_w = hist_sample(n_w, seed=999)
    w = rng.uniform(0.1, 5.0, size=n_w).astype(np.float64)
    suite.measure(
        case=f"hist1d_n{n_w}_bins{nbins_w}_weighted",
        params={"n": n_w, "bins": nbins_w, "weighted": True},
        baseline=(
            "numpy.histogram",
            lambda a=a_w, w=w: np.histogram(a, bins=nbins_w, range=(HIST_LO, HIST_HI), weights=w),
        ),
        candidates={
            "direct_index": lambda a=a_w, w=w: _hist_run(
                a, bins=nbins_w, range=(HIST_LO, HIST_HI), weights=w
            ),
        },
        check=hist_exact,
        samples=SAMPLES,
    )

    n_s, nbins_s = 10_000, 50
    a_s = hist_edge_salted(n_s, nbins_s, seed=100)
    suite.measure(
        case=f"hist1d_n{n_s}_bins{nbins_s}_edge_salted",
        params={"n": n_s, "bins": nbins_s, "edge_salted": True},
        baseline=(
            "numpy.histogram",
            lambda a=a_s: np.histogram(a, bins=nbins_s, range=(HIST_LO, HIST_HI)),
        ),
        candidates={
            "direct_index": lambda a=a_s: _hist_run(a, bins=nbins_s, range=(HIST_LO, HIST_HI)),
        },
        check=hist_exact,
        samples=SAMPLES,
    )

# --- 2. interp_uniform_grid: direct-index lerp on a uniform grid ------------

INTERP_LO, INTERP_HI = 0.0, 100.0


def interp_candidate(x, xp, fp):
    # pays the same scans _applicable does (diff-uniformity, then
    # isfinite on fp and x) without its NQ_MIN size gate, so below-floor
    # cells still measure the real guard-plus-run cost
    d = np.diff(xp)
    dx = d[0]
    uniform_ok = bool((np.abs(d - dx) <= UNIFORM_RTOL * dx).all())
    finite_ok = bool(np.isfinite(fp).all()) and bool(np.isfinite(x).all())
    assert uniform_ok and finite_ok, "guard scan refused uniform-grid witness"
    return _interp_run(x, xp, fp)


def interp_grid(m, use_arange):
    if use_arange:
        dx = (INTERP_HI - INTERP_LO) / (m - 1)
        xp = INTERP_LO + dx * np.arange(m, dtype=np.float64)
    else:
        xp = np.linspace(INTERP_LO, INTERP_HI, m)
    return xp.astype(np.float64)


INTERP_CASES = (
    [(3_000, 1_000), (100_000, 1_000)]
    if SMOKE
    else [
        (3_000, 1_000),
        (10_000, 1_000),
        (30_000, 1_000),
        (100_000, 1_000),
        (1_000_000, 10_000),
        (1_000_000, 100),
    ]
)

for nq, grid in INTERP_CASES:
    use_arange = (nq, grid) == (1_000_000, 100)
    xp = interp_grid(grid, use_arange)
    fp = (np.sin(xp) + 0.5 * xp).astype(np.float64)
    x = rng.uniform(INTERP_LO, INTERP_HI, size=nq).astype(np.float64)
    suite.measure(
        case=f"interp_nq{nq}_grid{grid}{'_arange' if use_arange else ''}",
        params={
            "nq": nq, "grid": grid, "nq_min": NQ_MIN, "uniform_rtol": UNIFORM_RTOL,
            "grid_kind": "arange" if use_arange else "linspace",
        },
        baseline=("numpy.interp", lambda x=x, xp=xp, fp=fp: np.interp(x, xp, fp)),
        candidates={
            "direct_index": lambda x=x, xp=xp, fp=fp: interp_candidate(x, xp, fp),
        },
        check=scaled_close(1e-9),
        samples=SAMPLES,
    )

# --- 3. take_index_assign: fancy-index gather + assignment -------------------


def take_arrays(n, dtype, seed):
    local = np.random.default_rng(seed)
    if dtype == np.float64:
        a = local.standard_normal(n).astype(np.float64)
    else:
        a = local.integers(-10**9, 10**9, size=n).astype(np.int64)
    indices = local.integers(0, n, size=n).astype(np.intp)
    return a, indices


def take_candidate(a, indices, out):
    out[...] = a[indices]
    return out


TAKE_SIZES = [1_000, 10_000] if SMOKE else [1_000, 3_000, 10_000, 100_000, 1_000_000, 10_000_000]

for n in TAKE_SIZES:
    a, indices = take_arrays(n, np.float64, seed=n)
    out_stock = np.empty(n, dtype=np.float64)
    out_cand = np.empty(n, dtype=np.float64)
    suite.measure(
        case=f"take_n{n}_float64",
        params={"n": n, "size_min": SIZE_MIN, "dtype": "float64"},
        baseline=(
            "numpy.take",
            lambda a=a, idx=indices, out=out_stock: np.take(a, idx, out=out),
        ),
        candidates={
            "index_assign": lambda a=a, idx=indices, out=out_cand: take_candidate(a, idx, out),
        },
        check=exact,
        samples=SAMPLES,
    )

if not SMOKE:
    n_i = 100_000
    a_i, indices_i = take_arrays(n_i, np.int64, seed=n_i + 1)
    out_stock_i = np.empty(n_i, dtype=np.int64)
    out_cand_i = np.empty(n_i, dtype=np.int64)
    suite.measure(
        case=f"take_n{n_i}_int64",
        params={"n": n_i, "size_min": SIZE_MIN, "dtype": "int64"},
        baseline=(
            "numpy.take",
            lambda a=a_i, idx=indices_i, out=out_stock_i: np.take(a, idx, out=out),
        ),
        candidates={
            "index_assign": lambda a=a_i, idx=indices_i, out=out_cand_i: take_candidate(
                a, idx, out
            ),
        },
        check=exact,
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
