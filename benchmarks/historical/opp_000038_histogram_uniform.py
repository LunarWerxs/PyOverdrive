"""OPP-000038: histogram2d/histogramdd with uniform-by-construction bins,
direct index computation vs per-dimension searchsorted.

numpy/numpy#17676 (aminnj, 2020): histogramdd builds explicit edge
arrays and searchsorts every sample per dimension even when bins are
uniform by construction (bins given as ints plus an explicit range);
computing bin indices directly is 4-5x faster in the reporter's use
case (their working numpy-fork patch "passes unit testing"; numpy's own
1-D histogram already takes exactly this shortcut internally, which is
the precedent). Candidate here is reimplemented from first principles.

THE correctness minefield (why the naive floor formula is not enough):
values exactly ON an interior edge must land in the same bin stock puts
them in, and stock's bins are the ROUNDED floats of linspace, while
(x - lo) * n / (hi - lo) rounds differently. The candidate therefore
does what numpy's 1-D internal path does: compute the floor index, then
CORRECT it against the actual edge array (one vectorized compare each
way), which keeps everything O(n) with no per-sample binary search but
is bit-exact against the same edges. The rightmost edge is inclusive;
out-of-range samples are dropped. Counts are integers: the check is
EXACT equality on every returned array including the edge arrays.

Cases: the reporter's own 5e6x2 gaussian at 100x100 bins; bin-count
sweep {10, 100, 1000} per axis at 1e6 samples; a 3-D histogramdd case;
an EDGE-SALTED case where a large fraction of samples sit exactly on
interior edges and on the two outermost edges (the decisive cell: if
the correction logic is wrong, this check fails); a weights= case
(weights ride through bincount unchanged).

House rules: never imports pyoverdrive. The candidate uses
clip/floor/bincount/linspace, none of which carry a registered fast
path for these shapes; a patched dispatch could not recurse.

Result JSON: benchmarks/results/OPP-000038/.
Run: .venv/Scripts/python benchmarks/historical/opp_000038_histogram_uniform.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 17676
SMOKE = "--smoke" in sys.argv


def _uniform_indices(x, lo, hi, nbins, edges):
    """Bin index per sample, bit-exact against stock's edge semantics:
    floor formula first, then correct against the true (rounded) edges,
    then the inclusive right edge, then an out-of-range mask."""
    idx = np.floor((x - lo) * (nbins / (hi - lo))).astype(np.intp)
    np.clip(idx, 0, nbins - 1, out=idx)
    # correction against the real edges (numpy's own 1-D internal trick):
    # a sample below its bin's left edge moves down; one at/above its
    # bin's right edge moves up
    idx[x < edges[idx]] -= 1
    idx2 = idx + 1
    np.clip(idx2, 0, nbins, out=idx2)
    idx[x >= edges[idx2]] += 1
    np.clip(idx, 0, nbins - 1, out=idx)
    # rightmost edge inclusive; anything outside [lo, hi] is dropped
    keep = (x >= edges[0]) & (x <= edges[-1])
    return idx, keep


def hist2d_uniform(x, y, bins, range_, weights=None):
    nx, ny = bins
    (xlo, xhi), (ylo, yhi) = range_
    ex = np.linspace(xlo, xhi, nx + 1)
    ey = np.linspace(ylo, yhi, ny + 1)
    ix, keepx = _uniform_indices(x, xlo, xhi, nx, ex)
    iy, keepy = _uniform_indices(y, ylo, yhi, ny, ey)
    keep = keepx & keepy
    flat = ix[keep] * ny + iy[keep]
    w = None if weights is None else weights[keep]
    h = np.bincount(flat, weights=w, minlength=nx * ny).reshape(nx, ny)
    if weights is None:
        h = h.astype(np.float64)
    return h, ex, ey


def histdd_uniform(sample, bins, range_):
    n, d = sample.shape
    edges = [np.linspace(lo, hi, b + 1) for b, (lo, hi) in zip(bins, range_)]
    keep = np.ones(n, dtype=bool)
    idxs = []
    for j in range(d):
        ij, kj = _uniform_indices(sample[:, j], range_[j][0], range_[j][1], bins[j], edges[j])
        idxs.append(ij)
        keep &= kj
    flat = np.zeros(keep.sum(), dtype=np.intp)
    for j in range(d):
        flat = flat * bins[j] + idxs[j][keep]
    h = np.bincount(flat, minlength=int(np.prod(bins))).reshape(bins).astype(np.float64)
    return h, edges


def check2d(cand, base):
    hc, exc, eyc = cand
    hb, exb, eyb = base
    return (
        hc.dtype == hb.dtype
        and np.array_equal(hc, hb)
        and np.array_equal(exc, exb)
        and np.array_equal(eyc, eyb)
    )


def checkdd(cand, base):
    hc, ec = cand
    hb, eb = base
    return (
        hc.dtype == hb.dtype
        and np.array_equal(hc, hb)
        and len(ec) == len(eb)
        and all(np.array_equal(c, b) for c, b in zip(ec, eb))
    )


suite = BenchSuite("OPP-000038", "uniform-bin histogram2d/dd: direct index vs searchsorted")
rng = np.random.default_rng(SEED)

N = 100_000 if SMOKE else 5_000_000
SAMPLES = 3 if SMOKE else 7
xy = rng.normal(0.0, 1.0, size=(N, 2))
x, y = xy[:, 0].copy(), xy[:, 1].copy()

suite.measure(
    case=f"hist2d_n{N}_bins100x100_gaussian",
    params={"n": N, "bins": [100, 100], "provenance": "reporter's own case"},
    baseline=(
        "numpy.histogram2d",
        lambda x=x, y=y: np.histogram2d(x, y, bins=[100, 100], range=[[-3, 3], [-3, 3]]),
    ),
    candidates={
        "direct_index": lambda x=x, y=y: hist2d_uniform(
            x, y, (100, 100), ((-3.0, 3.0), (-3.0, 3.0))
        )
    },
    check=check2d,
    samples=SAMPLES,
)

if not SMOKE:
    xm = rng.normal(0.0, 1.0, size=1_000_000)
    ym = rng.normal(0.0, 1.0, size=1_000_000)
    for b in (10, 1000):
        suite.measure(
            case=f"hist2d_n1000000_bins{b}x{b}",
            params={"n": 1_000_000, "bins": [b, b]},
            baseline=(
                "numpy.histogram2d",
                lambda x=xm, y=ym, b=b: np.histogram2d(x, y, bins=[b, b], range=[[-3, 3], [-3, 3]]),
            ),
            candidates={
                "direct_index": lambda x=xm, y=ym, b=b: hist2d_uniform(
                    x, y, (b, b), ((-3.0, 3.0), (-3.0, 3.0))
                )
            },
            check=check2d,
            samples=7,
        )

    # THE decisive cell: half the samples sit exactly ON edges (interior,
    # leftmost, rightmost, and just-outside values included)
    edges_probe = np.linspace(-3.0, 3.0, 101)
    exact_on = edges_probe[rng.integers(0, 101, size=500_000)]
    fill = rng.normal(0.0, 1.0, size=500_000)
    xe = np.concatenate([exact_on, fill, np.array([-3.0, 3.0, -3.0000001, 3.0000001])])
    ye = np.concatenate([fill, exact_on, np.array([3.0, -3.0, 0.0, 0.0])])
    suite.measure(
        case="hist2d_edge_salted_bitexactness",
        params={"n": int(xe.size), "bins": [100, 100], "probe": "values exactly on edges"},
        baseline=(
            "numpy.histogram2d",
            lambda x=xe, y=ye: np.histogram2d(x, y, bins=[100, 100], range=[[-3, 3], [-3, 3]]),
        ),
        candidates={
            "direct_index": lambda x=xe, y=ye: hist2d_uniform(
                x, y, (100, 100), ((-3.0, 3.0), (-3.0, 3.0))
            )
        },
        check=check2d,
        samples=7,
    )

    # weights ride through bincount
    w = rng.random(1_000_000)
    suite.measure(
        case="hist2d_n1000000_weighted",
        params={"n": 1_000_000, "bins": [100, 100], "weights": True},
        baseline=(
            "numpy.histogram2d",
            lambda x=xm, y=ym, w=w: np.histogram2d(
                x, y, bins=[100, 100], range=[[-3, 3], [-3, 3]], weights=w
            ),
        ),
        candidates={
            "direct_index": lambda x=xm, y=ym, w=w: hist2d_uniform(
                x, y, (100, 100), ((-3.0, 3.0), (-3.0, 3.0)), weights=w
            )
        },
        check=check2d,
        samples=7,
    )

    # 3-D histogramdd
    s3 = rng.normal(0.0, 1.0, size=(1_000_000, 3))
    suite.measure(
        case="histdd_n1000000_bins20x20x20",
        params={"n": 1_000_000, "bins": [20, 20, 20], "d": 3},
        baseline=(
            "numpy.histogramdd",
            lambda s=s3: np.histogramdd(s, bins=[20, 20, 20], range=[[-3, 3]] * 3),
        ),
        candidates={
            "direct_index": lambda s=s3: histdd_uniform(s, (20, 20, 20), ((-3.0, 3.0),) * 3)
        },
        check=checkdd,
        samples=7,
    )
    suite.save()
