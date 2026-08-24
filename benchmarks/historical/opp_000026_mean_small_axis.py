"""OPP-000026: reductions (mean/sum/add.reduce) that leave only a tiny
trailing axis, vs per-slice full reductions and einsum column sums.

numpy/numpy#8480 (gongbudaizhe, 2017, numpy 1.11.3): np.mean(img, axis=(0, 1))
on img = np.random.rand(1000, 1000, 3) took 12.5 ms, while the naive
per-slice list [np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])]
took 4.42 ms on the same array, a DERIVED 2.83x (12.5 / 4.42; no one in the
thread states a ratio). eric-wieser adds that the 2-D form
np.mean(img.reshape(-1, 3), axis=0) is "Also just as slow", that the problem
"applies to np.sum and np.add.reduce as well" (it lies in np.ufunc.reduce),
and that einsum "performs way better" (no number given). seberg's mechanism:
ufunc.reduce always prefers memory-order iteration, which wins when the kept
trailing axis is large and loses when it is tiny (tiny inner loop, so
per-iteration overhead dominates); the crossover was never measured in the
thread and is exactly what a ship predicate would need.

What this reproducer measures:

  1. The record's exact claim regime, unshrunk: C-contiguous float64
     (1000, 1000, 3) reduced with axis=(0, 1), for np.mean, np.sum, and
     np.add.reduce (eric-wieser's broadening). Each baseline call is a few
     ms, so the full named size fits the ~90s battery budget with a wide
     margin; nothing is shrunk in the non-smoke battery.
  2. A trailing-axis sweep k in {1, 2, 3, 4, 8, 16, 64, 256} at a fixed
     total of ~3e6 float64 elements (matching the thread's 1000*1000*3),
     in eric-wieser's 2-D form (rows, k) reduced along axis=0, to locate
     seberg's predicted crossover. The per-slice route is EXPECTED to lose
     at large k; demonstrating that loss is the point (it is the region a
     predicate must keep on stock numpy), so slower-than-baseline results
     at large k are findings, not failures.
  3. float32 alongside float64 on the headline mean case, and an F-order
     copy of the headline array as a control: the mechanism is
     memory-order-specific, so F-order input should behave differently and
     would likely have to fail any real predicate. For the F-order input
     the einsum candidate's reshape(-1, 3) is a copy, not a view; that
     cost is honestly included in its timing.

Candidates, both named in the thread, neither invented here:

  - "per_slice_*": the reporter's own route, one FULL reduction per kept
    slice, results assembled into an array (assembly cost included).
  - "einsum_*": eric-wieser's route, reshape(-1, k) then an einsum column
    sum ("ij->j"), divided by the row count for mean.

House-law notes: nothing here imports pyoverdrive. The per-slice candidates
do call np.mean / np.sum / np.add.reduce, but only as full reductions
(axis=None, or every axis of a 2-D slice), which keep NO trailing axis and
so fall outside the claim regime a small-trailing-axis predicate would
match; a patched dispatch would route them straight to stock, so they
cannot recurse. The einsum candidates avoid the affected operations
entirely (np.einsum is not one of them).

Correctness: mean/sum are float reductions and the reroute changes the
summation order versus stock numpy's pairwise traversal, so bit-identity is
NOT expected; the check is np.allclose with a dtype-scaled rtol (see
make_check) rather than exact equality. Result shape and dtype are
preserved by both candidates.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 8480
SMOKE = "--smoke" in sys.argv


def per_slice_mean_3d(a):
    """The reporter's exact candidate: one full mean per trailing slice.
    np.mean(slice) is an axis=None reduction (no kept trailing axis), so it
    is outside the claim regime and cannot recurse under a patched
    small-trailing-axis dispatch."""
    return np.array([np.mean(a[:, :, c]) for c in range(a.shape[2])])


def per_slice_sum_3d(a):
    return np.array([np.sum(a[:, :, c]) for c in range(a.shape[2])])


def per_slice_add_reduce_3d(a):
    return np.array([np.add.reduce(a[:, :, c], axis=None) for c in range(a.shape[2])])


def per_slice_mean_2d(a2):
    return np.array([np.mean(a2[:, j]) for j in range(a2.shape[1])])


def einsum_mean(a):
    """eric-wieser's route: reshape to (rows, k), einsum column sum, divide
    by rows. For non-C-contiguous input the reshape is a copy; that cost is
    part of the candidate and is included in its timing."""
    a2 = a.reshape(-1, a.shape[-1])
    return np.einsum("ij->j", a2) / a2.shape[0]


def einsum_sum(a):
    a2 = a.reshape(-1, a.shape[-1])
    return np.einsum("ij->j", a2)


def make_check(rtol):
    """allclose with a dtype-scaled rtol, atol=0 (values here are O(0.5)
    means or O(1e6) sums of positives, so rtol alone is the right guard).
    The candidates change summation order versus stock numpy's PAIRWISE
    summation, so bit-identity is not expected. Scale of the tolerance:
    for n ~ 3e6 uniform positives, pairwise error is ~eps*log2(n) and a
    blocked/naive order drifts like ~eps*sqrt(n), i.e. ~2e-13 relative in
    float64 (eps 2.2e-16) and ~1.2e-4 relative in float32 (eps 1.2e-7).
    rtol is set roughly 3-4 orders of magnitude above the float64 drift
    (1e-9) and one order above the float32 drift (1e-3), loose enough to
    never flag legitimate reordering, tight enough that a genuinely wrong
    reduction (wrong slice, wrong divisor: O(1) relative error) fails."""

    def check(candidate_result, baseline_result):
        return np.allclose(candidate_result, baseline_result, rtol=rtol, atol=0.0)

    return check


CHECK_F64 = make_check(1e-9)
CHECK_F32 = make_check(1e-3)

if SMOKE:
    H, W = 100, 100
    K_SWEEP = [3, 64]
    SWEEP_TOTAL = 30_000
else:
    H, W = 1000, 1000
    K_SWEEP = [1, 2, 3, 4, 8, 16, 64, 256]
    SWEEP_TOTAL = 3_000_000

samples = 3 if SMOKE else 9

suite = BenchSuite(
    "OPP-000026",
    "small trailing-axis reductions: per-slice / einsum vs ufunc.reduce",
)

rng = np.random.default_rng(SEED)

# --- 1. Headline claim regime: (H, W, 3) float64 C-order, three ops -------

img = rng.random(size=(H, W, 3))

suite.measure(
    case=f"mean_axis01_{H}x{W}x3_float64_C",
    params={"dtype": "float64", "shape": [H, W, 3], "axis": [0, 1], "order": "C"},
    baseline=("numpy.mean", lambda a=img: np.mean(a, axis=(0, 1))),
    candidates={
        "per_slice_mean": lambda a=img: per_slice_mean_3d(a),
        "einsum_mean": lambda a=img: einsum_mean(a),
    },
    check=CHECK_F64,
    samples=samples,
)

suite.measure(
    case=f"sum_axis01_{H}x{W}x3_float64_C",
    params={"dtype": "float64", "shape": [H, W, 3], "axis": [0, 1], "order": "C"},
    baseline=("numpy.sum", lambda a=img: np.sum(a, axis=(0, 1))),
    candidates={
        "per_slice_sum": lambda a=img: per_slice_sum_3d(a),
        "einsum_sum": lambda a=img: einsum_sum(a),
    },
    check=CHECK_F64,
    samples=samples,
)

suite.measure(
    case=f"add_reduce_axis01_{H}x{W}x3_float64_C",
    params={"dtype": "float64", "shape": [H, W, 3], "axis": [0, 1], "order": "C"},
    baseline=("numpy.add.reduce", lambda a=img: np.add.reduce(a, axis=(0, 1))),
    candidates={
        "per_slice_add_reduce": lambda a=img: per_slice_add_reduce_3d(a),
        "einsum_sum": lambda a=img: einsum_sum(a),
    },
    check=CHECK_F64,
    samples=samples,
)

# --- 2. float32 variant of the headline mean case -------------------------

img32 = rng.random(size=(H, W, 3), dtype=np.float32)

suite.measure(
    case=f"mean_axis01_{H}x{W}x3_float32_C",
    params={"dtype": "float32", "shape": [H, W, 3], "axis": [0, 1], "order": "C"},
    baseline=("numpy.mean", lambda a=img32: np.mean(a, axis=(0, 1))),
    candidates={
        "per_slice_mean": lambda a=img32: per_slice_mean_3d(a),
        "einsum_mean": lambda a=img32: einsum_mean(a),
    },
    check=CHECK_F32,
    samples=samples,
)

# --- 3. F-order control (mechanism is memory-order-specific) --------------

img_f = np.asfortranarray(img)

suite.measure(
    case=f"mean_axis01_{H}x{W}x3_float64_F",
    params={"dtype": "float64", "shape": [H, W, 3], "axis": [0, 1], "order": "F"},
    baseline=("numpy.mean", lambda a=img_f: np.mean(a, axis=(0, 1))),
    candidates={
        "per_slice_mean": lambda a=img_f: per_slice_mean_3d(a),
        "einsum_mean": lambda a=img_f: einsum_mean(a),
    },
    check=CHECK_F64,
    samples=samples,
)

# --- 4. Trailing-axis sweep (2-D form) to locate seberg's crossover -------

for k in K_SWEEP:
    rows = SWEEP_TOTAL // k
    a2 = rng.random(size=(rows, k))
    suite.measure(
        case=f"mean_axis0_rows{rows}_k{k}_float64_C",
        params={
            "dtype": "float64",
            "shape": [rows, k],
            "axis": 0,
            "order": "C",
            "trailing_k": k,
        },
        baseline=("numpy.mean", lambda a=a2: np.mean(a, axis=0)),
        candidates={
            "per_slice_mean": lambda a=a2: per_slice_mean_2d(a),
            "einsum_mean": lambda a=a2: einsum_mean(a),
        },
        check=CHECK_F64,
        samples=samples,
    )

if not SMOKE:
    suite.save()
