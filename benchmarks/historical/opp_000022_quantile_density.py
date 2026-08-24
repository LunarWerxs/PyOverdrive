"""OPP-000022: dense multi-kth partition density cliff, np.quantile /
ndarray.partition vs one full sort.

numpy/numpy#32187 reports that ndarray.partition (and therefore
np.quantile, which partitions on up to 2*len(q) neighbor indices for
method="linear") hits a sharp density cliff once consecutive kth values are
less than 4 apart: introselect's dumb_select fast path degrades the whole
partition to O(n^2) per lane. eendebakpt's confirming measurement (the one
concrete number in the thread): kth spacing 2 on a (2048, 300) float64
array partitioned along the length-2048 axis takes 496 ms against an
implied ~23.6 ms full sort, i.e. 21x, on unpatched numpy main (scalar
build). An upstream branch fix exists (eendebakpt's fix/partition-dense-kth,
no PR linked at ingestion), so the numpy actually installed here may sit on
either side of it; if the cliff is absent, the honest finding is the
residual sort-vs-partition ratio (~1.7x on his machine), and the record's
21x claim must be re-based, possibly to not_reproduced.

What this reproducer measures (the record's "What the Dyno reproducer must
measure" points 1-5):

  1. Partition-level cliff: stock np.partition(a, kth, axis=0) vs one full
     np.sort(a, axis=0) on (2048, 300) float64, at kth spacings
     {20, 10, 4, 3, 2, 1} (densities 0.05 to 1.00). The reporter's part 2
     grid ran densities 0.05 through 1.00; it is thinned here to six
     spacings that bracket the claimed threshold, spacing 4 (last point
     outside the claimed cliff), 3 (first point inside), 2 (the 21x claim
     regime), and 1 (densest), because intermediate densities add wall
     clock without adding threshold information.
  2. Quantile-level ratio table the thread never published:
     np.quantile(a, q, axis=-1, method="linear") vs a sort-plus-exact-lerp
     candidate on (300, 2048) float64, sweeping nq/m over the reporter's
     part 1 grid {0.05, 0.10, 0.20, 0.24, 0.25, 0.30, 0.50} with
     q = np.linspace(0, 1, nq).
  3. Correctness on NaN-bearing and duplicate-heavy inputs (two extra
     quantile cases at nq/m = 0.25; the NaN case salts half the slices so
     both the NaN-propagating and clean output branches are checked).
  4. Scaling of the spacing-2 regime: rows m in {1024, 4096, 8192} x 300
     cols (2048 is already covered by the spacing-2 case in point 1), to
     confirm or date the thread's quadratic 47 -> 2672 ms report.
  5. Threshold calibration falls out of points 1 and 2: the measured
     spacing-4 vs spacing-3 pair and the nq/m table locate the crossover
     on this hardware, which is what a dispatch predicate would use.

Sizes and dtypes are the record's own, (2048, 300) and (300, 2048)
float64, unshrunk. The only budget concessions: the density grid thinning
described above, and samples lowered to 3 (from 5) for the two largest
scaling cases (4096 and 8192 rows, up to ~2.7 s per baseline call on
unpatched numpy per the thread) and for the two point-3 special cases
(whose purpose is correctness checking, not timing resolution, so extra
samples buy nothing). At the thread's own unpatched timings the
full battery lands around 50-60 s; on a fixed numpy it is a few seconds.
Either way it stays under the ~90 s house budget.

Baselines are the stock calls exactly as a user writes them. np.partition
(copying) is used rather than in-place ndarray.partition so every sample
sees the identical un-partitioned input, and so baseline and candidate each
include exactly one array copy, keeping the comparison symmetric; the
issue's part 2 measured the same operation in-place.

Candidates (neither imports pyoverdrive; neither calls np.quantile,
np.partition, or np.argpartition, the ops this record covers, so nothing
here can recurse if those ops are patched):

  - "full_sort" (partition level): one np.sort along the axis. A fully
    sorted array satisfies partition's documented guarantee; the check
    compares the two outputs only at the kth rows, where both must hold
    the exact order statistics, so the comparison is exact equality, not
    a tolerance. Bytes elsewhere legitimately differ (the record's own
    reason to keep any real fast path at the quantile level).
  - "sort_exact_lerp" (quantile level): one np.sort along the axis, then a
    replication of numpy's _quantile method="linear" arithmetic, the
    (n - 1) * q virtual index, the -1 substitution for above-bound
    indices, gamma taken against the substituted integer index, and the
    _lerp gamma >= 0.5 stability rewrite, plus NaN propagation via
    isnan(sorted[-1]). Because the arithmetic is replicated operation for
    operation, the correctness target is bit-identical output
    (np.array_equal, equal_nan=True only for the NaN case), not a float
    tolerance. This is the route the record's notes name (the OPP-000013
    shipped-path shape, and the reporter's own manual_quantile baseline
    plus exact-lerp discipline), not an invention.

Not measured here: np.argpartition (the thread says it shows the same
cliff; it adds no information to the routing decision), non-linear method=
variants, weights=, and tuple axis (semantics hazards listed in the record
for any real dispatch, but outside the measured claim regime).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 32187
SMOKE = "--smoke" in sys.argv


def full_sort(a):
    """Partition-level candidate: one full sort along axis 0."""
    return np.sort(a, axis=0)


def sort_lerp_quantile(a, q, axis):
    """Quantile-level candidate: full sort along the axis, then numpy's own
    method="linear" interpolation arithmetic replicated exactly (see module
    docstring). q must be a 1-D float64 array."""
    moved = np.moveaxis(a, axis, 0)
    n = moved.shape[0]
    srt = np.sort(moved, axis=0)
    virt = (n - 1) * q  # numpy's get_virtual_index for method="linear"
    prev_i = np.floor(virt).astype(np.intp)
    next_i = prev_i + 1
    above = virt >= n - 1
    # numpy substitutes -1 (the last element) for above-bound indices and
    # computes gamma against the substituted index; replicate both.
    prev_i[above] = -1
    next_i[above] = -1
    gamma = (virt - prev_i).reshape(virt.shape + (1,) * (srt.ndim - 1))
    below_v = srt[prev_i]
    above_v = srt[next_i]
    diff = above_v - below_v
    # numpy's _lerp: a + diff*t, rewritten as b - diff*(1-t) where t >= 0.5
    # for numerical stability. np.where selects between the two expressions
    # computed with the identical operations, so the bits match.
    result = np.where(
        gamma >= 0.5,
        above_v - diff * (1 - gamma),
        below_v + diff * gamma,
    )
    # NaN propagation: after a sort, any slice containing NaN has NaN last.
    has_nan = np.isnan(srt[-1, ...])
    if has_nan.any():
        result[..., has_nan] = np.nan
    return result


def make_kth_check(kth):
    """Exact equality at the kth rows only: both partition and a full sort
    must place the exact order statistics there; bytes elsewhere are
    unspecified for partition and legitimately differ."""

    def check(cand, base):
        return np.array_equal(cand[kth, :], base[kth, :])

    return check


def exact_equal(cand, base):
    return np.array_equal(cand, base)


def exact_equal_nan(cand, base):
    # equal_nan only for the NaN-salted case: stock np.quantile propagates
    # NaN for NaN-bearing slices and the candidate must match those too.
    return np.array_equal(cand, base, equal_nan=True)


if SMOKE:
    PART_SHAPE = (256, 64)
    SPACINGS = [2]
    QUANT_SHAPE = (64, 256)
    NQ_RATIOS = [0.25]
    SCALING_ROWS = []
    SPECIALS = ["nan"]
    SAMPLES_MAIN = 3
    SAMPLES_HEAVY = 2
else:
    PART_SHAPE = (2048, 300)
    SPACINGS = [20, 10, 4, 3, 2, 1]
    QUANT_SHAPE = (300, 2048)
    NQ_RATIOS = [0.05, 0.10, 0.20, 0.24, 0.25, 0.30, 0.50]
    SCALING_ROWS = [1024, 4096, 8192]
    SPECIALS = ["nan", "dups"]
    SAMPLES_MAIN = 5
    SAMPLES_HEAVY = 3

suite = BenchSuite(
    "OPP-000022",
    "dense multi-kth partition/quantile density cliff vs one full sort",
)

rng = np.random.default_rng(SEED)

# --- Part 1 (+5): partition vs full sort across kth densities -----------
part_rows, part_cols = PART_SHAPE
for spacing in SPACINGS:
    a = rng.standard_normal(PART_SHAPE)
    kth = np.arange(0, part_rows, spacing)
    suite.measure(
        case=f"partition_{part_rows}x{part_cols}_spacing{spacing}",
        params={
            "dtype": "float64",
            "shape": list(PART_SHAPE),
            "axis": 0,
            "kth_spacing": spacing,
            "len_kth": int(kth.size),
            "kth_density": round(kth.size / part_rows, 4),
        },
        baseline=(
            "numpy.partition",
            lambda a=a, kth=kth: np.partition(a, kth, axis=0),
        ),
        candidates={"full_sort": lambda a=a: full_sort(a)},
        check=make_kth_check(kth),
        samples=SAMPLES_MAIN,
    )

# --- Part 4: spacing-2 scaling with row count ---------------------------
for m in SCALING_ROWS:
    a = rng.standard_normal((m, part_cols))
    kth = np.arange(0, m, 2)
    suite.measure(
        case=f"partition_scaling_{m}x{part_cols}_spacing2",
        params={
            "dtype": "float64",
            "shape": [m, part_cols],
            "axis": 0,
            "kth_spacing": 2,
            "len_kth": int(kth.size),
        },
        baseline=(
            "numpy.partition",
            lambda a=a, kth=kth: np.partition(a, kth, axis=0),
        ),
        candidates={"full_sort": lambda a=a: full_sort(a)},
        check=make_kth_check(kth),
        samples=SAMPLES_HEAVY if m > 2048 else SAMPLES_MAIN,
    )

# --- Part 2: quantile-level ratio table ---------------------------------
qm_rows, qm_cols = QUANT_SHAPE
a_q = rng.standard_normal(QUANT_SHAPE)
for ratio in NQ_RATIOS:
    nq = int(round(ratio * qm_cols))
    q = np.linspace(0.0, 1.0, nq)
    suite.measure(
        case=f"quantile_{qm_rows}x{qm_cols}_nq{nq}",
        params={
            "dtype": "float64",
            "shape": list(QUANT_SHAPE),
            "axis": -1,
            "method": "linear",
            "nq": nq,
            "nq_over_m": ratio,
        },
        baseline=(
            "numpy.quantile",
            lambda a=a_q, q=q: np.quantile(a, q, axis=-1, method="linear"),
        ),
        candidates={
            "sort_exact_lerp": lambda a=a_q, q=q: sort_lerp_quantile(a, q, -1)
        },
        check=exact_equal,
        samples=SAMPLES_MAIN,
    )

# --- Part 3: NaN-bearing and duplicate-heavy correctness cases ----------
for special in SPECIALS:
    nq = int(round(0.25 * qm_cols))
    q = np.linspace(0.0, 1.0, nq)
    if special == "nan":
        a_s = rng.standard_normal(QUANT_SHAPE)
        # Salt exactly half the slices (rows; axis=-1 reduces each row) with
        # one NaN each, so both output branches get checked.
        nan_rows = rng.choice(qm_rows, size=qm_rows // 2, replace=False)
        a_s[nan_rows, rng.integers(0, qm_cols, size=nan_rows.size)] = np.nan
        check = exact_equal_nan
    else:
        # Duplicate-heavy: 8 distinct values across the whole array.
        a_s = rng.integers(0, 8, size=QUANT_SHAPE).astype(np.float64)
        check = exact_equal
    suite.measure(
        case=f"quantile_{special}_{qm_rows}x{qm_cols}_nq{nq}",
        params={
            "dtype": "float64",
            "shape": list(QUANT_SHAPE),
            "axis": -1,
            "method": "linear",
            "nq": nq,
            "nq_over_m": 0.25,
            "input": special,
        },
        baseline=(
            "numpy.quantile",
            lambda a=a_s, q=q: np.quantile(a, q, axis=-1, method="linear"),
        ),
        candidates={
            "sort_exact_lerp": lambda a=a_s, q=q: sort_lerp_quantile(a, q, -1)
        },
        check=check,
        samples=SAMPLES_HEAVY,
    )

if not SMOKE:
    suite.save()
