"""OPP-000018: einsum ~20x slower than other libraries (mostly fixed upstream
via optimize=True's batched-matmul path, PR #23513; the default
optimize=False path is what remains slow).

numpy/numpy#22604 reports np.einsum('thd,Thd->thT', x, y) with x, y both
shape (1000, 1, 500) float64, called with no `optimize` kwarg (numpy's
default optimize=False path), running ~20x slower than torch/jax/tf's
default einsum, which route this batched-matmul-shaped contraction through
BLAS. PR #23513 (merged) closed most of that gap, but only when the caller
passes optimize=True; optimize=False stays the default specifically to
protect low-overhead small contractions (dgasmith's np.einsum('i,i->', v, v)
example). This reproducer measures:

  - baseline: np.einsum('thd,Thd->thT', x, y) with no optimize kwarg, i.e.
    numpy's current default (naive nested-loop) path.
  - candidate "optimize_true": the same call with optimize=True -- numpy's
    own PR #23513 batched-matmul path, already shipped upstream.
  - candidate "matmul": a hand-written batched-matmul equivalent standing in
    for the transparent fast path PyOverdrive's fix (opportunity doc step 2)
    would take. Built with the `@` operator (ndarray.__matmul__) rather than
    calling numpy.matmul by name, per house rule: a fast-path candidate must
    not call a public numpy name it might itself patch some day when an
    equivalent private route exists -- `@` resolves through ndarray's C-level
    matmul slot, not through the `numpy.matmul` module attribute PyOverdrive
    would be the one patching if it ever ships this fix.

  - a small-contraction control case, np.einsum('i,i->', v, v) for v of
    length 3 and length 100, with the same three variants, to check whether
    dgasmith's small-array-overhead objection to defaulting optimize=True
    still holds on current numpy.

Sizes: the opportunity doc's step 1 asks for shapes (1000, 1, 500) and
(10000, 1, 500) (both operands) at float32 and float64. The (10000, 1, 500)
shape was only ever measured upstream WITH optimize=True (Kai-Striega's
profiling); nobody in the thread measured the naive optimize=False baseline
at that size, and its cost is O(t*T*d) -- 10000x10000x500 is ~100x the
1000x1000x500 case's baseline cost, which would blow the time budget on
repeated sampling. Per the task's "shrink the largest size... and say so"
instruction, this reproducer uses (2000, 1, 500) as the "large" shape
instead (4x the base case's compute, not 100x) and samples it 5x instead of
7x, to keep the whole non-smoke battery under ~90s while still showing how
the gap scales with size.

Correctness: both candidates reorder the summation (BLAS reduction instead
of the naive C accumulation loop), so results are compared with
np.allclose, not bit-identity, per the opportunity doc's own note that BLAS-
routed einsum changes floating-point summation order. Tolerance is widened
for float32: with d=500 (or d=100) terms summed per output element and
inputs drawn uniform in [0, 1), sqrt(d)*eps accumulation for float32
(eps ~1.19e-7) works out to a relative error on the order of 1e-6 to 1e-5
against O(1)-to-O(100)-magnitude sums, so rtol=1e-3/atol=1e-3 is a
comfortable, justified margin; float64's eps (~2.22e-16) leaves the same
accumulation many orders of magnitude below numpy's own allclose defaults,
so a tighter but still non-bit-identical rtol=1e-7/atol=1e-9 is used there.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 22604
SMOKE = "--smoke" in sys.argv


def matmul_thd_Thd_thT(x, y):
    """Batched-matmul equivalent of einsum('thd,Thd->thT', x, y).

    x: (t, h, d), y: (T, h, d) -> out: (t, h, T), contracting over d and
    batching over h. Uses `@` (ndarray.__matmul__) rather than numpy.matmul
    by name -- see module docstring.
    """
    xb = x.transpose(1, 0, 2)  # (h, t, d)
    yb = y.transpose(1, 2, 0)  # (h, d, T)
    out = xb @ yb  # (h, t, T)
    return out.transpose(1, 0, 2)  # (t, h, T)


def close_enough(expected, actual):
    """allclose, not bit-identity -- both candidates reorder summation
    (BLAS reduction vs naive C loop). Tolerance justified in module
    docstring."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.dtype == np.float32 or actual.dtype == np.float32:
        return np.allclose(expected, actual, rtol=1e-3, atol=1e-3)
    return np.allclose(expected, actual, rtol=1e-7, atol=1e-9)


def make_operands(rng, t, h, d, dtype):
    x = rng.uniform(size=(t, h, d)).astype(dtype)
    y = rng.uniform(size=(t, h, d)).astype(dtype)
    return x, y


if SMOKE:
    main_shapes = [(50, 1, 20)]
    small_lengths = [3]
    samples_main = 3
    samples_small = 3
else:
    main_shapes = [(1000, 1, 500), (2000, 1, 500)]
    small_lengths = [3, 100]
    samples_small = 11

dtypes = [(np.float64, "float64"), (np.float32, "float32")]

suite = BenchSuite(
    "OPP-000018",
    "einsum default (optimize=False) vs optimize=True and a hand-written "
    "batched-matmul equivalent, for a batched-matmul-shaped contraction plus "
    "a small-contraction control case",
)

rng = np.random.default_rng(SEED)

for shape_idx, (t, h, d) in enumerate(main_shapes):
    # Larger shape gets fewer samples to keep the battery under budget; see
    # module docstring.
    samples = samples_main if SMOKE else (7 if shape_idx == 0 else 5)
    for dtype, dtype_name in dtypes:
        x, y = make_operands(rng, t, h, d, dtype)
        case = f"einsum_thd_Thd_thT_t{t}_h{h}_d{d}_{dtype_name}"
        params = {"dtype": dtype_name, "t": t, "h": h, "d": d}

        suite.measure(
            case=case,
            params=params,
            baseline=(
                "numpy.einsum(optimize=False)",
                lambda x=x, y=y: np.einsum("thd,Thd->thT", x, y),
            ),
            candidates={
                "optimize_true": lambda x=x, y=y: np.einsum(
                    "thd,Thd->thT", x, y, optimize=True
                ),
                "matmul": lambda x=x, y=y: matmul_thd_Thd_thT(x, y),
            },
            check=close_enough,
            samples=samples,
        )

for n in small_lengths:
    v = rng.uniform(size=n).astype(np.float64)
    case = f"einsum_dot_len{n}"
    params = {"dtype": "float64", "len": n}

    suite.measure(
        case=case,
        params=params,
        baseline=(
            "numpy.einsum(optimize=False)",
            lambda v=v: np.einsum("i,i->", v, v),
        ),
        candidates={
            "optimize_true": lambda v=v: np.einsum("i,i->", v, v, optimize=True),
            "matmul": lambda v=v: v @ v,
        },
        check=close_enough,
        samples=samples_small,
    )

if not SMOKE:
    suite.save()
