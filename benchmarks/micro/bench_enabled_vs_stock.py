"""End-to-end proof: pyoverdrive.enable() vs stock NumPy through the public API.

Two things at once:
1. The wins must survive the full stack (patch + Gearbox dispatch + predicate),
   not just exist in isolated candidate functions.
2. The protected baseline: regimes that must NOT regress beyond dispatch tax,
   i.e. small arrays, excluded dtypes, kwargs routes, 1-D/2-D inner (spec: no
   more than 2 percent credible regression on unaffected cases; tiny-array
   calls pay the measured few-hundred-ns dispatch tax, which the small-n cases
   here quantify honestly).

Run: .venv/Scripts/python benchmarks/micro/bench_enabled_vs_stock.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

import pyoverdrive

SMOKE = "--smoke" in sys.argv

from pyoverdrive.fastpaths.parallel_binary import SUPPORTED as BINARY_SUPPORTED
from pyoverdrive.fastpaths.parallel_ufunc import SUPPORTED as PYRALLEL_SUPPORTED

STOCK_UNIQUE = np.unique
STOCK_INNER = np.inner
STOCK_INTERSECT = np.intersect1d
STOCK_ASCONTIG = np.ascontiguousarray
STOCK_CONVOLVE = np.convolve
STOCK_CORRELATE = np.correlate
STOCK_NANQUANTILE = np.nanquantile
STOCK_EINSUM = np.einsum
STOCK_SEARCHSORTED = np.searchsorted
STOCK_ISCLOSE = np.isclose
STOCK_ISIN = np.isin
STOCK_QUANTILE = np.quantile
STOCK_DOT = np.dot
STOCK_SORT = np.sort
STOCK_MEAN = np.mean
STOCK_SUM = np.sum
STOCK_ROLL = np.roll
STOCK_ARGMAX = np.argmax
STOCK_MATMUL = np.matmul
STOCK_EIGVALSH = np.linalg.eigvalsh
STOCK_INV = np.linalg.inv
STOCK_MEDIAN = np.median
STOCK_HIST2D = np.histogram2d
STOCK_PERCENTILE = np.percentile
STOCK_NANMEAN = np.nanmean
STOCK_NANSUM = np.nansum
STOCK_NANSTD = np.nanstd
STOCK_NANARGMAX = np.nanargmax
STOCK_NANMEDIAN = np.nanmedian
STOCK_NANPERCENTILE = np.nanpercentile
STOCK_DET = np.linalg.det
STOCK_SLOGDET = np.linalg.slogdet
STOCK_SOLVE = np.linalg.solve
STOCK_NAN_TO_NUM = np.nan_to_num
STOCK_CHOLESKY = np.linalg.cholesky
STOCK_INTERP = np.interp
STOCK_TAKE = np.take
STOCK_QR = np.linalg.qr
STOCK_APPLY_ALONG_AXIS = np.apply_along_axis
STOCK_VECTORIZE = np.vectorize
STOCK_PINV = np.linalg.pinv
STOCK_NORM = np.linalg.norm
STOCK_SVD = np.linalg.svd
STOCK_UFUNCS = {op: getattr(np, op) for op in PYRALLEL_SUPPORTED}
STOCK_BINARY = {op: getattr(np, op) for op in BINARY_SUPPORTED}

pyoverdrive.enable(
    ["numpy.unique", "numpy.unique_values", "numpy.inner", "numpy.intersect1d",
     "numpy.ascontiguousarray", "numpy.convolve", "numpy.correlate",
     "numpy.nanquantile", "numpy.einsum", "numpy.searchsorted", "numpy.isclose",
     "numpy.isin", "numpy.dot", "numpy.quantile", "numpy.sort", "numpy.mean",
     "numpy.sum", "numpy.roll", "numpy.argmax", "numpy.matmul",
     "numpy.linalg.eigvalsh", "numpy.linalg.inv", "numpy.median",
     "numpy.histogram2d", "numpy.percentile", "numpy.nanmean", "numpy.nansum",
     "numpy.nanstd", "numpy.nanvar", "numpy.nanargmax", "numpy.nanargmin",
     "numpy.nanmedian", "numpy.nanpercentile", "numpy.linalg.det",
     "numpy.linalg.slogdet", "numpy.linalg.solve", "numpy.nan_to_num",
     "numpy.linalg.cholesky", "numpy.interp", "numpy.take", "numpy.linalg.qr",
     "numpy.apply_along_axis", "numpy.vectorize",
     "numpy.linalg.pinv", "numpy.linalg.norm", "numpy.linalg.svd"]
    + [f"numpy.{op}" for op in PYRALLEL_SUPPORTED]
    + [f"numpy.{op}" for op in BINARY_SUPPORTED]
)

suite = BenchSuite("MVP-BASELINE", "pyoverdrive.enable() vs stock, public API")
rng = np.random.default_rng(20260823)

N_BIG = 10_000 if SMOKE else 1_000_000
samples_big = 3 if SMOKE else 9

# --- unique: the win, through the public API --------------------------------
big_i64 = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max,
                       size=N_BIG, dtype=np.int64)
suite.measure(
    case=f"unique_int64_n{N_BIG}_highcard",
    params={"dtype": "int64", "n": N_BIG, "expect": "big win"},
    baseline=("stock_unique", lambda d=big_i64: STOCK_UNIQUE(d)),
    candidates={"pyoverdrive": lambda d=big_i64: np.unique(d)},
    check=np.array_equal,
    samples=samples_big,
)

# --- small-size win region (n >= 64 dispatches after calibration) -----------
small_i64 = rng.integers(0, 100, size=300, dtype=np.int64)
suite.measure(
    case="unique_int64_n300_small_dispatch",
    params={"dtype": "int64", "n": 300, "expect": "modest win"},
    baseline=("stock_unique", lambda d=small_i64: STOCK_UNIQUE(d)),
    candidates={"pyoverdrive": lambda d=small_i64: np.unique(d)},
    check=np.array_equal,
    samples=3 if SMOKE else 15,
)

# --- protected baseline: must ride within dispatch tax of stock -------------
tiny_i64 = rng.integers(0, 20, size=32, dtype=np.int64)
suite.measure(
    case="unique_int64_n32_below_threshold",
    params={"dtype": "int64", "n": 32, "expect": "fallback, tax only"},
    baseline=("stock_unique", lambda d=tiny_i64: STOCK_UNIQUE(d)),
    candidates={"pyoverdrive": lambda d=tiny_i64: np.unique(d)},
    check=np.array_equal,
    samples=3 if SMOKE else 15,
)

big_f64 = rng.random(N_BIG)
suite.measure(
    case=f"unique_float64_n{N_BIG}_excluded_dtype",
    params={"dtype": "float64", "n": N_BIG, "expect": "fallback, tax only"},
    baseline=("stock_unique", lambda d=big_f64: STOCK_UNIQUE(d)),
    candidates={"pyoverdrive": lambda d=big_f64: np.unique(d)},
    check=np.array_equal,
    samples=samples_big,
)

kw_i64 = rng.integers(0, 1000, size=N_BIG, dtype=np.int64)
suite.measure(
    case=f"unique_return_counts_n{N_BIG}_kwargs_route",
    params={"dtype": "int64", "n": N_BIG, "expect": "fallback, tax only"},
    baseline=("stock_unique", lambda d=kw_i64: STOCK_UNIQUE(d, return_counts=True)),
    candidates={"pyoverdrive": lambda d=kw_i64: np.unique(d, return_counts=True)},
    check=lambda c, b: all(np.array_equal(x, y) for x, y in zip(c, b)),
    samples=samples_big,
)

# --- unique axis=0 single column: the win and its guard ---------------------
from pyoverdrive.fastpaths.unique_axis0_column import SIZE_THRESHOLD as AXIS0_THRESHOLD

col = rng.integers(0, 2000, size=(10_000 if SMOKE else 100_000, 1), dtype=np.int64)
suite.measure(
    case=f"unique_axis0_int64_n{col.shape[0]}_single_column",
    params={"dtype": "int64", "n": col.shape[0], "expect": "big win"},
    baseline=("stock_unique", lambda d=col: STOCK_UNIQUE(d, axis=0)),
    candidates={"pyoverdrive": lambda d=col: np.unique(d, axis=0)},
    check=np.array_equal,
    samples=3 if SMOKE else 7,
)
two_col = rng.integers(0, 2000, size=(10_000 if SMOKE else 100_000, 2), dtype=np.int64)
suite.measure(
    case=f"unique_axis0_int64_n{two_col.shape[0]}_two_columns_guard",
    params={"dtype": "int64", "n": two_col.shape[0], "expect": "fallback, tax only"},
    baseline=("stock_unique", lambda d=two_col: STOCK_UNIQUE(d, axis=0)),
    candidates={"pyoverdrive": lambda d=two_col: np.unique(d, axis=0)},
    check=np.array_equal,
    samples=3 if SMOKE else 7,
)

# --- small-int radix routes (OPP-000010) ------------------------------------
si_n = 5_000 if SMOKE else 100_000
si16 = rng.integers(-32768, 32768, size=si_n, dtype=np.int16)
suite.measure(
    case=f"unique_int16_n{si_n}_highcard_radix",
    params={"dtype": "int16", "n": si_n, "expect": "big win via kind='stable'"},
    baseline=("stock_unique", lambda d=si16: STOCK_UNIQUE(d)),
    candidates={"pyoverdrive": lambda d=si16: np.unique(d)},
    check=np.array_equal,
    samples=3 if SMOKE else 9,
)
siu = rng.integers(0, 65536, size=si_n, dtype=np.uint16)
siv = rng.integers(0, 65536, size=max(si_n // 5, 100), dtype=np.uint16)
suite.measure(
    case=f"intersect1d_uint16_n{si_n}_radix",
    params={"dtype": "uint16", "n": si_n, "expect": "win via kind='stable'"},
    baseline=("stock_intersect1d", lambda a=siu, b=siv: STOCK_INTERSECT(a, b)),
    candidates={"pyoverdrive": lambda a=siu, b=siv: np.intersect1d(a, b)},
    check=np.array_equal,
    samples=3 if SMOKE else 9,
)

# --- inner: the win and its guards ------------------------------------------
if SMOKE:
    a3 = rng.random((5, 5, 32)); b3 = rng.random((100, 32))
else:
    a3 = rng.random((25, 25, 500)); b3 = rng.random((10_000, 500))
suite.measure(
    case="inner_3d_reported_shape",
    params={"shape_a": list(a3.shape), "shape_b": list(b3.shape),
            "expect": "big win"},
    baseline=("stock_inner", lambda a=a3, b=b3: STOCK_INNER(a, b)),
    candidates={"pyoverdrive": lambda a=a3, b=b3: np.inner(a, b)},
    check=lambda c, b: np.allclose(c, b, rtol=1e-9, atol=1e-12),
    samples=3 if SMOKE else 5,
)

a2 = rng.random((100, 100) if SMOKE else (500, 500))
b2 = rng.random(a2.shape)
suite.measure(
    case="inner_2d_guard",
    params={"shape": list(a2.shape), "expect": "no dispatch, tax only"},
    baseline=("stock_inner", lambda a=a2, b=b2: STOCK_INNER(a, b)),
    candidates={"pyoverdrive": lambda a=a2, b=b2: np.inner(a, b)},
    check=np.array_equal,
    samples=3 if SMOKE else 11,
)

a1 = rng.random(1000)
b1 = rng.random(1000)
suite.measure(
    case="inner_1d_guard_microsecond_scale",
    params={"n": 1000, "expect": "no dispatch, tax visible at ns scale"},
    baseline=("stock_inner", lambda a=a1, b=b1: STOCK_INNER(a, b)),
    candidates={"pyoverdrive": lambda a=a1, b=b1: np.inner(a, b)},
    check=np.array_equal,
    samples=3 if SMOKE else 15,
)

# --- intersect1d: sorted-set win, random-input win, and its guards ----------
from pyoverdrive.fastpaths.intersect_sorted import SIZE_THRESHOLD as INTERSECT_THRESHOLD

n_ref = 10_000 if SMOKE else 1_000_000
n_q = 1_000 if SMOKE else 100_000
universe = 3 * n_ref
ref_sorted = np.sort(rng.choice(universe, size=n_ref, replace=False)).astype(np.int64)
q_sorted = np.sort(rng.choice(universe, size=n_q, replace=False)).astype(np.int64)
ref_random = rng.integers(0, universe, size=n_ref, dtype=np.int64)
q_random = rng.integers(0, universe, size=n_q, dtype=np.int64)
suite.measure(
    case=f"intersect1d_int64_sorted_n{n_ref}_m{n_q}",
    params={"dtype": "int64", "len_a": n_ref, "len_b": n_q, "regime": "sorted", "expect": "huge win"},
    baseline=("stock_intersect1d", lambda a=ref_sorted, b=q_sorted: STOCK_INTERSECT(a, b)),
    candidates={"pyoverdrive": lambda a=ref_sorted, b=q_sorted: np.intersect1d(a, b)},
    check=np.array_equal,
    samples=3 if SMOKE else 5,
)
suite.measure(
    case=f"intersect1d_int64_random_n{n_ref}_m{n_q}",
    params={"dtype": "int64", "len_a": n_ref, "len_b": n_q, "regime": "random", "expect": "big win"},
    baseline=("stock_intersect1d", lambda a=ref_random, b=q_random: STOCK_INTERSECT(a, b)),
    candidates={"pyoverdrive": lambda a=ref_random, b=q_random: np.intersect1d(a, b)},
    check=np.array_equal,
    samples=3 if SMOKE else 5,
)
half = INTERSECT_THRESHOLD // 2 - 1
small_a = rng.integers(0, 3 * half, size=half, dtype=np.int64)
small_b = rng.integers(0, 3 * half, size=half, dtype=np.int64)
suite.measure(
    case=f"intersect1d_int64_random_n{half}_m{half}_below_threshold",
    params={"dtype": "int64", "len_a": half, "len_b": half, "expect": "fallback, tax only"},
    baseline=("stock_intersect1d", lambda a=small_a, b=small_b: STOCK_INTERSECT(a, b)),
    candidates={"pyoverdrive": lambda a=small_a, b=small_b: np.intersect1d(a, b)},
    check=np.array_equal,
    samples=3 if SMOKE else 15,
)
suite.measure(
    case=f"intersect1d_assume_unique_n{n_ref}_m{n_q}_kwargs_route",
    params={"dtype": "int64", "len_a": n_ref, "len_b": n_q, "expect": "fallback, tax only"},
    baseline=("stock_intersect1d", lambda a=ref_sorted, b=q_sorted: STOCK_INTERSECT(a, b, assume_unique=True)),
    candidates={"pyoverdrive": lambda a=ref_sorted, b=q_sorted: np.intersect1d(a, b, assume_unique=True)},
    check=np.array_equal,
    samples=3 if SMOKE else 7,
)

# --- relayout: transposed 2-D copy through the public API -------------------
from pyoverdrive.fastpaths.relayout_blocked import SUPPORTED as RELAYOUT_SUPPORTED

for dtype, threshold in list(RELAYOUT_SUPPORTED.items())[:2]:
    dt = np.dtype(dtype).name
    n_side = 128 if SMOKE else 2048
    base = rng.standard_normal((n_side, n_side)).astype(dtype)
    xt = base.T  # F-contiguous view
    suite.measure(
        case=f"ascontiguousarray_{dt}_{n_side}x{n_side}_transposed",
        params={"dtype": dt, "n": n_side, "expect": "big win"},
        baseline=("stock_ascontiguousarray", lambda x=xt: STOCK_ASCONTIG(x)),
        candidates={"pyoverdrive": lambda x=xt: np.ascontiguousarray(x)},
        check=lambda c, b: c.flags.c_contiguous and np.array_equal(c, b),
        samples=3 if SMOKE else 7,
    )
    suite.measure(
        case=f"ascontiguousarray_{dt}_{n_side}x{n_side}_already_c_guard",
        params={"dtype": dt, "n": n_side, "expect": "no copy either way, tax only"},
        baseline=("stock_ascontiguousarray", lambda x=base: STOCK_ASCONTIG(x)),
        candidates={"pyoverdrive": lambda x=base: np.ascontiguousarray(x)},
        check=lambda c, b: c is b,
        samples=3 if SMOKE else 15,
    )
    small = rng.standard_normal((200, 200)).astype(dtype).T
    suite.measure(
        case=f"ascontiguousarray_{dt}_200x200_below_threshold",
        params={"dtype": dt, "n": 200, "expect": "fallback, tax only"},
        baseline=("stock_ascontiguousarray", lambda x=small: STOCK_ASCONTIG(x)),
        candidates={"pyoverdrive": lambda x=small: np.ascontiguousarray(x)},
        check=np.array_equal,
        samples=3 if SMOKE else 15,
    )

# --- PyRallel: threaded unary ufuncs through the public API -----------------
# The win at the top of the calibrated range, the marginal win at the
# crossover, and the protected baselines (below threshold, excluded dtype,
# strided input, kwargs route), for the first op in the table and exp if
# present. Bit-identity is the check, as everywhere for this family.
def _bit_identical(c, b):
    return c.dtype == b.dtype and c.shape == b.shape and np.array_equal(c, b, equal_nan=True)


_DOMAIN = {"sin": (0.0, 6.28), "cos": (0.0, 6.28), "tan": (-1.5, 1.5), "exp": (-5.0, 5.0),
           "log": (0.1, 100.0), "log10": (0.1, 100.0), "tanh": (-4.0, 4.0), "sqrt": (0.0, 100.0)}

_pyrallel_ops = [op for op in ("sin", "exp") if op in PYRALLEL_SUPPORTED]
for op in _pyrallel_ops:
    stock = STOCK_UFUNCS[op]
    patched = getattr(np, op)
    lo, hi = _DOMAIN[op]
    for dtype, threshold in PYRALLEL_SUPPORTED[op].items():
        dt = np.dtype(dtype).name
        n_top = 10_000 if SMOKE else 10_000_000
        if not SMOKE and n_top < threshold:
            n_top = threshold
        x_top = np.linspace(lo, hi, n_top, dtype=dtype)
        suite.measure(
            case=f"{op}_{dt}_n{n_top}_top_of_range",
            params={"op": op, "dtype": dt, "n": n_top, "expect": "big win"},
            baseline=(f"stock_{op}", lambda s=stock, x=x_top: s(x)),
            candidates={"pyoverdrive": lambda f=patched, x=x_top: f(x)},
            check=_bit_identical,
            samples=3 if SMOKE else 7,
        )
        x_cross = np.linspace(lo, hi, threshold, dtype=dtype)
        suite.measure(
            case=f"{op}_{dt}_n{threshold}_at_crossover",
            params={"op": op, "dtype": dt, "n": threshold, "expect": "marginal win"},
            baseline=(f"stock_{op}", lambda s=stock, x=x_cross: s(x)),
            candidates={"pyoverdrive": lambda f=patched, x=x_cross: f(x)},
            check=_bit_identical,
            samples=3 if SMOKE else 11,
        )
        out_buf = np.empty_like(x_top)
        suite.measure(
            case=f"{op}_{dt}_n{n_top}_out_inplace_idiom",
            params={"op": op, "dtype": dt, "n": n_top, "expect": "big win, out= form"},
            baseline=(f"stock_{op}", lambda s=stock, x=x_top, o=out_buf: s(x, out=o)),
            candidates={"pyoverdrive": lambda f=patched, x=x_top, o=out_buf: f(x, out=o)},
            check=_bit_identical,
            samples=3 if SMOKE else 7,
        )
        x_below = np.linspace(lo, hi, threshold - 1, dtype=dtype)
        suite.measure(
            case=f"{op}_{dt}_n{threshold - 1}_below_threshold",
            params={"op": op, "dtype": dt, "n": threshold - 1, "expect": "fallback, tax only"},
            baseline=(f"stock_{op}", lambda s=stock, x=x_below: s(x)),
            candidates={"pyoverdrive": lambda f=patched, x=x_below: f(x)},
            check=_bit_identical,
            samples=3 if SMOKE else 11,
        )
        x_strided = np.linspace(lo, hi, 2 * n_top, dtype=dtype)[::2]
        suite.measure(
            case=f"{op}_{dt}_n{n_top}_strided_guard",
            params={"op": op, "dtype": dt, "n": n_top, "expect": "fallback, tax only"},
            baseline=(f"stock_{op}", lambda s=stock, x=x_strided: s(x)),
            candidates={"pyoverdrive": lambda f=patched, x=x_strided: f(x)},
            check=_bit_identical,
            samples=3 if SMOKE else 7,
        )
        mask = np.ones(n_top, dtype=bool)
        suite.measure(
            case=f"{op}_{dt}_n{n_top}_where_kwarg_route",
            params={"op": op, "dtype": dt, "n": n_top, "expect": "fallback, tax only"},
            baseline=(f"stock_{op}", lambda s=stock, x=x_top, m=mask: s(x, where=m, out=np.zeros_like(x))),
            candidates={"pyoverdrive": lambda f=patched, x=x_top, m=mask: f(x, where=m, out=np.zeros_like(x))},
            check=_bit_identical,
            samples=3 if SMOKE else 7,
        )

# --- PyRallel binary family: add through the public API --------------------
if "add" in BINARY_SUPPORTED:
    stock_add = STOCK_BINARY["add"]
    for dtype, threshold in BINARY_SUPPORTED["add"].items():
        dt = np.dtype(dtype).name
        n_top = 10_000 if SMOKE else max(10_000_000, threshold)
        if dtype.kind == "f":
            a_top = rng.random(n_top).astype(dtype); b_top = rng.random(n_top).astype(dtype)
        else:
            a_top = rng.integers(0, 1000, size=n_top, dtype=dtype); b_top = rng.integers(0, 1000, size=n_top, dtype=dtype)
        suite.measure(
            case=f"add_{dt}_n{n_top}_top_of_range",
            params={"op": "add", "dtype": dt, "n": n_top, "expect": "modest win (bandwidth)"},
            baseline=("stock_add", lambda s=stock_add, a=a_top, b=b_top: s(a, b)),
            candidates={"pyoverdrive": lambda a=a_top, b=b_top: np.add(a, b)},
            check=_bit_identical,
            samples=3 if SMOKE else 7,
        )
        out_buf = np.empty_like(a_top)
        suite.measure(
            case=f"add_{dt}_n{n_top}_out_inplace_idiom",
            params={"op": "add", "dtype": dt, "n": n_top, "expect": "modest win, out= form"},
            baseline=("stock_add", lambda s=stock_add, a=a_top, b=b_top, o=out_buf: s(a, b, out=o)),
            candidates={"pyoverdrive": lambda a=a_top, b=b_top, o=out_buf: np.add(a, b, out=o)},
            check=_bit_identical,
            samples=3 if SMOKE else 7,
        )
        suite.measure(
            case=f"add_{dt}_n{n_top}_operator_form_not_patched",
            params={"op": "add", "dtype": dt, "n": n_top, "expect": "parity: a + b bypasses numpy.add"},
            baseline=("stock_add", lambda s=stock_add, a=a_top, b=b_top: s(a, b)),
            candidates={"operator_plus": lambda a=a_top, b=b_top: a + b},
            check=_bit_identical,
            samples=3 if SMOKE else 7,
        )
        suite.measure(
            case=f"add_{dt}_n{n_top}_broadcast_guard",
            params={"op": "add", "dtype": dt, "n": n_top, "expect": "fallback, tax only"},
            baseline=("stock_add", lambda s=stock_add, a=a_top, b=b_top: s(a, b[:1])),
            candidates={"pyoverdrive": lambda a=a_top, b=b_top: np.add(a, b[:1])},
            check=_bit_identical,
            samples=3 if SMOKE else 7,
        )
        break  # one dtype is enough for the public-API proof; the battery has the rest

# --- fftconvolve: the win and its guards ------------------------------------
cv_n = 2_000 if SMOKE else 10_000
cv_a = rng.standard_normal(cv_n)
cv_v = rng.standard_normal(1_000)
suite.measure(
    case=f"convolve_float64_n{cv_n}_m1000_full_default_mode",
    params={"dtype": "float64", "n": cv_n, "m": 1000, "expect": "big win"},
    baseline=("stock_convolve", lambda a=cv_a, v=cv_v: STOCK_CONVOLVE(a, v)),
    candidates={"pyoverdrive": lambda a=cv_a, v=cv_v: np.convolve(a, v)},
    # FFT error scales with the operand norms, not the (possibly tiny) edge
    # lags, so the bound is an absolute one far above ~5e-12 measured
    check=lambda c, b: np.allclose(c, b, rtol=1e-9, atol=1e-9),
    samples=3 if SMOKE else 7,
)
cv_ai = rng.integers(-100, 101, size=cv_n, dtype=np.int64)
cv_vi = rng.integers(-100, 101, size=1_000, dtype=np.int64)
suite.measure(
    case=f"correlate_int64_n{cv_n}_m1000_full_kwarg",
    params={"dtype": "int64", "n": cv_n, "m": 1000, "expect": "big win, bit-identical"},
    baseline=("stock_correlate", lambda a=cv_ai, v=cv_vi: STOCK_CORRELATE(a, v, mode="full")),
    candidates={"pyoverdrive": lambda a=cv_ai, v=cv_vi: np.correlate(a, v, mode="full")},
    check=np.array_equal,
    samples=3 if SMOKE else 7,
)
thin_a = rng.standard_normal(5_000)
thin_v = rng.standard_normal(300)
suite.measure(
    case="convolve_float64_n5000_m300_thin_kernel_guard",
    params={"dtype": "float64", "n": 5000, "m": 300, "expect": "fallback, tax only"},
    baseline=("stock_convolve", lambda a=thin_a, v=thin_v: STOCK_CONVOLVE(a, v)),
    candidates={"pyoverdrive": lambda a=thin_a, v=thin_v: np.convolve(a, v)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)
suite.measure(
    case=f"convolve_float64_n{cv_n}_m1000_same_mode",
    params={"dtype": "float64", "n": cv_n, "m": 1000, "mode": "same",
            "expect": "win: the everyday smoothing idiom"},
    baseline=("stock_convolve", lambda a=cv_a, v=cv_v: STOCK_CONVOLVE(a, v, "same")),
    candidates={"pyoverdrive": lambda a=cv_a, v=cv_v: np.convolve(a, v, "same")},
    check=lambda c, b: np.allclose(c, b, rtol=1e-9, atol=1e-9) and c.shape == b.shape,
    samples=3 if SMOKE else 7,
)
suite.measure(
    case=f"correlate_float64_n{cv_n}_m1000_default_valid_now_dispatches",
    params={"dtype": "float64", "n": cv_n, "m": 1000, "mode": "valid (default)",
            "expect": "win: bare np.correlate(a, v) on unequal lengths"},
    baseline=("stock_correlate", lambda a=cv_a, v=cv_v: STOCK_CORRELATE(a, v)),
    candidates={"pyoverdrive": lambda a=cv_a, v=cv_v: np.correlate(a, v)},
    check=lambda c, b: np.allclose(c, b, rtol=1e-9, atol=1e-9) and c.shape == b.shape,
    samples=3 if SMOKE else 7,
)
nan_a = cv_a.copy()
nan_a[123] = np.nan
suite.measure(
    case=f"convolve_float64_n{cv_n}_m1000_nan_guard",
    params={"dtype": "float64", "n": cv_n, "m": 1000, "expect": "fallback: FFT would smear the NaN"},
    baseline=("stock_convolve", lambda a=nan_a, v=cv_v: STOCK_CONVOLVE(a, v)),
    candidates={"pyoverdrive": lambda a=nan_a, v=cv_v: np.convolve(a, v)},
    check=lambda c, b: bool(np.array_equal(c, b, equal_nan=True)),
    samples=3 if SMOKE else 5,
)

# --- nanquantile: the win and its guards ------------------------------------
nq_shape = (10, 40) if SMOKE else (27, 100)
nq = rng.uniform(-5.0, 5.0, size=nq_shape)
nq[rng.random(nq_shape) < 0.1] = np.nan
suite.measure(
    case=f"nanquantile_float64_{nq_shape[0]}x{nq_shape[1]}_axis0",
    params={"shape": list(nq_shape), "q": 0.8, "nan_frac": 0.1, "expect": "big win"},
    baseline=("stock_nanquantile", lambda a=nq: STOCK_NANQUANTILE(a, 0.8, axis=0)),
    candidates={"pyoverdrive": lambda a=nq: np.nanquantile(a, 0.8, axis=0)},
    check=lambda c, b: bool(np.array_equal(c, b, equal_nan=True)),
    samples=3 if SMOKE else 7,
)
nq_anti = rng.uniform(size=(10_000, 3))
nq_anti[rng.random((10_000, 3)) < 0.1] = np.nan
suite.measure(
    case="nanquantile_float64_10000x3_anti_regime_guard",
    params={"shape": [10_000, 3], "q": 0.8, "expect": "fallback, tax only"},
    baseline=("stock_nanquantile", lambda a=nq_anti: STOCK_NANQUANTILE(a, 0.8, axis=0)),
    candidates={"pyoverdrive": lambda a=nq_anti: np.nanquantile(a, 0.8, axis=0)},
    check=lambda c, b: bool(np.array_equal(c, b, equal_nan=True)),
    samples=3 if SMOKE else 7,
)
nq_q = rng.uniform(size=nq_shape)
suite.measure(
    case=f"nanquantile_float64_{nq_shape[0]}x{nq_shape[1]}_q_sequence_guard",
    params={"shape": list(nq_shape), "q": [0.2, 0.8], "expect": "fallback, tax only"},
    baseline=("stock_nanquantile", lambda a=nq_q: STOCK_NANQUANTILE(a, [0.2, 0.8], axis=0)),
    candidates={"pyoverdrive": lambda a=nq_q: np.nanquantile(a, [0.2, 0.8], axis=0)},
    check=lambda c, b: bool(np.array_equal(c, b, equal_nan=True)),
    samples=3 if SMOKE else 7,
)

# --- einsum: the win and its guards -----------------------------------------
es_t = 200 if SMOKE else 1000
es_x = rng.uniform(0.5, 1.5, size=(es_t, 1, 500))
es_y = rng.uniform(0.5, 1.5, size=(es_t, 1, 500))
_es_close = lambda c, b: bool(  # noqa: E731
    np.allclose(c, b, rtol=1e-6, atol=1e-9 * max(1.0, float(np.abs(b).max())))
)
suite.measure(
    case=f"einsum_thd_float64_t{es_t}_reported_shape",
    params={"pattern": "thd,Thd->thT", "t": es_t, "d": 500, "expect": "big win"},
    baseline=("stock_einsum", lambda x=es_x, y=es_y: STOCK_EINSUM("thd,Thd->thT", x, y)),
    candidates={"pyoverdrive": lambda x=es_x, y=es_y: np.einsum("thd,Thd->thT", x, y)},
    check=_es_close,
    samples=3 if SMOKE else 5,
)
es_small = rng.uniform(0.5, 1.5, size=3)
suite.measure(
    case="einsum_inner_len3_dgasmith_guard",
    params={"pattern": "i,i->", "len": 3, "expect": "fallback: the counter-case that keeps optimize off by default"},
    baseline=("stock_einsum", lambda v=es_small: STOCK_EINSUM("i,i->", v, v)),
    candidates={"pyoverdrive": lambda v=es_small: np.einsum("i,i->", v, v)},
    check=lambda c, b: bool(np.isclose(float(c), float(b))),
    samples=3 if SMOKE else 11,
)
es_mid = rng.uniform(0.5, 1.5, size=100_000)
suite.measure(
    case="einsum_inner_len100000_scalar_floor_guard",
    params={"pattern": "i,i->", "len": 100_000, "expect": "fallback: scalar-output floor is 1e6"},
    baseline=("stock_einsum", lambda v=es_mid: STOCK_EINSUM("i,i->", v, v)),
    candidates={"pyoverdrive": lambda v=es_mid: np.einsum("i,i->", v, v)},
    check=lambda c, b: bool(np.isclose(float(c), float(b))),
    samples=3 if SMOKE else 7,
)

# --- searchsorted: the win and its guards -----------------------------------
ss_n = 20_000 if SMOKE else 1_000_000
ss_x = np.sort(rng.standard_normal(ss_n))
ss_v = rng.standard_normal(ss_n)
suite.measure(
    case=f"searchsorted_float64_n{ss_n}_random_queries",
    params={"len_x": ss_n, "len_y": ss_n, "expect": "big win"},
    baseline=("stock_searchsorted", lambda x=ss_x, v=ss_v: STOCK_SEARCHSORTED(x, v)),
    candidates={"pyoverdrive": lambda x=ss_x, v=ss_v: np.searchsorted(x, v)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)
ss_sorted = np.sort(ss_v)
suite.measure(
    case=f"searchsorted_float64_n{ss_n}_sorted_queries_guard",
    params={"len_x": ss_n, "len_y": ss_n, "expect": "fallback: stock is already fast on sorted queries"},
    baseline=("stock_searchsorted", lambda x=ss_x, v=ss_sorted: STOCK_SEARCHSORTED(x, v)),
    candidates={"pyoverdrive": lambda x=ss_x, v=ss_sorted: np.searchsorted(x, v)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)

# --- isclose: the win and its guards ----------------------------------------
ic_a = rng.uniform(-5.0, 5.0, 500)
ic_b = ic_a + rng.uniform(-1e-7, 1e-7, 500)
suite.measure(
    case="isclose_float64_n500",
    params={"n": 500, "expect": "overhead win"},
    baseline=("stock_isclose", lambda a=ic_a, b=ic_b: STOCK_ISCLOSE(a, b)),
    candidates={"pyoverdrive": lambda a=ic_a, b=ic_b: np.isclose(a, b)},
    check=np.array_equal,
    samples=3 if SMOKE else 11,
)
suite.measure(
    case="isclose_scalar_pair",
    params={"expect": "overhead win, the reported case"},
    baseline=("stock_isclose", lambda: STOCK_ISCLOSE(0.5, 0.50000001)),
    candidates={"pyoverdrive": lambda: np.isclose(0.5, 0.50000001)},
    check=lambda c, b: bool(c == b),
    samples=3 if SMOKE else 15,
)
ic_nan = ic_a.copy()
ic_nan[7] = np.nan
suite.measure(
    case="isclose_float64_n500_nan_guard",
    params={"n": 500, "expect": "fallback: finiteness scan then stock"},
    baseline=("stock_isclose", lambda a=ic_nan, b=ic_b: STOCK_ISCLOSE(a, b)),
    candidates={"pyoverdrive": lambda a=ic_nan, b=ic_b: np.isclose(a, b)},
    check=np.array_equal,
    samples=3 if SMOKE else 11,
)
ic_big = rng.uniform(-5.0, 5.0, 100_000)
suite.measure(
    case="isclose_float64_n100000_above_cap_guard",
    params={"n": 100_000, "expect": "fallback, tax only"},
    baseline=("stock_isclose", lambda a=ic_big: STOCK_ISCLOSE(a, a)),
    candidates={"pyoverdrive": lambda a=ic_big: np.isclose(a, a)},
    check=np.array_equal,
    samples=3 if SMOKE else 7,
)

# --- isin StringDType + mixed dot (OPP-000023 / OPP-000027) ------------------
sd_vocab = np.array([f"word_{i}" for i in range(3)], dtype=np.dtypes.StringDType())
sd_n = 3_000 if SMOKE else 30_000
sd_el = sd_vocab[rng.integers(0, 3, size=sd_n)]
suite.measure(
    case=f"isin_stringdtype_n{sd_n}_card3_self",
    params={"n": sd_n, "cardinality": 3, "expect": "the 2319x headline regime"},
    baseline=("stock_isin", lambda a=sd_el: STOCK_ISIN(a, a)),
    candidates={"pyoverdrive": lambda a=sd_el: np.isin(a, a)},
    check=np.array_equal,
    samples=3 if SMOKE else 5,
)
sd_u = sd_el.astype("U10")
suite.measure(
    case=f"isin_unicode_n{sd_n}_guard",
    params={"n": sd_n, "expect": "fallback: U dtype stays on stock"},
    baseline=("stock_isin", lambda a=sd_u: STOCK_ISIN(a, a)),
    candidates={"pyoverdrive": lambda a=sd_u: np.isin(a, a)},
    check=np.array_equal,
    samples=3 if SMOKE else 5,
)
dm_A = rng.standard_normal((200, 100) if SMOKE else (2000, 500))
dm_b = rng.standard_normal(dm_A.shape[1]) + 1j * rng.standard_normal(dm_A.shape[1])
suite.measure(
    case=f"dot_real{dm_A.shape[0]}x{dm_A.shape[1]}_complex_vector",
    params={"shape": list(dm_A.shape), "expect": "win via view-as-real GEMV"},
    baseline=("stock_dot", lambda A=dm_A, b=dm_b: STOCK_DOT(A, b)),
    candidates={"pyoverdrive": lambda A=dm_A, b=dm_b: np.dot(A, b)},
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-12, atol=1e-12 * max(1.0, float(np.abs(b).max())))),
    samples=3 if SMOKE else 9,
)
dm_c = dm_A.astype(np.complex128)
suite.measure(
    case="dot_complex_matrix_real_vector_guard",
    params={"expect": "fallback: reverse direction measured no gap"},
    baseline=("stock_dot", lambda A=dm_c, b=dm_b: STOCK_DOT(A, b.real.copy())),
    candidates={"pyoverdrive": lambda A=dm_c, b=dm_b: np.dot(A, b.real.copy())},
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-12)),
    samples=3 if SMOKE else 7,
)

# --- quantile dense q (OPP-000022) ------------------------------------------
qd_a = rng.standard_normal((40, 1024) if SMOKE else (300, 2048))
qd_q = np.linspace(0.0, 1.0, 64 if SMOKE else 512)
suite.measure(
    case=f"quantile_dense_{qd_a.shape[0]}x{qd_a.shape[1]}_nq{qd_q.size}",
    params={"shape": list(qd_a.shape), "nq": qd_q.size, "expect": "big win past the density cliff"},
    baseline=("stock_quantile", lambda a=qd_a, q=qd_q: STOCK_QUANTILE(a, q, axis=-1)),
    candidates={"pyoverdrive": lambda a=qd_a, q=qd_q: np.quantile(a, q, axis=-1)},
    check=lambda c, b: bool(np.array_equal(c, b, equal_nan=True)),
    samples=3 if SMOKE else 7,
)
qp_q = np.linspace(0.0, 100.0, 64 if SMOKE else 512)
suite.measure(
    case=f"percentile_dense_{qd_a.shape[0]}x{qd_a.shape[1]}_nq{qp_q.size}",
    params={"shape": list(qd_a.shape), "nq": qp_q.size, "expect": "the quantile-sibling win"},
    baseline=("stock_percentile", lambda a=qd_a, q=qp_q: STOCK_PERCENTILE(a, q, axis=-1)),
    candidates={"pyoverdrive": lambda a=qd_a, q=qp_q: np.percentile(a, q, axis=-1)},
    check=lambda c, b: bool(np.array_equal(c, b, equal_nan=True)),
    samples=3 if SMOKE else 7,
)
qd_small = np.linspace(0.0, 1.0, 3)
suite.measure(
    case="quantile_q3_below_floor_guard",
    params={"nq": 3, "expect": "fallback, tax only"},
    baseline=("stock_quantile", lambda a=qd_a, q=qd_small: STOCK_QUANTILE(a, q, axis=-1)),
    candidates={"pyoverdrive": lambda a=qd_a, q=qd_small: np.quantile(a, q, axis=-1)},
    check=lambda c, b: bool(np.array_equal(c, b, equal_nan=True)),
    samples=3 if SMOKE else 7,
)

# --- single-char sort/unique via int view (OPP-000024) -----------------------
cv_alpha = np.array(list("ASDFGHJKLZ"), dtype="U1")
cv_n = 1_000 if SMOKE else 10_000
cv_u1 = cv_alpha[rng.integers(0, 10, size=cv_n)]
suite.measure(
    case=f"sort_U1_n{cv_n}",
    params={"dtype": "U1", "n": cv_n, "expect": "the 33x sort-kernel win"},
    baseline=("stock_sort", lambda x=cv_u1: STOCK_SORT(x)),
    candidates={"pyoverdrive": lambda x=cv_u1: np.sort(x)},
    check=lambda c, b: c.dtype == b.dtype and bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 9,
)
suite.measure(
    case=f"unique_U1_counts_n{cv_n}",
    params={"dtype": "U1", "n": cv_n, "expect": "the 26x counts win"},
    baseline=("stock_unique", lambda x=cv_u1: STOCK_UNIQUE(x, return_counts=True)),
    candidates={"pyoverdrive": lambda x=cv_u1: np.unique(x, return_counts=True)},
    check=lambda c, b: c[0].dtype == b[0].dtype
    and all(np.array_equal(ci, bi) for ci, bi in zip(c, b)),
    samples=3 if SMOKE else 9,
)
cv_small = cv_alpha[rng.integers(0, 10, size=500)]
suite.measure(
    case="sort_U1_n500_below_floor_guard",
    params={"dtype": "U1", "n": 500, "expect": "fallback, tax only"},
    baseline=("stock_sort", lambda x=cv_small: STOCK_SORT(x)),
    candidates={"pyoverdrive": lambda x=cv_small: np.sort(x)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 15,
)
cv_u2 = cv_u1.astype("U2")
suite.measure(
    case=f"sort_U2_n{cv_n}_excluded_dtype_guard",
    params={"dtype": "U2", "n": cv_n, "expect": "fallback: multi-char stays on stock"},
    baseline=("stock_sort", lambda x=cv_u2: STOCK_SORT(x)),
    candidates={"pyoverdrive": lambda x=cv_u2: np.sort(x)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)

# --- tiny-trailing-axis reductions (OPP-000026) -------------------------------
rt_img = rng.random(size=(100, 100, 3) if SMOKE else (1000, 1000, 3))
_rt_close = lambda c, b: c.dtype == b.dtype and bool(
    np.allclose(c, b, rtol=1e-9, atol=0.0)
)
suite.measure(
    case=f"mean_{'x'.join(map(str, rt_img.shape))}_axis01",
    params={"shape": list(rt_img.shape), "expect": "the issue's own 3.9x regime"},
    baseline=("stock_mean", lambda a=rt_img: STOCK_MEAN(a, axis=(0, 1))),
    candidates={"pyoverdrive": lambda a=rt_img: np.mean(a, axis=(0, 1))},
    check=_rt_close,
    samples=3 if SMOKE else 9,
)
rt_2d = rng.random(size=(10_000, 2) if SMOKE else (100_000, 2))
suite.measure(
    case=f"sum_{rt_2d.shape[0]}x2_axis0",
    params={"shape": list(rt_2d.shape), "expect": "the 9x k=2 sum win"},
    baseline=("stock_sum", lambda a=rt_2d: STOCK_SUM(a, axis=0)),
    candidates={"pyoverdrive": lambda a=rt_2d: np.sum(a, axis=0)},
    check=_rt_close,
    samples=3 if SMOKE else 9,
)
rt_k8 = rng.random(size=(20_000, 8))
suite.measure(
    case="mean_20000x8_wide_k_guard",
    params={"shape": [20_000, 8], "expect": "fallback: k=8 measured losing"},
    baseline=("stock_mean", lambda a=rt_k8: STOCK_MEAN(a, axis=0)),
    candidates={"pyoverdrive": lambda a=rt_k8: np.mean(a, axis=0)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 9,
)

# --- batched 2x2 eigvalsh closed form (OPP-000030) ----------------------------
ev_n = 1_000 if SMOKE else 10_000
ev_a = rng.uniform(-1.0, 1.0, size=(ev_n, 2, 2))
ev_a = np.ascontiguousarray(ev_a @ np.swapaxes(ev_a, -1, -2) + 0.1 * np.eye(2))
suite.measure(
    case=f"eigvalsh_2x2_batch{ev_n}",
    params={"batch": ev_n, "expect": "the 31x closed-form win"},
    baseline=("stock_eigvalsh", lambda a=ev_a: STOCK_EIGVALSH(a)),
    candidates={"pyoverdrive": lambda a=ev_a: np.linalg.eigvalsh(a)},
    check=lambda c, b: c.dtype == b.dtype
    and bool((np.abs(c - b) <= 1e-9 * np.maximum(1.0, np.abs(b).max(axis=-1, keepdims=True))).all()),
    samples=3 if SMOKE else 9,
)
ev_small = ev_a[:50]
suite.measure(
    case="eigvalsh_2x2_batch50_below_floor_guard",
    params={"batch": 50, "expect": "fallback, tax + finite-scan only"},
    baseline=("stock_eigvalsh", lambda a=ev_small: STOCK_EIGVALSH(a)),
    candidates={"pyoverdrive": lambda a=ev_small: np.linalg.eigvalsh(a)},
    check=lambda c, b: bool(np.allclose(c, b)),
    samples=3 if SMOKE else 15,
)

# --- complex-by-real matmul split (OPP-000029) ---------------------------------
mm_shape = (64, 400, 400) if SMOKE else (64, 2000, 2000)
mm_c = (
    rng.uniform(0.5, 1.5, size=mm_shape[:2]) + 1j * rng.uniform(0.5, 1.5, size=mm_shape[:2])
).astype(np.complex128)
mm_r = rng.uniform(0.5, 1.5, size=mm_shape[1:])
suite.measure(
    case=f"matmul_C{mm_shape[0]}x{mm_shape[1]}_R{mm_shape[1]}x{mm_shape[2]}",
    params={"shape": list(mm_shape), "expect": "the 3x split-GEMM win"},
    baseline=("stock_matmul", lambda c=mm_c, r=mm_r: STOCK_MATMUL(c, r)),
    candidates={"pyoverdrive": lambda c=mm_c, r=mm_r: np.matmul(c, r)},
    check=lambda c, b: c.dtype == b.dtype
    and bool(np.allclose(c, b, rtol=1e-12, atol=1e-12 * float(np.abs(b).max()))),
    samples=3 if SMOKE else 9,
)
mm_c_big = np.vstack([mm_c] * (8 if SMOKE else 8))  # m > 256: stays on stock
suite.measure(
    case=f"matmul_C{mm_c_big.shape[0]}x{mm_shape[1]}_wide_m_guard",
    params={"m": mm_c_big.shape[0], "expect": "fallback: square/tall C measured losing"},
    baseline=("stock_matmul", lambda c=mm_c_big, r=mm_r: STOCK_MATMUL(c, r)),
    candidates={"pyoverdrive": lambda c=mm_c_big, r=mm_r: np.matmul(c, r)},
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-12, atol=1e-12 * float(np.abs(b).max()))),
    samples=3 if SMOKE else 7,
)

# --- small 1-D roll via concatenate (OPP-000032) -------------------------------
rl_d = rng.random(1_000)
suite.measure(
    case="roll_1d_n1000_shift7",
    params={"n": 1_000, "shift": 7, "expect": "the ~4.7x machinery-overhead win"},
    baseline=("stock_roll", lambda d=rl_d: STOCK_ROLL(d, 7)),
    candidates={"pyoverdrive": lambda d=rl_d: np.roll(d, 7)},
    check=lambda c, b: c.dtype == b.dtype and bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 15,
)
rl_big = rng.random(50_000)
suite.measure(
    case="roll_1d_n50000_above_cap_guard",
    params={"n": 50_000, "expect": "fallback, tax only"},
    baseline=("stock_roll", lambda d=rl_big: STOCK_ROLL(d, 7)),
    candidates={"pyoverdrive": lambda d=rl_big: np.roll(d, 7)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 9,
)

# --- calibration-gated argmax (OPP-000034) ------------------------------------
# This row's meaning depends on THIS machine's calibration: where the
# per-machine probe enabled argmax_blocked_transpose it measures the win
# (Intel Alder Lake: ~2.5x); where the probe declined (AMD Zen 4) the
# path is off and the row honestly measures stock parity plus tax. The
# case name records which, so the two fingerprints' evidence files stay
# self-describing.
from pyoverdrive.dispatcher.gearbox import GEARBOX as _GB

am_a = rng.random(size=(400, 400) if SMOKE else (3_200, 3_200))
_am_state = (
    "calibrated_on"
    if _GB.decide("numpy.argmax", (am_a,), {"axis": 0})[0] != "stock"
    else "calibrated_off"
)
suite.measure(
    case=f"argmax_axis0_{am_a.shape[0]}x{am_a.shape[1]}_{_am_state}",
    params={"shape": list(am_a.shape), "expect": "win iff this machine's calibration enabled it"},
    baseline=("stock_argmax", lambda a=am_a: STOCK_ARGMAX(a, axis=0)),
    candidates={"pyoverdrive": lambda a=am_a: np.argmax(a, axis=0)},
    check=lambda c, b: c.dtype == b.dtype and bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)

# --- batch-5 families ---------------------------------------------------------
iv_n = 1_000 if SMOKE else 10_000
iv_a = rng.uniform(-1.0, 1.0, size=(iv_n, 3, 3))
iv_a = np.ascontiguousarray(iv_a @ np.swapaxes(iv_a, -1, -2) + 0.1 * np.eye(3))
suite.measure(
    case=f"inv_3x3_batch{iv_n}",
    params={"batch": iv_n, "expect": "the 3-8x adjugate win"},
    baseline=("stock_inv", lambda a=iv_a: STOCK_INV(a)),
    candidates={"pyoverdrive": lambda a=iv_a: np.linalg.inv(a)},
    check=lambda c, b: c.dtype == b.dtype
    and bool((np.abs(c - b) <= 1e-9 * np.maximum(1.0, np.abs(b).max(axis=(-2, -1), keepdims=True))).all()),
    samples=3 if SMOKE else 9,
)
iv_small = iv_a[:50]
suite.measure(
    case="inv_3x3_batch50_below_floor_guard",
    params={"batch": 50, "expect": "fallback, tax + scans only"},
    baseline=("stock_inv", lambda a=iv_small: STOCK_INV(a)),
    candidates={"pyoverdrive": lambda a=iv_small: np.linalg.inv(a)},
    check=lambda c, b: bool(np.allclose(c, b)),
    samples=3 if SMOKE else 15,
)

io_vocab = [f"key_{i:05d}" for i in range(5_000)]
io_n = 3_000 if SMOKE else 30_000
io_el = np.array([io_vocab[i] for i in rng.integers(0, 5_000, size=io_n)], dtype=object)
io_te = np.array(
    [io_vocab[i] for i in rng.choice(5_000, size=io_n // 10, replace=False)], dtype=object
)
suite.measure(
    case=f"isin_object_n{io_n}_m{io_n // 10}",
    params={"n": io_n, "m": io_n // 10, "expect": "the 262x object-set win"},
    baseline=("stock_isin", lambda e=io_el, t=io_te: STOCK_ISIN(e, t)),
    candidates={"pyoverdrive": lambda e=io_el, t=io_te: np.isin(e, t)},
    check=np.array_equal,
    samples=3 if SMOKE else 5,
)
suite.measure(
    case="isin_object_n100_m5_below_floor_guard",
    params={"n": 100, "m": 5, "expect": "fallback, tax only"},
    baseline=("stock_isin", lambda e=io_el[:100], t=io_te[:5]: STOCK_ISIN(e, t)),
    candidates={"pyoverdrive": lambda e=io_el[:100], t=io_te[:5]: np.isin(e, t)},
    check=np.array_equal,
    samples=3 if SMOKE else 11,
)

md_x = rng.random(1_001)
suite.measure(
    case="median_1d_n1001",
    params={"n": 1_001, "expect": "the ~2.4x partition win"},
    baseline=("stock_median", lambda x=md_x: STOCK_MEDIAN(x)),
    candidates={"pyoverdrive": lambda x=md_x: np.median(x)},
    check=lambda c, b: bool(np.asarray(c) == np.asarray(b)),
    samples=3 if SMOKE else 15,
)
md_big = rng.random(50_000)
suite.measure(
    case="median_1d_n50000_above_cap_guard",
    params={"n": 50_000, "expect": "fallback, tax only"},
    baseline=("stock_median", lambda x=md_big: STOCK_MEDIAN(x)),
    candidates={"pyoverdrive": lambda x=md_big: np.median(x)},
    check=lambda c, b: bool(np.asarray(c) == np.asarray(b)),
    samples=3 if SMOKE else 9,
)

h2_n = 100_000 if SMOKE else 1_000_000
h2_x = rng.normal(0.0, 1.0, size=h2_n)
h2_y = rng.normal(0.0, 1.0, size=h2_n)
_h2_check = lambda c, b: all(np.array_equal(ci, bi) for ci, bi in zip(c, b)) and c[0].dtype == b[0].dtype
suite.measure(
    case=f"hist2d_n{h2_n}_bins100x100",
    params={"n": h2_n, "bins": [100, 100], "expect": "the 1.6-2.4x direct-index win"},
    baseline=("stock_hist2d", lambda x=h2_x, y=h2_y: STOCK_HIST2D(x, y, bins=[100, 100], range=[[-3, 3], [-3, 3]])),
    candidates={"pyoverdrive": lambda x=h2_x, y=h2_y: np.histogram2d(x, y, bins=[100, 100], range=[[-3, 3], [-3, 3]])},
    check=_h2_check,
    samples=3 if SMOKE else 7,
)
suite.measure(
    case=f"hist2d_n{h2_n}_bins10x10_below_floor_guard",
    params={"n": h2_n, "bins": [10, 10], "expect": "fallback, tax only"},
    baseline=("stock_hist2d", lambda x=h2_x, y=h2_y: STOCK_HIST2D(x, y, bins=[10, 10], range=[[-3, 3], [-3, 3]])),
    candidates={"pyoverdrive": lambda x=h2_x, y=h2_y: np.histogram2d(x, y, bins=[10, 10], range=[[-3, 3], [-3, 3]])},
    check=_h2_check,
    samples=3 if SMOKE else 7,
)

ur_rows = rng.integers(-50, 50, size=(10_000 if SMOKE else 100_000, 2), dtype=np.int64)
suite.measure(
    case=f"unique_rows_n{ur_rows.shape[0]}_k2",
    params={"n": int(ur_rows.shape[0]), "k": 2, "expect": "the ~5x lexsort win"},
    baseline=("stock_unique", lambda a=ur_rows: STOCK_UNIQUE(a, axis=0)),
    candidates={"pyoverdrive": lambda a=ur_rows: np.unique(a, axis=0)},
    check=lambda c, b: c.dtype == b.dtype and bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)
ur_f = rng.random(size=(20_000, 2))
suite.measure(
    case="unique_rows_float_excluded_guard",
    params={"n": 20_000, "k": 2, "expect": "fallback: float rows stay on stock"},
    baseline=("stock_unique", lambda a=ur_f: STOCK_UNIQUE(a, axis=0)),
    candidates={"pyoverdrive": lambda a=ur_f: np.unique(a, axis=0)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)

se_a = np.sort(rng.integers(-(2**60), 2**60, size=1_000 if SMOKE else 100_000, dtype=np.int64))
se_key = 2**70
suite.measure(
    case=f"searchsorted_extreme_key_n{se_a.size}",
    params={"n": int(se_a.size), "key": "2**70", "expect": "O(1) vs stock's bigint walk"},
    baseline=("stock_searchsorted", lambda a=se_a, k=se_key: STOCK_SEARCHSORTED(a, k)),
    candidates={"pyoverdrive": lambda a=se_a, k=se_key: np.searchsorted(a, k)},
    check=lambda c, b: type(c) is type(b) and c == b,
    samples=3 if SMOKE else 7,
)
suite.measure(
    case="searchsorted_inrange_pyint_guard",
    params={"n": int(se_a.size), "expect": "fallback: in-range keys measured no win"},
    baseline=("stock_searchsorted", lambda a=se_a: STOCK_SEARCHSORTED(a, 12345)),
    candidates={"pyoverdrive": lambda a=se_a: np.searchsorted(a, 12345)},
    check=lambda c, b: type(c) is type(b) and c == b,
    samples=3 if SMOKE else 15,
)

tiny = np.linspace(0.0, 1.0, 100)
if "sin" in PYRALLEL_SUPPORTED:
    suite.measure(
        case="sin_n100_microsecond_scale",
        params={"n": 100, "expect": "no dispatch, tax visible at ns scale"},
        baseline=("stock_sin", lambda s=STOCK_UFUNCS["sin"], x=tiny: s(x)),
        candidates={"pyoverdrive": lambda x=tiny: np.sin(x)},
        check=_bit_identical,
        samples=3 if SMOKE else 15,
    )

# --- batch-6 families: nan-family scans, int BLAS matmul, small-batch linalg,
# --- nan_to_num - each a win row plus the honest guard/fallback rows --------
nm2 = rng.standard_normal((1000, 100))
suite.measure(
    case="nanmean_1000x100_axis1",
    params={"shape": [1000, 100], "expect": "scan + plain mean"},
    baseline=("stock_nanmean", lambda a=nm2: STOCK_NANMEAN(a, axis=1)),
    candidates={"pyoverdrive": lambda a=nm2: np.nanmean(a, axis=1)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 9,
)
nm_salted = rng.standard_normal(100_000)
nm_salted[rng.random(100_000) < 0.01] = np.nan
suite.measure(
    case="nanmean_nan_present_guard",
    params={"n": 100_000, "expect": "internal fallback: scan cost only"},
    baseline=("stock_nanmean", lambda a=nm_salted: STOCK_NANMEAN(a)),
    candidates={"pyoverdrive": lambda a=nm_salted: np.nanmean(a)},
    check=lambda c, b: bool(np.array_equal(c, b, equal_nan=True)),
    samples=3 if SMOKE else 9,
)
ns1 = rng.standard_normal(100_000)
suite.measure(
    case="nansum_n100000",
    params={"n": 100_000, "expect": "scan + plain sum"},
    baseline=("stock_nansum", lambda a=ns1: STOCK_NANSUM(a)),
    candidates={"pyoverdrive": lambda a=ns1: np.nansum(a)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 9,
)
suite.measure(
    case="nanstd_1000x100_axis1",
    params={"shape": [1000, 100], "expect": "scan + plain std"},
    baseline=("stock_nanstd", lambda a=nm2: STOCK_NANSTD(a, axis=1)),
    candidates={"pyoverdrive": lambda a=nm2: np.nanstd(a, axis=1)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 9,
)
na2 = rng.standard_normal((1000, 1000))
suite.measure(
    case="nanargmax_1000x1000_axis1",
    params={"shape": [1000, 1000], "expect": "scan + plain argmax"},
    baseline=("stock_nanargmax", lambda a=na2: STOCK_NANARGMAX(a, axis=1)),
    candidates={"pyoverdrive": lambda a=na2: np.nanargmax(a, axis=1)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 9,
)
suite.measure(
    case="nanmedian_1000x1000_axis1",
    params={"shape": [1000, 1000], "expect": "scan + vectorized median"},
    baseline=("stock_nanmedian", lambda a=na2: STOCK_NANMEDIAN(a, axis=1)),
    candidates={"pyoverdrive": lambda a=na2: np.nanmedian(a, axis=1)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)
nmed_anti = rng.standard_normal((200, 10_000))
suite.measure(
    case="nanmedian_long_slices_guard",
    params={"shape": [200, 10_000], "expect": "fallback: reduced_len over cap"},
    baseline=("stock_nanmedian", lambda a=nmed_anti: STOCK_NANMEDIAN(a, axis=1)),
    candidates={"pyoverdrive": lambda a=nmed_anti: np.nanmedian(a, axis=1)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)
np_2d = rng.standard_normal((27, 100))
np_2d[rng.random((27, 100)) < 0.1] = np.nan
suite.measure(
    case="nanpercentile_27x100_axis0",
    params={"shape": [27, 100], "nan_frac": 0.1, "expect": "masked route at q/100"},
    baseline=("stock_nanpercentile", lambda a=np_2d: STOCK_NANPERCENTILE(a, 80.0, axis=0)),
    candidates={"pyoverdrive": lambda a=np_2d: np.nanpercentile(a, 80.0, axis=0)},
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-12, equal_nan=True)),
    samples=3 if SMOKE else 9,
)
imm_x = rng.integers(-1000, 1000, (400, 400)).astype(np.int64)
imm_y = rng.integers(-1000, 1000, (400, 400)).astype(np.int64)
suite.measure(
    case="matmul_int64_400",
    params={"n": 400, "expect": "exact f64 BLAS round-trip"},
    baseline=("stock_matmul", lambda x=imm_x, y=imm_y: STOCK_MATMUL(x, y)),
    candidates={"pyoverdrive": lambda x=imm_x, y=imm_y: np.matmul(x, y)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)
imm_big = (imm_x.astype(np.int64) * 0 + 2**30)  # k*max*max far over 2^53
suite.measure(
    case="matmul_int64_overbound_guard",
    params={"n": 400, "expect": "fallback: exactness bound refuses"},
    baseline=("stock_matmul", lambda x=imm_big, y=imm_big: STOCK_MATMUL(x, y)),
    candidates={"pyoverdrive": lambda x=imm_big, y=imm_big: np.matmul(x, y)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)
imm32_x = rng.integers(-500, 500, (200, 200)).astype(np.int32)
imm32_y = rng.integers(-500, 500, (200, 200)).astype(np.int32)
suite.measure(
    case="dot_int32_200",
    params={"n": 200, "expect": "exact f64 BLAS round-trip"},
    baseline=("stock_dot", lambda x=imm32_x, y=imm32_y: STOCK_DOT(x, y)),
    candidates={"pyoverdrive": lambda x=imm32_x, y=imm32_y: np.dot(x, y)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)
lsb3 = rng.standard_normal((5_000, 3, 3))
lsb_b = rng.standard_normal((5_000, 3, 1))
suite.measure(
    case="det_3x3_batch5000",
    params={"batch": 5_000, "expect": "cofactor closed form"},
    baseline=("stock_det", lambda m=lsb3: STOCK_DET(m)),
    candidates={"pyoverdrive": lambda m=lsb3: np.linalg.det(m)},
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-9, atol=1e-12)),
    samples=3 if SMOKE else 7,
)
lsb2 = rng.standard_normal((5_000, 2, 2))
suite.measure(
    case="slogdet_2x2_batch5000",
    params={"batch": 5_000, "expect": "sign/log of closed det"},
    baseline=("stock_slogdet", lambda m=lsb2: STOCK_SLOGDET(m)),
    candidates={"pyoverdrive": lambda m=lsb2: np.linalg.slogdet(m)},
    check=lambda c, b: bool(np.array_equal(c[0], b[0]))
    and bool(np.allclose(c[1], b[1], rtol=1e-9, atol=1e-12)),
    samples=3 if SMOKE else 7,
)
suite.measure(
    case="solve_3x3_batch5000",
    params={"batch": 5_000, "expect": "Cramer closed form"},
    baseline=("stock_solve", lambda m=lsb3, b=lsb_b: STOCK_SOLVE(m, b)),
    candidates={"pyoverdrive": lambda m=lsb3, b=lsb_b: np.linalg.solve(m, b)},
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-8, atol=1e-12)),
    samples=3 if SMOKE else 7,
)
lsb_sing = rng.standard_normal((5_000, 3, 3))
lsb_sing[7, -1, :] = lsb_sing[7, 0, :]  # one exactly singular matrix
suite.measure(
    case="det_singular_stack_guard",
    params={"batch": 5_000, "expect": "fallback: conditioning guard refuses"},
    baseline=("stock_det", lambda m=lsb_sing: STOCK_DET(m)),
    candidates={"pyoverdrive": lambda m=lsb_sing: np.linalg.det(m)},
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-9, atol=1e-12)),
    samples=3 if SMOKE else 7,
)
ntn = rng.standard_normal(1_000_000 if not SMOKE else 20_000)
ntn[rng.random(ntn.size) < 0.01] = np.nan
suite.measure(
    case=f"nan_to_num_n{ntn.size}",
    params={"n": int(ntn.size), "expect": "copy + copyto masks"},
    baseline=("stock_nan_to_num", lambda a=ntn: STOCK_NAN_TO_NUM(a)),
    candidates={"pyoverdrive": lambda a=ntn: np.nan_to_num(a)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)

# --- batch 7: cholesky/eigvalsh-3x3 closed forms, einsum chain, ---------------
# --- nan_to_num overrides (OPP-000047/48/49 + OPP-000046 extension) -----------
ch3_n = 3_000 if not SMOKE else 1_200  # mid-window cell (3x3 uncapped since batch 9)
ch3 = rng.standard_normal((ch3_n, 3, 3))
ch3 = np.ascontiguousarray(ch3 @ np.swapaxes(ch3, -1, -2) + 3.0 * np.eye(3))
suite.measure(
    case=f"cholesky_3x3_batch{ch3_n}",
    params={"batch": ch3_n, "expect": "Cholesky-Crout closed form"},
    baseline=("stock_cholesky", lambda a=ch3: STOCK_CHOLESKY(a)),
    candidates={"pyoverdrive": lambda a=ch3: np.linalg.cholesky(a)},
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-9, atol=1e-12)),
    samples=3 if SMOKE else 7,
)
ch2_n = 5_000 if not SMOKE else 1_200
ch2 = rng.standard_normal((ch2_n, 2, 2))
ch2 = np.ascontiguousarray(ch2 @ np.swapaxes(ch2, -1, -2) + 2.0 * np.eye(2))
suite.measure(
    case=f"cholesky_2x2_batch{ch2_n}",
    params={"batch": ch2_n, "expect": "Cholesky-Crout closed form"},
    baseline=("stock_cholesky", lambda a=ch2: STOCK_CHOLESKY(a)),
    candidates={"pyoverdrive": lambda a=ch2: np.linalg.cholesky(a)},
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-9, atol=1e-12)),
    samples=3 if SMOKE else 7,
)
ev3_n = 10_000 if not SMOKE else 500
ev3 = rng.standard_normal((ev3_n, 3, 3))
ev3 = np.ascontiguousarray((ev3 + np.swapaxes(ev3, -1, -2)) / 2.0)
suite.measure(
    case=f"eigvalsh_3x3_batch{ev3_n}",
    params={"batch": ev3_n, "expect": "trig closed form"},
    baseline=("stock_eigvalsh", lambda a=ev3: STOCK_EIGVALSH(a)),
    candidates={"pyoverdrive": lambda a=ev3: np.linalg.eigvalsh(a)},
    check=lambda c, b: c.dtype == b.dtype
    and bool((np.abs(c - b) <= 1e-9 * np.maximum(1.0, np.abs(b).max(axis=-1, keepdims=True))).all()),
    samples=3 if SMOKE else 7,
)
ec_n = 64 if not SMOKE else 32
ec_x = rng.standard_normal((ec_n, ec_n))
ec_y = rng.standard_normal((ec_n, ec_n))
ec_z = rng.standard_normal((ec_n, ec_n))
suite.measure(
    case=f"einsum_chain_3op_n{ec_n}",
    params={"n": ec_n, "expect": "optimize=True routing for the 3-operand chain"},
    baseline=(
        "stock_einsum",
        lambda x=ec_x, y=ec_y, z=ec_z: STOCK_EINSUM("ij,jk,kl->il", x, y, z),
    ),
    candidates={
        "pyoverdrive": lambda x=ec_x, y=ec_y, z=ec_z: np.einsum("ij,jk,kl->il", x, y, z)
    },
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-9, atol=1e-8)),
    samples=3 if SMOKE else 7,
)
ntn_kw = rng.standard_normal(1_000_000 if not SMOKE else 20_000)
ntn_kw[rng.random(ntn_kw.size) < 0.01] = np.nan
ntn_kw[rng.random(ntn_kw.size) < 0.001] = np.inf
suite.measure(
    case=f"nan_to_num_overrides_n{ntn_kw.size}",
    params={"n": int(ntn_kw.size), "expect": "copy + copyto masks, override fills"},
    baseline=(
        "stock_nan_to_num",
        lambda a=ntn_kw: STOCK_NAN_TO_NUM(a, nan=1.5, posinf=100.0),
    ),
    candidates={
        "pyoverdrive": lambda a=ntn_kw: np.nan_to_num(a, nan=1.5, posinf=100.0)
    },
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)

# --- batch 8: hist1d uniform, interp uniform-grid, take-out -------------------
ip_nq = 100_000 if not SMOKE else 20_000
ip_xp = np.linspace(0.0, 1.0, 1_000)
ip_fp = rng.standard_normal(1_000)
ip_x = rng.uniform(-0.1, 1.1, ip_nq)
suite.measure(
    case=f"interp_uniform_nq{ip_nq}",
    params={"nq": ip_nq, "grid": 1_000, "expect": "direct-index lerp, no bisection"},
    baseline=("stock_interp", lambda x=ip_x, xp=ip_xp, fp=ip_fp: STOCK_INTERP(x, xp, fp)),
    candidates={"pyoverdrive": lambda x=ip_x, xp=ip_xp, fp=ip_fp: np.interp(x, xp, fp)},
    check=lambda c, b: bool(
        (np.abs(c - b) <= 1e-9 * np.maximum(1.0, np.abs(b).max())).all()
    ),
    samples=3 if SMOKE else 7,
)
tk_n = 1_000_000 if not SMOKE else 50_000
tk_a = rng.standard_normal(tk_n)
tk_idx = rng.integers(0, tk_n, tk_n // 2).astype(np.intp)
# separate out buffers: sharing one would make the check compare a
# buffer with itself
tk_out_s = np.empty(tk_idx.size)
tk_out_c = np.empty(tk_idx.size)
suite.measure(
    case=f"take_out_n{tk_idx.size}",
    params={"gathered": int(tk_idx.size), "expect": "fancy-index gather + assign"},
    baseline=("stock_take", lambda a=tk_a, i=tk_idx, o=tk_out_s: STOCK_TAKE(a, i, out=o)),
    candidates={"pyoverdrive": lambda a=tk_a, i=tk_idx, o=tk_out_c: np.take(a, i, out=o)},
    check=lambda c, b: bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)

# --- batch 9: fused cholesky large-batch, eigvalsh split, einsum ellipsis -----
ch3b_n = 30_000 if not SMOKE else 5_000  # the regime the old 3x3 cap forfeited
ch3b = rng.standard_normal((ch3b_n, 3, 3))
ch3b = np.ascontiguousarray(ch3b @ np.swapaxes(ch3b, -1, -2) + 3.0 * np.eye(3))
suite.measure(
    case=f"cholesky_3x3_batch{ch3b_n}",
    params={"batch": ch3b_n, "expect": "fused chunked Cholesky-Crout, capless window"},
    baseline=("stock_cholesky", lambda a=ch3b: STOCK_CHOLESKY(a)),
    candidates={"pyoverdrive": lambda a=ch3b: np.linalg.cholesky(a)},
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-9, atol=1e-12)),
    samples=3 if SMOKE else 7,
)
ev3s_n = 10_000 if not SMOKE else 2_000
ev3s = rng.standard_normal((ev3s_n, 3, 3))
ev3s = np.ascontiguousarray((ev3s + np.swapaxes(ev3s, -1, -2)) / 2.0)
ev3s[rng.choice(ev3s_n, ev3s_n // 100, replace=False)] = np.diag([1.0, 1.0, 5.0])
suite.measure(
    case=f"eigvalsh_3x3_split_batch{ev3s_n}_degen1pct",
    params={
        "batch": ev3s_n,
        "degen_frac": 0.01,
        "expect": "trig closed form + stock split for the degenerate 1%",
    },
    baseline=("stock_eigvalsh", lambda a=ev3s: STOCK_EIGVALSH(a)),
    candidates={"pyoverdrive": lambda a=ev3s: np.linalg.eigvalsh(a)},
    check=lambda c, b: c.dtype == b.dtype
    and bool((np.abs(c - b) <= 1e-9 * np.maximum(1.0, np.abs(b).max(axis=-1, keepdims=True))).all()),
    samples=3 if SMOKE else 7,
)
ee_b, ee_n = (64, 32) if not SMOKE else (40, 16)  # smoke keeps min-size >= the 2op floor
ee_x = rng.standard_normal((ee_b, ee_n, ee_n))
ee_y = rng.standard_normal((ee_b, ee_n, ee_n))
ee_z = rng.standard_normal((ee_b, ee_n, ee_n))
suite.measure(
    case=f"einsum_ellipsis_2op_B{ee_b}_n{ee_n}",
    params={"B": ee_b, "n": ee_n, "expect": "optimize=True routing, ellipsis spelling"},
    baseline=(
        "stock_einsum",
        lambda x=ee_x, y=ee_y: STOCK_EINSUM("...ij,...jk->...ik", x, y),
    ),
    candidates={
        "pyoverdrive": lambda x=ee_x, y=ee_y: np.einsum("...ij,...jk->...ik", x, y)
    },
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-9, atol=1e-8)),
    samples=3 if SMOKE else 7,
)
suite.measure(
    case=f"einsum_ellipsis_chain_B{ee_b}_n{ee_n}",
    params={"B": ee_b, "n": ee_n, "expect": "optimize=True routing, ellipsis chain"},
    baseline=(
        "stock_einsum",
        lambda x=ee_x, y=ee_y, z=ee_z: STOCK_EINSUM("...ij,...jk,...kl->...il", x, y, z),
    ),
    candidates={
        "pyoverdrive": lambda x=ee_x, y=ee_y, z=ee_z: np.einsum(
            "...ij,...jk,...kl->...il", x, y, z
        )
    },
    check=lambda c, b: bool(np.allclose(c, b, rtol=1e-9, atol=1e-8)),
    samples=3 if SMOKE else 7,
)

# --- batch 10: qr small-batch Householder closed form -------------------------
qr_n = 10_000 if not SMOKE else 1_000
qr_a = np.ascontiguousarray(rng.standard_normal((qr_n, 3, 3)))
suite.measure(
    case=f"qr_3x3_batch{qr_n}_reduced",
    params={"batch": qr_n, "mode": "reduced", "expect": "unrolled Householder, Q and R"},
    baseline=("stock_qr", lambda a=qr_a: STOCK_QR(a)),
    candidates={"pyoverdrive": lambda a=qr_a: np.linalg.qr(a)},
    check=lambda c, b: type(c) is type(b)
    and bool(np.allclose(c.Q, b.Q, rtol=1e-9, atol=1e-9))
    and bool(np.allclose(c.R, b.R, rtol=1e-9, atol=1e-9 * max(1.0, float(np.abs(b.R).max())))),
    samples=3 if SMOKE else 7,
)
qr_a2 = np.ascontiguousarray(rng.standard_normal((qr_n, 2, 2)))
suite.measure(
    case=f"qr_2x2_batch{qr_n}_r",
    params={"batch": qr_n, "mode": "r", "expect": "unrolled Householder, R only"},
    baseline=("stock_qr", lambda a=qr_a2: STOCK_QR(a, mode="r")),
    candidates={"pyoverdrive": lambda a=qr_a2: np.linalg.qr(a, mode="r")},
    check=lambda c, b: type(c) is np.ndarray
    and bool(np.allclose(c, b, rtol=1e-9, atol=1e-9 * max(1.0, float(np.abs(b).max())))),
    samples=3 if SMOKE else 7,
)

# --- batch 11: the Python-loop interceptions ---------------------------------
aaa_n = 20_000 if not SMOKE else 2_000
aaa = rng.standard_normal((aaa_n, 50))
suite.measure(
    case=f"apply_along_axis_mean_slices{aaa_n}",
    params={"slices": aaa_n, "expect": "axis= reduction instead of a Python loop"},
    baseline=("stock_apply_along_axis", lambda a=aaa: STOCK_APPLY_ALONG_AXIS(np.mean, -1, a)),
    candidates={"pyoverdrive": lambda a=aaa: np.apply_along_axis(np.mean, -1, a)},
    check=lambda c, b: c.dtype == b.dtype and bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)
vec_n = 1_000_000 if not SMOKE else 50_000
vec_x = np.abs(rng.standard_normal(vec_n)) + 1e-6
suite.measure(
    case=f"vectorize_sqrt_n{vec_n}",
    params={"n": vec_n, "expect": "wrapped ufunc called directly"},
    baseline=("stock_vectorize", lambda x=vec_x: STOCK_VECTORIZE(np.sqrt)(x)),
    candidates={"pyoverdrive": lambda x=vec_x: np.vectorize(np.sqrt)(x)},
    check=lambda c, b: c.dtype == b.dtype and bool(np.array_equal(c, b)),
    samples=3 if SMOKE else 7,
)

# --- batch 12: the singular-value family on small-matrix batches -------------
sv_n = 10_000 if not SMOKE else 1_000
sv_a = np.ascontiguousarray(rng.standard_normal((sv_n, 3, 3)))
sv_a2 = np.ascontiguousarray(rng.standard_normal((sv_n, 2, 2)))
suite.measure(
    case=f"pinv_3x3_batch{sv_n}",
    params={"batch": sv_n, "expect": "adjugate inverse under a conditioning band"},
    baseline=("stock_pinv", lambda a=sv_a: STOCK_PINV(a)),
    candidates={"pyoverdrive": lambda a=sv_a: np.linalg.pinv(a)},
    check=lambda c, b: c.shape == b.shape
    and bool(np.all(np.abs(c - b) <= 1e-9 * max(float(np.abs(b).max()), 1e-300))),
    samples=3 if SMOKE else 7,
)
suite.measure(
    case=f"norm2_2x2_batch{sv_n}",
    params={"batch": sv_n, "expect": "largest singular value from the gram closed form"},
    baseline=("stock_norm", lambda a=sv_a2: STOCK_NORM(a, ord=2, axis=(-2, -1))),
    candidates={"pyoverdrive": lambda a=sv_a2: np.linalg.norm(a, ord=2, axis=(-2, -1))},
    check=lambda c, b: c.shape == b.shape
    and bool(np.all(np.abs(c - b) <= 1e-12 * np.maximum(np.abs(b), 1e-300))),
    samples=3 if SMOKE else 7,
)
suite.measure(
    case=f"svdvals_2x2_batch{sv_n}",
    params={"batch": sv_n, "expect": "singular values from the gram closed form"},
    baseline=("stock_svd", lambda a=sv_a2: STOCK_SVD(a, compute_uv=False)),
    candidates={"pyoverdrive": lambda a=sv_a2: np.linalg.svd(a, compute_uv=False)},
    check=lambda c, b: c.shape == b.shape
    and bool(np.all(np.abs(c - b) <= 1e-9 * np.maximum(b[..., :1], 1e-300))),
    samples=3 if SMOKE else 7,
)

pyoverdrive.disable()
if not SMOKE:
    suite.save()
