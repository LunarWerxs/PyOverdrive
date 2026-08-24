"""fftconvolve calibration: (n, m) shape sweep x padding policy x dtype.

The OPP-000016 reproducer measured only EQUAL-length self-convolution
(n = m in {1000, 5000, 20000}: 2.3x at the floor, 1518x at the top). What
it left unmeasured is exactly what a dispatch predicate needs: where the
win dies as the operands become unequal (naive work is n*m, FFT work is
(n+m) log(n+m), so a thin kernel over a long signal is the FFT's worst
regime), and whether power-of-two padding (the reproducer's choice, up to
2x oversize) loses to 5-smooth padding (<= ~1.06x oversize typical) badly
enough to matter. This battery sweeps both, on convolve and correlate,
for float64 / int64 / int32, and feeds the SUPPORTED table in
src/pyoverdrive/fastpaths/fftconvolve.py.

Checks: float64 np.allclose(rtol=1e-6, atol=1e-6) as in the reproducer
(the tight per-element error contract lives in the differential battery,
not here); integer dtypes exact equality after rounding.

Result JSON: benchmarks/results/FFTCONV-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_fftconvolve_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 1858
SMOKE = "--smoke" in sys.argv


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def next_smooth5(n):
    """Smallest 2**a * 3**b * 5**c >= n (scipy's next_fast_len idea, reimplemented)."""
    best = next_pow2(n)
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            q = -(-n // p35)
            cand = p35 << max(0, (q - 1).bit_length())
            if n <= cand < best:
                best = cand
            p35 *= 3
        p5 *= 5
    return best


def fft_full(a, v, pad):
    n = a.size + v.size - 1
    fft_len = pad(n)
    out = np.fft.irfft(np.fft.rfft(a, fft_len) * np.fft.rfft(v, fft_len), fft_len)[:n]
    if a.dtype.kind == "f":
        return out
    return np.rint(out).astype(a.dtype)


def f64_check(c, b):
    return np.allclose(c, b, rtol=1e-6, atol=1e-6)


# (n, m) pairs: product floor probes at 1e6, thin corners at fixed product,
# and the scaling region above. Below-floor witnesses stay in to show the
# loss, not just the win.
if SMOKE:
    F64_SHAPES = [(200, 100)]
    INT_SHAPES = [(200, 100)]
    SAMPLES = 3
else:
    F64_SHAPES = [
        (300, 300),
        (500, 500),
        (1000, 1000),
        (2000, 500),
        (4000, 250),
        (10000, 100),
        (10000, 300),
        (10000, 1000),
        (20000, 1000),
        (100000, 100),
        (100000, 300),
        (100000, 1000),
    ]
    INT_SHAPES = [(1000, 1000), (10000, 100), (10000, 1000), (100000, 1000)]
    SAMPLES = 5

suite = BenchSuite("FFTCONV-CAL", "FFT full-mode convolve/correlate vs naive, shape sweep")
rng = np.random.default_rng(SEED)


def make(dtype, size):
    if np.dtype(dtype).kind == "f":
        return rng.standard_normal(size)
    return rng.integers(-100, 101, size=size, dtype=dtype)


def cases(op_name, stock, a, v, check, samples):
    suite.measure(
        case=f"{op_name}_{np.dtype(a.dtype).name}_n{a.size}_m{v.size}",
        params={"dtype": np.dtype(a.dtype).name, "n": a.size, "m": v.size,
                "product": a.size * v.size, "mode": "full"},
        baseline=(f"numpy.{op_name}", lambda: stock(a, v, "full")),
        candidates={
            "fft_pow2": lambda: fft_full(a, v, next_pow2) if op_name == "convolve"
            else fft_full(a, v[::-1], next_pow2),
            "fft_smooth5": lambda: fft_full(a, v, next_smooth5) if op_name == "convolve"
            else fft_full(a, v[::-1], next_smooth5),
        },
        check=check,
        samples=samples,
    )


for n, m in F64_SHAPES:
    a, v = make(np.float64, n), make(np.float64, m)
    cases("convolve", np.convolve, a, v, f64_check, SAMPLES)
    cases("correlate", np.correlate, a, v, f64_check, SAMPLES)

for dtype in (np.int64, np.int32):
    for n, m in INT_SHAPES:
        a, v = make(dtype, n), make(dtype, m)
        cases("convolve", np.convolve, a, v, np.array_equal, SAMPLES)
# one integer correlate witness (the mechanism is convolve + a free
# reversed view, so per-shape coverage adds battery time, not information)
if not SMOKE:
    a, v = make(np.int64, 10000, ), make(np.int64, 1000)
    cases("correlate", np.correlate, a, v, np.array_equal, SAMPLES)

# --- mode='same' / mode='valid' ---------------------------------------------
# same: stock computes max(M,N) outputs of up to min terms, so its naive
# work is close to full's. valid: stock computes only (max-min+1) outputs,
# so for near-equal lengths it does almost NO work and the FFT (which
# always pays the full transform) should lose - the equal-length valid
# case below is the expected-loss witness that pins the guard.


def fft_mode(a, v, mode, correlate=False):
    if correlate:
        v = v[::-1]
    full = fft_full(a, v, next_smooth5)
    mn, mx = min(a.size, v.size), max(a.size, v.size)
    if mode == "same":
        # correlate with the second operand longer: numpy's internal swap
        # flips the centering for even min-lengths (probed exhaustively)
        start = (mn // 2) if (correlate and a.size < v.size) else (mn - 1) // 2
        return full[start : start + mx]
    return full[mn - 1 : mx]


if not SMOKE:
    MODE_SHAPES = {
        "same": [(1000, 1000), (2000, 2000), (3000, 1000), (10000, 1000), (20000, 1000), (100000, 1000)],
        "valid": [(10000, 9999), (3000, 2000), (5000, 4000), (10000, 1000), (100000, 1000)],
    }
    for mode, shapes in MODE_SHAPES.items():
        for n, m in shapes:
            a, v = make(np.float64, n), make(np.float64, m)
            naive_work = (max(n, m) - min(n, m) + 1) * min(n, m) if mode == "valid" else n * m
            for op_name, corr in (("convolve", False), ("correlate", True)):
                stock = np.correlate if corr else np.convolve
                suite.measure(
                    case=f"{op_name}_{mode}_float64_n{n}_m{m}",
                    params={"dtype": "float64", "n": n, "m": m, "mode": mode,
                            "naive_work": naive_work},
                    baseline=(f"numpy.{op_name}", lambda s=stock, a=a, v=v, mode=mode: s(a, v, mode)),
                    candidates={
                        "fft_smooth5_sliced": lambda a=a, v=v, mode=mode, corr=corr: fft_mode(a, v, mode, corr),
                    },
                    check=f64_check,
                    samples=SAMPLES,
                )
    ai, vi = make(np.int64, 10000), make(np.int64, 1000)
    for mode in ("same", "valid"):
        suite.measure(
            case=f"convolve_{mode}_int64_n10000_m1000",
            params={"dtype": "int64", "n": 10000, "m": 1000, "mode": mode},
            baseline=("numpy.convolve", lambda a=ai, v=vi, mode=mode: np.convolve(a, v, mode)),
            candidates={
                "fft_smooth5_sliced": lambda a=ai, v=vi, mode=mode: np.rint(
                    fft_mode(a.astype(np.float64), v.astype(np.float64), mode)
                ).astype(np.int64),
            },
            check=np.array_equal,
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
