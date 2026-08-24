"""OPP-000016: FFT-based fast path for full-mode np.correlate / np.convolve
on large 1-D real arrays.

numpy/numpy#1858 (trac #1260) reports that np.correlate(x, x, 'full') and
np.convolve(x, x, 'full') use NumPy's naive, direct-summation algorithm,
roughly O(n*m) work, and that this "takes around an hour" at n = 1e6, while
MATLAB's xcorr (which uses FFT internally, per commenter josef-pkt) finishes
the same size in about 0.35s -- a reported >=171.4x lower bound (derived,
not measured: the numpy run was killed before completing). The default
mode='valid' correlate(x, x) is NOT slow (it returns a single value); the
complaint is specifically full-mode, all-lags convolution/correlation.

This reproducer measures:

  - baseline: np.correlate(a, a, mode='full') and np.convolve(a, a,
    mode='full') exactly as a user would call them today (naive direct
    summation, O(n^2) for equal-length self-correlation/convolution).
  - candidate "fft_rfft_irfft": an FFT-based full convolution built from
    numpy.fft.rfft/irfft as sketched by commenter endolith and named in
    OPP-000016.md step 1, zero-padded to the next power of two so the
    transform lands on a fast FFT length, then truncated to the true
    full-mode output length. np.correlate(a, b, 'full') is computed via
    the standard identity correlate(a, b) == convolve(a, b[::-1]) for real
    1-D input (no conjugation needed; inputs here are real).

Sizes and dtypes: the report (step 1) names sizes [1_000, 10_000, 100_000,
1_000_000] and dtypes float64/int64. The full sweep is NOT reproduced here:
naive full-mode correlate/convolve is exactly the O(n^2) operation this
issue is about, so a single size=100_000 self-correlation baseline call
already costs on the order of 1e10 multiply-adds (multiple seconds), and
size=1_000_000 is the ~hour-long call the original report was killed for.
Running that across 2 operations x 2 dtypes x (warmup + 5-11 samples) would
blow the ~90s house budget for the whole battery by orders of magnitude, so
the swept sizes are shrunk to [1_000, 5_000, 20_000] -- large enough for an
O(n^2) baseline to visibly dominate an O(n log n) FFT candidate, small
enough (worst case ~4e8 multiply-adds at n=20_000) to keep the full
non-smoke battery to low tens of seconds on a fast machine. This follows
the report's spirit (sweep sizes, show the crossover) rather than its exact
numbers.

Correctness:
  - float64: np.allclose(candidate, baseline, rtol=1e-6, atol=1e-6). The
    naive baseline is not itself bit-exact against the FFT candidate (the
    two accumulate float64 rounding in different orders), and pv's own
    estimate in the thread is "about 1e-11" relative error for FFT
    convolution at moderate N; rtol/atol of 1e-6 leaves roughly five
    orders of magnitude of headroom above that estimate to absorb the
    largest swept size's slightly worse conditioning, while still failing
    hard (relative error O(1)) on a genuinely broken implementation.
  - int64: per pv, "you cannot perform FFT convolution with integers ...
    cast to complex [is] probably always needed". The candidate casts to
    float64 (rfft/irfft only need real float, not full complex), computes,
    then rounds to the nearest integer (np.rint) and casts back to int64.
    Inputs are drawn from a bounded range ([-100, 100]) so worst-case
    magnitudes (~n * 100 * 100) stay far below both int64 range and the
    float64 rounding error's distance from the nearest half-integer
    boundary, so the check is exact equality (np.array_equal) after
    rounding, not allclose -- if FFT rounding ever pushed a result across
    a rounding boundary at these sizes, that would itself be a
    correctness finding worth surfacing, not something to tolerance away.

Not measured here (out of the report's scope for this reproducer): mode=
'valid' single-lag calls (already fast, per josef-pkt) and object-dtype
arrays (cournape: FFT "does not work for object arrays") -- both must stay
on the naive path in any real fast-path dispatch; this reproducer only
tests the regime where FFT is being proposed as a replacement.
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


def fft_convolve_full(a, b):
    """FFT-based full-mode linear convolution, real 1-D inputs only.

    Zero-pads both operands (via rfft's own n= padding) to the next power
    of two at or above the true full-mode output length, so the transform
    lands on a fast FFT size, then truncates the inverse transform back to
    the true length len(a) + len(b) - 1.
    """
    n = len(a) + len(b) - 1
    fft_len = next_pow2(n)
    fa = np.fft.rfft(a, fft_len)
    fb = np.fft.rfft(b, fft_len)
    return np.fft.irfft(fa * fb, fft_len)[:n]


def fft_correlate_full(a, b):
    """FFT-based full-mode correlation.

    Uses the standard identity correlate(a, b, 'full') == convolve(a,
    b[::-1], 'full') for real 1-D input (numpy's correlate conjugates the
    second argument, which is a no-op on reals).
    """
    return fft_convolve_full(a, b[::-1])


def fft_convolve_full_int(a, b):
    res = fft_convolve_full(a.astype(np.float64), b.astype(np.float64))
    return np.rint(res).astype(np.int64)


def fft_correlate_full_int(a, b):
    res = fft_correlate_full(a.astype(np.float64), b.astype(np.float64))
    return np.rint(res).astype(np.int64)


def allclose_check(cand, base):
    return np.allclose(cand, base, rtol=1e-6, atol=1e-6)


if SMOKE:
    SIZES = [50, 200]
else:
    SIZES = [1_000, 5_000, 20_000]

DTYPES = ["float64", "int64"]

suite = BenchSuite("OPP-000016", "FFT full-mode np.correlate/np.convolve vs naive")
samples = 3 if SMOKE else 7

rng = np.random.default_rng(SEED)

for n in SIZES:
    for dtype in DTYPES:
        if dtype == "float64":
            a = rng.standard_normal(n)
            candidates_corr = {"fft_rfft_irfft": lambda a=a: fft_correlate_full(a, a)}
            candidates_conv = {"fft_rfft_irfft": lambda a=a: fft_convolve_full(a, a)}
            check = allclose_check
        else:
            a = rng.integers(-100, 101, size=n, dtype=np.int64)
            candidates_corr = {
                "fft_rfft_irfft_int": lambda a=a: fft_correlate_full_int(a, a)
            }
            candidates_conv = {
                "fft_rfft_irfft_int": lambda a=a: fft_convolve_full_int(a, a)
            }
            check = np.array_equal

        params = {"dtype": dtype, "n": n, "mode": "full"}

        suite.measure(
            case=f"correlate_full_n{n}_{dtype}",
            params=params,
            baseline=("numpy.correlate", lambda a=a: np.correlate(a, a, mode="full")),
            candidates=candidates_corr,
            check=check,
            samples=samples,
        )

        suite.measure(
            case=f"convolve_full_n{n}_{dtype}",
            params=params,
            baseline=("numpy.convolve", lambda a=a: np.convolve(a, a, mode="full")),
            candidates=candidates_conv,
            check=check,
            samples=samples,
        )

if not SMOKE:
    suite.save()
