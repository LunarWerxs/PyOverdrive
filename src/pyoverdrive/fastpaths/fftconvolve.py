"""Fast paths: numpy.convolve / numpy.correlate (full/same/valid) via rfft/irfft.

Provenance (OPP-000016): numpy/numpy#1858 (trac #1260, 2009, still open)
reports full-mode np.correlate/np.convolve using naive O(n*m) direct
summation, "takes around an hour" at n=1e6 where MATLAB's FFT-based xcorr
needs 0.35s (a >=171x floor; the numpy run was killed unfinished). The FFT
route is commenter endolith's sketch; maintainer cournape's own proposal in
the thread is a size-based heuristic between naive and FFT, which is
exactly what this dispatch implements. Dyno reproduced 2.3x at n=1000 up to
1518x at n=20000 float64 (benchmarks/results/OPP-000016/), and the shipped
thresholds come from the FFTCONV-CAL battery
(benchmarks/micro/bench_fftconvolve_calibration.py,
benchmarks/results/FFTCONV-CAL/).

Correctness contract, shared by both registered paths:

- Applies to mode 'full', 'same', and 'valid', passed positionally, by
  keyword, or omitted where a mode is the default (convolve defaults to
  'full', correlate to 'valid'). Exactly those strings: stock rejects
  abbreviations (numpy 2.x: "Use one of 'valid', 'same', or 'full'").
  'same' and 'valid' are centered/trimmed slices of the full output, with
  the slice formulas verified against stock exhaustively across both
  length orders and parities - including numpy's internal operand swap
  for correlate, which flips 'same' centering when the second operand is
  longer and the shorter length is even.
- Both operands must be plain 1-D ndarrays of the SAME dtype, from the
  SUPPORTED table, with min(n, m) >= MIN_LEN and the MODE's naive work
  estimate above its measured floor. The FFT always pays the full
  transform, but stock's cost depends on the mode: full/same perform
  ~n*m multiply-adds, valid only (max-min+1)*min - a near-equal-length
  valid call is nearly free on stock (measured 0.01-0.02x for the FFT
  there) and must stay on it, which is what the per-mode work floors in
  _MODE_WORK_FLOOR encode.
- float64: results are numerically equal to stock, not bit-identical (the
  two algorithms accumulate rounding in different orders; pv's estimate in
  the thread is ~1e-11 relative error, and the differential battery pins
  |got - stock| <= 1e-12 * ||a||_2 * ||v||_2 * log2(fft_len), far below
  that). Inputs containing NaN/Inf are REFUSED (abs-max scan in the
  predicate), and so are finite inputs that could overflow either the
  RESULT (max|a| * max|v| * min(n, m) >= 1e300; hypothesis net,
  ~8e213-scale draws) or a single operand's TRANSFORM
  ((n + m) * max(|a|, |v|) >= 1e300: rfft coefficients are sums of up to
  fft_len terms, and an all-zero other operand then turns 0 * inf into
  NaN everywhere - the same net found this second hole on the Linux
  sweep). Naive summation keeps all such damage local to the lags that
  touch it, while an FFT smears it, so every case stays on stock. The
  scan is O(n + m) against an O(n*m) win regime.
- int64/int32: computed in float64 and rounded back (per pv: FFT
  convolution needs the cast; NumPy has no integer FFT). The result is
  bit-identical to stock, guaranteed by the predicate bound
  max|a| * max|v| * min(n, m) <= min(2**52 - 1, dtype max), the integer
  exactness condition scipy's choose_conv_method has shipped for years
  (reimplemented here, no scipy code). Under it, every intermediate and
  final value is an exactly representable float64 integer and stock's own
  integer accumulator cannot have wrapped, so rounding is exact. Above it
  (or when stock would overflow and wrap) the call stays on stock,
  wrap-around semantics and all.
- correlate(a, v, 'full') is computed as convolve(a, v[::-1], 'full'):
  exact for real input regardless of which operand is longer (verified
  against stock's internal swap in the differential battery; conjugation
  is a no-op on reals, and complex dtypes are not in SUPPORTED).

Comparison mode: numeric for float64, bit-identical for integers (spec
section 9; the provenance record carries the per-dtype split). Kill
switches: PYOVERDRIVE_DISABLE=fftconvolve / fftcorrelate.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..dispatcher.gearbox import FastPath

_PROVENANCE = {
    "opportunity": "OPP-000016",
    "source": "https://github.com/numpy/numpy/issues/1858",
    "license": (
        "FFT route from the issue thread (endolith), integer exactness bound "
        "from scipy choose_conv_method; both reimplemented, no third-party code"
    ),
    "comparison_mode": "numeric (float64 rtol ~1e-9; integer dtypes bit-identical)",
}


class _Spec(NamedTuple):
    min_len: int  # shorter operand must have at least this many elements
    product: int  # n * m must be at least this (naive work estimate)
    int_bound: int  # max|a| * max|v| * min(n, m) ceiling; 0 = float path


# CALIBRATION (fp 8f8198d9abab, benchmarks/results/FFTCONV-CAL/, 2026-08-23,
# 11-14% foreign load, below the contended threshold). 5-smooth padding
# ships because power-of-two padding can oversize the transform by up to 2x
# and the battery shows that gap deciding real cases: 20000x1000 float64 is
# 4.22x smooth5 vs 1.76x pow2, and 4000x250 WINS at 1.83x smooth5 while
# pow2 loses it (0.90x). Both floors are the measured edge of the winning
# region, applied to all three dtypes:
# - min_len 1000: thin kernels lose or go marginal however large the
#   product gets (m=100: 0.37-0.92x everywhere; m=300 wins 2.1x at
#   n=10000 but decays to 1.12x by n=100000, below the 1.3x min-win).
# - product 1e6: (500,500) is only 0.97-1.20x; (1000,1000) is the first
#   clean win.
# Measured with both floors satisfied: float64 2.3x (1000x1000) to 5.1x
# (10000x1000, and 2.9x at 100000x1000); int64 5.9-13.4x; int32 6.7-18.0x.
# (2000,500), (4000,250) and (10000,300) also win 1.5-2.2x but sit outside
# the simple two-floor rule and stay on stock; a finer work-ratio predicate
# is a possible future refinement, with its own battery.
_INT_EXACT = 2**52 - 1  # float64 exact-integer ceiling (scipy's nmant bound)
SUPPORTED: dict[np.dtype, _Spec] = {
    np.dtype(np.float64): _Spec(1000, 1_000_000, 0),
    np.dtype(np.int64): _Spec(1000, 1_000_000, _INT_EXACT),
    np.dtype(np.int32): _Spec(1000, 1_000_000, min(_INT_EXACT, 2**31 - 1)),
}
_MIN_LEN_FLOOR = min(s.min_len for s in SUPPORTED.values())

# Per-mode naive-work floors (same battery, mode section, 56-86% load so
# every ratio is understated). 'same': (1000,1000) work 1e6 straddled the
# 1.3x min-win across three runs (1.09-1.51x), (3000,1000) work 3e6 is the
# first clean edge (2.88-2.93x; 3.3-4.9x above it; int64 14.3x). 'valid':
# (3000,2000) work 2e6 is the edge (1.57-1.70x; 2.3-3.3x above; int64
# 12.7x), and the near-equal-length witness (10000,9999), work 2e4, loses
# 0.01-0.02x, which the work model itself excludes.
_MODE_WORK_FLOOR = {"full": 1_000_000, "same": 3_000_000, "valid": 2_000_000}


def _next_smooth5(n: int) -> int:
    """Smallest 2**a * 3**b * 5**c >= n (scipy's next_fast_len idea, reimplemented).

    pocketfft is fast on 2/3/5-smooth lengths; padding to the next power of
    two instead can oversize the transform by up to 2x, which the FFTCONV-CAL
    battery measured as the difference between winning and losing on
    unequal-length shapes.
    """
    best = 1 << (n - 1).bit_length()
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


def _fft_convolve_f64(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    n = a.size + v.size - 1
    fft_len = _next_smooth5(n)
    fa = np.fft.rfft(a, fft_len)
    fb = np.fft.rfft(v, fft_len)
    return np.fft.irfft(fa * fb, fft_len)[:n]


def _fft_convolve(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    if a.dtype.kind == "f":
        return _fft_convolve_f64(a, v)
    res = _fft_convolve_f64(a.astype(np.float64), v.astype(np.float64))
    return np.rint(res).astype(a.dtype)


_MODES = ("full", "same", "valid")


def _operands(args: tuple, kwargs: dict, default_mode: str):
    """Return (a, v, mode) for an admissible two-operand call, else None."""
    if len(args) == 3:
        if kwargs:
            return None
        a, v, mode = args
    elif len(args) == 2:
        if kwargs and (len(kwargs) != 1 or "mode" not in kwargs):
            return None
        a, v = args
        mode = kwargs.get("mode", default_mode)
    else:
        return None
    if mode not in _MODES:
        return None
    return a, v, mode


def _admissible(a, v, mode) -> bool:
    if type(a) is not np.ndarray or type(v) is not np.ndarray:
        return False
    if a.ndim != 1 or v.ndim != 1 or a.dtype != v.dtype:
        return False
    lo = a.size if a.size < v.size else v.size
    if lo < _MIN_LEN_FLOOR:  # cheapest refusal for the common small call
        return False
    spec = SUPPORTED.get(a.dtype)
    if spec is None or lo < spec.min_len:
        return False
    # the FFT always pays the full transform, so the guard compares against
    # what stock actually computes for the MODE: full/same do ~n*m
    # multiply-adds, valid only (max-min+1)*min - near-equal-length valid
    # calls are nearly free on stock and must stay there (measured losing)
    hi = a.size if a.size >= v.size else v.size
    naive_work = (hi - lo + 1) * lo if mode == "valid" else a.size * v.size
    if naive_work < _MODE_WORK_FLOOR[mode]:
        return False
    if spec.int_bound == 0:
        # one abs-max scan per operand covers four refusals at once: NaN
        # (nan < inf is False), inf, RESULT overflow, and TRANSFORM
        # overflow. Finite inputs near sqrt(float64 max) overflow during
        # convolution (hypothesis net, ~8e213 draws), and - the second,
        # subtler hole the same net found on a later platform sweep - a
        # SINGLE operand near 1e305 overflows its own rfft (coefficients
        # are sums of up to fft_len terms), after which 0 * inf = NaN
        # smears every lag even though the product bound passed (a was
        # all zeros: 0 * huge * n < 1e300). Naive summation keeps all of
        # these local; every one must stay on stock.
        amax = float(np.abs(a).max())
        vmax = float(np.abs(v).max())
        if not (amax < np.inf and vmax < np.inf):
            return False
        peak = amax if amax >= vmax else vmax
        if (a.size + v.size) * peak >= 1e300:
            return False
        return amax * vmax * lo < 1e300
    amax = max(-int(a.min()), int(a.max()))
    vmax = max(-int(v.min()), int(v.max()))
    return amax * vmax * lo <= spec.int_bound


def _conv_applicable(args: tuple, kwargs: dict) -> bool:
    ops = _operands(args, kwargs, "full")
    return ops is not None and _admissible(*ops)


def _corr_applicable(args: tuple, kwargs: dict) -> bool:
    ops = _operands(args, kwargs, "valid")
    return ops is not None and _admissible(*ops)


def _slice_mode(full, n_a, n_v, mode, correlate):
    if mode == "full":
        return full
    mn = n_a if n_a < n_v else n_v
    mx = n_a if n_a >= n_v else n_v
    if mode == "valid":
        return full[mn - 1 : mx]
    # 'same': centered slice of the full output. For correlate with the
    # SECOND operand longer, numpy computes the swapped correlation and
    # reverses it, which flips the centering for even min-lengths; probed
    # exhaustively across both orders and parities (see the differential
    # battery), the start index is:
    start = (mn // 2) if (correlate and n_a < n_v) else (mn - 1) // 2
    return full[start : start + mx]


def _convolve_run(a, v, mode="full"):
    return _slice_mode(_fft_convolve(a, v), a.size, v.size, mode, False)


def _correlate_run(a, v, mode="valid"):
    return _slice_mode(_fft_convolve(a, v[::-1]), a.size, v.size, mode, True)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="fftconvolve",
            op="numpy.convolve",
            applicable=_conv_applicable,
            run=_convolve_run,
            provenance=_PROVENANCE,
        )
    )
    gearbox.register(
        FastPath(
            name="fftcorrelate",
            op="numpy.correlate",
            applicable=_corr_applicable,
            run=_correlate_run,
            provenance=_PROVENANCE,
        )
    )
