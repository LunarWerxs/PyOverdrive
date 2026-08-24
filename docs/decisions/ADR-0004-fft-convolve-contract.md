# ADR-0004: The FFT convolve/correlate contract

Date: 2026-08-23 · Status: accepted

Context: OPP-000016 (numpy/numpy#1858) is the corpus headline, 1518x
reproduced, but an FFT route changes semantics naive summation has: results
become approximate, NaNs smear across every output lag, and integers pass
through float64. Shipping it meant deciding, for the first time in the
project, how a fast path may differ from stock and what it must refuse.

## Decisions

1. **A fast path may refuse inputs to keep its claim honest.** Naive
   summation keeps a NaN's damage local to the lags that touch it; an FFT
   smears it everywhere. Rather than weakening the equality claim to
   "except non-finite inputs", the predicate scans both operands with
   `isfinite` and stays on stock when anything non-finite appears. The scan
   is O(n + m) against an O(n * m) win regime; MVP-BASELINE carries a NaN
   guard row proving the refusal (stock-identical output, tax only).

2. **Integers ship bit-identical, floats ship numeric, one family.** Under
   the bound `max|a| * max|v| * min(n, m) <= min(2**52 - 1, dtype max)`
   (scipy `choose_conv_method`'s integer-precision condition, reimplemented)
   every value in the computation is an exactly representable float64
   integer AND stock's own integer accumulator cannot have wrapped, so
   rounding back is exact and wrap-around semantics never diverge. Above
   the bound the call stays on stock, wrap-around and all. float64 claims
   numeric equality with error scaled by operand norms (~1e-12 measured,
   pv's thread estimate is ~1e-11); the provenance string carries the
   per-dtype split and `selfcheck` exercises both paths.

3. **5-smooth padding, not power-of-two.** The reproducer used pow2; the
   FFTCONV-CAL battery showed padding policy deciding real cases:
   20000x1000 float64 is 4.22x with 5-smooth vs 1.76x with pow2, and
   4000x250 flips from a loss (0.90x) to a win (1.83x). pow2 can oversize
   the transform by up to 2x; 5-smooth stays within ~6% typical.

4. **Two floors, deliberately conservative.** min(n, m) >= 1000 and
   n * m >= 1e6. Thin kernels are the FFT's worst regime (naive work n*m,
   FFT work (n+m) log(n+m)): m=100 loses everywhere measured, m=300 decays
   below min-win by n=100000. The simple rule leaves measured 1.5-2.2x wins
   on stock at (2000,500), (4000,250), (10000,300); a work-ratio predicate
   could claim them but needs its own battery, and the battery showed the
   ratio alone does not separate the marginal cases (fixed FFT overhead
   dominates small totals). Every case inside the rule wins >= 2.3x.

5. **mode='full' only.** convolve's default; correlate must ask (its
   default 'valid' is cheap for equal lengths and unmeasured otherwise).
   'same'/'valid' are slices of full-mode plus different alignment rules;
   they stay on stock until measured, same for float32 (its naive-vs-FFT
   tolerance story does not close cleanly at the numeric mode's 1e-4 with
   near-zero edge lags, so no claim is made).

## Consequences

- First family with a comparison mode split by dtype; `_equal` in
  diagnostics needs no change (integers pass the numeric tolerance
  trivially, and the differential battery carries exact equality).
- The correlate identity `correlate(a, v, 'full') == convolve(a, v[::-1],
  'full')` is pinned by asymmetric both-order differential tests, since
  self-correlation (the reproducer's check) is palindromic and would hide
  a reversal bug.
- A user convolving signals containing NaN/Inf sees stock behavior,
  always, by construction.
