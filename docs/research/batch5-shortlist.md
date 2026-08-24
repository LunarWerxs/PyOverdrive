# Batch-5 lead shortlist (mined 2026-08-24)

Sweep: 14 fresh GitHub search angles chosen NOT to overlap batch 3's ten
(that sweep centered on generic slow/slower title words; this one hunts
mechanism classes, especially the batched-small-linalg vein that
OPP-000030's 31x eigvalsh win proved out, plus op families no record
covers). 71 fresh candidates after dedup against the 94 already-seen
issues (corpus sources + batch-3 shortlist). Title triage kept 12 for
thread reading; 4 read first-hand + 8 scored by a thread-reading agent.

Chosen for ingestion (OPP-000035..040):

| issue | OPP | lead | thread evidence |
|---|---|---|---|
| [17166](https://github.com/numpy/numpy/issues/17166) | OPP-000035 | np.linalg.inv on stacked 3x3 via vectorized adjugate | reporter: 79.2 ms vs 371 us (213x DERIVED); independent: 13.4 ms vs 152 us (88x); anti-regime at single matrix; ilayn's singularity objection |
| [14997](https://github.com/numpy/numpy/issues/14997) | OPP-000036 | np.isin on object arrays via Python set | O(n*m) always taken for object dtype; sibling of the shipped 2317x StringDType path; hashability + NaN-identity hazards |
| [18298](https://github.com/numpy/numpy/issues/18298) | OPP-000037 | np.median via partition | 11.2x at n=11/101, 6.9x at 1001 (reporter); seberg: must replicate the [-1] NaN-check partition; overhead-class, small-n |
| [17676](https://github.com/numpy/numpy/issues/17676) | OPP-000038 | histogram2d/dd uniform-bin direct indexing | reporter measured 4-5x with a working patch; edge-value rounding is the correctness minefield |
| [29719](https://github.com/numpy/numpy/issues/29719) | OPP-000039 | searchsorted with Python int keys | ~30000x (12 s / 100 calls vs 0.37 ms); closed by-design upstream, never fixed - a live gap |
| [11136](https://github.com/numpy/numpy/issues/11136) | OPP-000040 | unique(axis=0) int rows via void view | 2018: ~3x view trick; 2026 numbers on numpy 2.3.5 in-thread; int-only (NaN/-0.0 bit-pattern hazard) |

Read-and-declined from this sweep (reasons in one line each):

- 27007 (sort with order=): single-field argsort route breaks tie
  semantics (stock compares remaining fields on equal keys); viable only
  with duplicate detection - parked for a later batch.
- 28592 (matrix 2-norm): the gap is a BLAS thread-count artifact and the
  eigvalsh(A@A.T) trick squares the condition number - declined.
- 13836 (log2 slower than log): log(x)/log(2) is not bit-identical and
  SIMD work likely landed upstream - declined.
- 14761 (hypot): slower BY DESIGN for overflow safety; closed wontfix -
  declined.
- 13001 (np.full): sub-microsecond wrapper overhead, inside dispatch-tax
  noise - declined.
- 6948 (innerproduct to syrk): no measured numbers anywhere in-thread;
  aliasing detection fragile - declined.

The remaining ~59 fresh title-triage rejects live in the sweep artifacts
(scratchpad sweep5/, not committed); the strongest were tax-dominated
small-array complaints or fixed-upstream regressions. Search angles that
produced nothing new: loop-faster-than, pandas-faster, and the
einsum/where/norm clusters (all hits already known from batch 3).
