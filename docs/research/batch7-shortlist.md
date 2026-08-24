# Batch-7 leads: probe verdicts (2026-08-24)

Source bench: the triage-kept, unprobed list at the end of
docs/research/batch6-shortlist.md. Every lead was probed by one
deterministic dev-box script (scratchpad probe_batch7.py, contended box,
order-of-magnitude only; candidate timings INCLUDE their guard costs)
before any ingestion decision.

Ingested and shipped (records OPP-000047..049 + OPP-000046 extension):

| lead | OPP | dev-box probe |
|---|---|---|
| cholesky 2x2/3x3 per-matrix potrf overhead (scipy#24474 demand) | OPP-000047 | 2x2: 1.20x @100 -> 9.95x @10k; 3x3: 0.74x @100, 2.0-2.5x @1k-100k; maxerr 3.6e-17 of scale |
| eigvalsh 3x3 trig closed form (numpy#22158 class, OPP-000030 sibling) | OPP-000048 | 1.41x @100, 7.06x @1k, 5.38x @10k, 4.30x @100k; maxerr 9.3e-14 of scale |
| einsum >=3-operand chain: optimize=False runs ONE fused loop over the union of every index (numpy#11714 class) | OPP-000049 | ij,jk,kl->il n=64: 249x; the shipped einsum_optimize path was two-operand only, so this whole regime was uncovered |
| nan_to_num scalar overrides (nan=/posinf=/neginf=), numpy#23140 | OPP-000046 ext | 1.23x @10k, 2.03x @1e6, bit-exact |

Probed and DECLINED with measured proof (same script):

- numpy#3994 complex abs via sqrt(re^2+im^2): 0.14-0.30x across 1e4-1e7
  even with the guard costs amortized - numpy 2.4's complex abs is
  simply fast now. Stale.
- numpy#18153 tril/triu_indices closed-form construction: 0.26-0.75x
  across n=100-5000, all k probed - stock's tri+nonzero beats the
  repeat/cumsum build despite its n*m mask allocation. Stale.
- numpy#16573 median even-length averaging at large n: 1.06-1.13x at
  1e5/1e6, below min-win - consistent with OPP-000037's measured wash
  above its size cap; the small-n regime is already shipped
  (median_partition). Covered/stale.
- numpy#28921 ndindex and numpy#20790 __array_function__ overhead: no
  op-level function call to intercept (iterator class / core-C protocol
  dispatch) - same jurisdiction ruling as numpy#2269/#3446 in batch 6.
  Permanently out of scope.

Also verified while recording: numpy#11714's exact originally-reported
shape ('ij,ixy,ji->xy', the one optimize=True LOST 30x on in 2018) now
WINS 63.8x/247x/729x at i=60/100/200 under the modern planner (dev-box
probe, in OPP-000049) - the regression is long fixed upstream, the
default just never moved.

Post-ship probes, both DECLINED with idle-box proof (2026-08-24, fp
9bbe7063c555):

- cholesky d=4: the RAW closed form wins on both boxes (idle: 1.70x at
  300 to 2.60x at 1000; dev: up to 4.20x), but a shippable guarded
  route (LDL recurrences + scale pass + pivot bail) measures only
  1.30-1.32x in its best cells [1000, 3000] - exactly ON the 1.3x bar,
  inside the measured run-to-run swing. Declined until something
  changes the guard economics.
- LDL + mid-run-bail refactor of the shipped d=2/3 paths (predicate
  down to pure metadata, pivots free as LDL's D): measured only
  marginal in-window gains (3x3 at 3000: 2.15x vs the shipped 1.84x;
  2x2 at 5000: 2.47x vs 2.39x) and does NOT re-open the large-batch
  regime - d=3 still washes at >= 5000 (1.09x/0.97x). The wall is
  memory-bound, not guard tax: raw Crout at d=3 batch 5000 runs 69.7us
  where every guarded route runs ~350us, a jump that appears exactly
  where the route's full-stack temporaries (~10 arrays) spill L2 -
  also the mechanism behind the 5000 cell's bistability. Not worth
  churning a just-shipped, evidence-cited module.

Still open on the bench (untouched this batch): eig / eigh with
vectors remain blocked on the LAPACK sign-convention comparison
contract (see eigvalsh_2x2.py header - unchanged by this batch); the
eigvalsh-3x3 near-degenerate bail currently refuses the whole stack
when ANY matrix trips it - a split-and-recombine route (fast path for
the clean majority, stock for the degenerate few) was not measured; a
temporaries-lean cholesky formulation that stays under the L2 cliff at
batch >= 5000 (the raw-Crout numbers say ~5x is sitting there if the
guard can be paid without extra full-stack passes).
