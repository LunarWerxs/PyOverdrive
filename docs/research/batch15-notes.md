# Batch 15: the aspect-ratio sweep, and a clean verdict (2026-08-25)

Batch 14 ended with a stated gap: the pessimization sweep scaled every
cell's leading axis both ways, but a loss that needs a particular ASPECT
RATIO - a long contraction against few rows, which is exactly what
np.inner's 0.38x corner looked like - can never appear from scaling,
because it has the same element count as the canonical input. This batch
built the instrument and ran it. It found nothing, and the skips were
audited so that the nothing can be believed.

## The instrument

`tools/verify_no_pessimization.py --shapes`: every cell with a
2-D-or-deeper input is re-judged at roughly constant volume with its
trailing axis grown 4x and 16x while the leading axis shrinks by the same
factor, and the reverse. Two directions because batch 14's lesson was that
both directions of an unsampled axis bite.

Operand coupling is not knowable in general, so two rules are tried in
turn and the first one the path dispatches on is judged:

- **shared-trailing**: every array whose last axis matches the longest
  last axis moves together. Keeps np.inner's stacked operands and a
  matrix-vector product valid; leaves small parameter arrays (a quantile
  vector beside its sample) alone.
- **chain**: two 2-D operands with `a.shape[-1] == b.shape[0]` are a
  product chain, so the SHARED inner dimension grows while the outer
  dimensions shrink - the shared-trailing rule would break the chain.

A rule that produces an input the op cannot run, or that the predicate
refuses, costs a printed skip and nothing else. 116 shape cells across the
29 paths that have a shape at all; 1-D cells are filtered in the parent
rather than paying a subprocess each to say so.

## The verdict, from the idle Intel box

33 cells dispatch and NONE is below 1.0x. The weakest is einsum at 1.26x
(<16, a 30x480 chain); everything else clears 1.4x, and the stock-is-slow
families (apply_along_axis, the masked nan-quantiles) run to 402x.

The result that matters most is a refusal: `inner_tensordot>4` - the long
contraction against few rows that shipped at 0.38x for months - does NOT
dispatch. That is batch 14's restrictive gate holding in a direction it
was never explicitly tested in. The lead-heavy directions dispatch and win
(6.47x at <4, 2.76x at <16).

## Why 83 skips can be believed

A sweep whose cells silently fail to construct would read exactly like a
sweep of well-gated paths, so the suspicious skips were checked by hand:
the chain variants really do produce valid products - `(25,400)x(400,25)`
for int matmul, `(16,4800)x(4800,150)` for the complex split - and the
predicates genuinely refuse them, because at constant volume a 16x aspect
move pushes the output (or the batch) below the path's measured floor.
That is the gate doing its job: a refused shape cannot lose.

The square-matrix families (det/inv/solve/eigvalsh/cholesky/qr/svd) refuse
everything, correctly - reshaping a (N,d,d) stack breaks squareness and
their predicates require it. Their aspect axis is batch-vs-d, and d is a
separate calibration row per dimension; see the gaps below.

## Still not covered, stated in the tool

- Regimes on a KEYWORD axis rather than a shape (histogram bin counts -
  that one bit in batch 14 and now has a sample floor).
- Per-dimension cells for the small-matrix linalg families: the selfcheck
  input picks one d, other d's have their own calibration rows but only
  one is swept.
- Layout: every reshape here lands C-contiguous, so the F-contiguity-gated
  relayout path skips its shape cells.
- Chains of three or more operands get no chain variant (einsum_chain's
  predicate refused the broken reshape, so nothing was at risk).
- Intermediate factors: a path that dispatches at aspect 1x and refuses at
  4x could in principle lose at 2x. The factors sample the class, they do
  not exhaust it.

## State at the end of the batch

No src change - this batch shipped an instrument and a verified negative.
The Intel sweep instruction is now `--sizes` AND `--shapes`. 2207 tests
here / 2208 on the Intel box, 69 registered paths (67 always-on), 693 size
cells and 33 dispatching shape cells all >= 1.0x.
