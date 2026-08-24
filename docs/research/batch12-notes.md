# Batch 12: the singular-value family, and one lead refuted (2026-08-25)

Batch 12 shipped three paths from one closed-form core (OPP-000056) and
DECLINED the 64.8x masked-array lead on semantics. 44 always-on families
/ 70 registered paths.

## Shipped: pinv, norm(ord=2), svd(compute_uv=False) on 2x2/3x3 batches

Same per-matrix-LAPACK-dispatch vein as the five small-matrix families
already shipped. The singular values of A are the square roots of the
eigenvalues of the gram matrix A^T A, which is 3x3 or smaller here and
has a closed form; for a well-conditioned square matrix pinv(A) is just
inv(A), which the adjugate gives directly.

Margins with every guard priced in (dev box, both bands and both
degeneracy guards live): pinv 2x2 18.9-24.0x, 3x3 8.6-11.3x; norm2 2x2
14.0-15.1x, 3x3 8.0-11.3x; svdvals 2x2 10.6-12.1x, 3x3 7.8-8.9x.

Two implementation notes that carried most of the speed:

- The gram is built ENTRYWISE (six products for d=3), never with a
  batched matmul. That one choice moved the measured pinv margin from
  ~4x to ~25x: the matmul cost more than the entire closed form it fed.
- Band-trippers are SPLIT OUT and served by stock, not bailed on. This
  is the batch-10 qr lesson applied before it could bite: random
  matrices trip a conditioning band with some per-matrix probability, so
  a large enough stack almost surely contains one. A stack of 100_000
  random 3x3s does, and the first build refused the whole thing.

## The defect this batch nearly shipped

The module was designed around ONE hazard - ill-conditioning - because
forming the gram squares the condition number, so accuracy degrades like
eps*cond^2. Bands were measured per dimension and set with a decade of
margin. That analysis was correct and it was not enough.

A COALESCING PAIR of singular values is a second, orthogonal hazard, and
the conditioning band is blind to it: it happens at any conditioning,
including on a perfectly well-conditioned matrix. Measured on stacks
whose sigma_min/sigma_max is a healthy 0.3, so every band check passed:

    d=3  singular values (1, 1, 0.3)   -> 4.3e-09 svdvals, 4.5e-09 norm2
    d=3  singular values (1, 0.3, 0.3) -> 1.3e-08 svdvals
    d=2  singular values (1, 1)        -> 7.5e-09

Four to thirteen times outside the 1e-9 contract, silently. Two
different mechanisms underneath: for d=3 the arccos amplification that
eigvalsh_3x3's DEGENERACY_MIN already documents; for d=2 catastrophic
cancellation in the discriminant t^2 - 4*det, whose square root then
carries ~sqrt(eps) error. One constant guards both (squared relative
separation >= 1e-12), delivering 3.3e-11 at the d=2 threshold and
~1e-10 at the d=3 one, while diverting only ~0.4% of a random stack.

Worth recording HOW it surfaced, because the lesson generalizes. Not
from the conditioning sweep - that was the axis the module was designed
around, and it looked clean. It came from an adversarial test agent
building fixtures and remarking that one shape held ~4e-9 error
"regardless of true conditioning". norm2 was the sharpest case: it had
been reasoned about as needing no guard AT ALL, on the correct
observation that sigma_max is eps-accurate at every condition number -
reasoning that was simply blind to a second failure axis. The guard you
designed is not always the guard you need, and the way to find the
other one is to go looking for it on purpose.

Also deliberately not served: np.linalg.cond (divides by sigma_min,
amplifying exactly the digits the gram route loses) and
np.linalg.matrix_rank (an earlier panel showed its INTEGER contract
breaks about half the time near the rank tolerance, and no band closes
that).

## Declined: np.ma.apply_along_axis (64.8x)

The masked-array sibling of the batch-11 100x path measured 64.8x, and
is refused. An adversarial semantics pass found five divergences from
the naive substitution, of which the first alone is fatal:

1. Mask representation: whenever the result has no masked elements,
   apply_along_axis collapses .mask to the scalar `nomask`, while the
   direct axis= call materializes a full boolean array. That is the
   COMMON case, not an edge case.
2. For any/all on a fully-masked slice, apply_along_axis's result
   carries the generic float fill_value 1e+20 where the direct call
   correctly carries bool True.
3. 1-D input returns a 0-d MaskedArray vs a bare scalar.
4. MaskedArray subclasses are demoted to the base class by one route
   and preserved by the other.
5. A zero-length iteration dimension raises IndexError rather than the
   plain version's ValueError.

The order-sensitivity split from the plain sibling carries over too
(mean/sum/std/var diverge off the last axis). A faithful port would have
to reproduce nomask-collapsing and an any/all fill_value quirk that is
arguably a NumPy bug - matching a bug on purpose, for a win that only
reaches np.ma users. Not worth the contract risk.

## Bench after batch 12

Still queued from the batch-11 shortlist: the pad narrow core
(13.9-15.2x, panel-specified with seven enumerated traps), det/slogdet
on 4x4 batches (1.6-3.8x), genfromtxt routed to loadtxt for clean simple
calls (3.8x, bit-equal), and the np.ma fused-mask arithmetic family
(domained ops are 6.2x plain divide, with a credible ~2-3x path). The
I/O family still needs an owner call on a disk-bound evidence standard.
