# Batch-11 shortlist: the order-of-magnitude hunt (2026-08-24)

Owner directive: comprehensive analysis, at least five places with 10x,
at least one with 100x. Method: deterministic probes with honest
candidates (guards priced) on the dev box (fp 8f8198d9abab), the full
core set re-measured on the idle box (fp 9bbe7063c555, 0-1% load,
numpy 2.5.2), then a five-skeptic adversarial semantics panel (each
lead's numpy implementation read and its traps tested empirically)
plus two miners over upstream issues and numpy/ma internals.

## The >=100x places (two found; requirement was one)

| lead | measured | verdict |
|---|---|---|
| np.apply_along_axis(np.mean/sum/max, axis, a) -> the axis= reduction | mean 103.9-292.1x dev / 109.4-179.2x idle; sum 81-235x; max 55-114x; median 8.4-20.3x | panel: SHIP with four guards (below) |
| np.vectorize(ufunc) gated core: f64 unary listed ufuncs, non-scalar non-empty input | sin 20.7x, exp 36.6x, sqrt 101.1x - BIT-IDENTICAL on float64 | panel refuted the blanket short-circuit (3 empirical breaks) but its own servable core is exactly this gate |

apply_along_axis guards (from the panel, each tested): refuse unless
func1d is identity-matched with no extra args/kwargs; type(arr) is
exactly np.ndarray (np.matrix gives a DIFFERENT, wrong-shaped stock
answer; subclasses have their own reduction behavior); every non-axis
dim > 0 (stock raises ValueError there, the axis= form does not); 1-D
input wraps the scalar back into a 0-d ndarray to match stock's return
type.

vectorize traps that the gate excludes (each empirically confirmed by
the skeptic): scalar input returns 0-d ndarray from stock vs np.float64
raw (gate: non-scalar only); empty input raises from stock (gate:
size > 0); float32 differs bitwise between the object-loop and SIMD
paths (gate: float64 only). Remaining architecture question: numpy.
vectorize is a CLASS, so this ships as a subclass replacing the attr -
a new pattern for the dispatcher, flagged for design before build.

## The >=10x places (six distinct functions; requirement was five)

1. np.apply_along_axis - reducer class, 55-292x (above).
2. np.ma.apply_along_axis - same mechanism on masked arrays, 64.8x
   measured (np.ma.mean(md, axis=1) is the true equivalent); needs its
   own masked-semantics verify pass before build.
3. np.vectorize - gated f64-ufunc core, 20.7-101.1x, bit-identical.
4. np.linalg.pinv on (...,2,2) batches via adjugate inverse with the
   conditioning band priced in the candidate: 20.1x at n=1000, 34.3x
   at 10_000, 22.0x at 100_000 (err 4.5e-13..5.9e-11). Panel: the
   adjugate route holds the 1e-9 contract to cond~1e7 (2x2) / 1e5
   (3x3); band + split-to-stock is the established house pattern.
   (The earlier 7.7-8.0x "pinv" cell used stock LAPACK inv as a lazy
   proxy; the real adjugate candidate is 3-4x faster than that.)
5. np.pad - 1-D constant-mode small arrays, 13.9x dev / 15.2x idle at
   n=64. Panel: SHIP for the narrow core (plain ndarray, f64/i64,
   default mode, tuple/scalar pad_width, real numpy casting for
   constant_values - seven traps enumerated, all gateable).
6. apply_along_axis median regime, 8.4-20.3x (same interception as #1,
   listed separately because its margin class differs).

## Solid sub-10x adds (the "more things to add" bench)

- svd-values family 2x2/3x3 batches via A^T A closed forms: svdvals
  3.7-5.6x, norm(ord=2) 3.6-4.9x (panel: largest-SV is eps-accurate at
  ANY cond - no band needed for norm ord=2), cond 3.5-4.7x (band
  ~3e3/3e2), pinv-3x3 (see above). matrix_rank EXCLUDED: the panel
  showed its integer contract breaks ~50% of the time whenever true
  sv_min sits within ~4 orders of the tolerance - no band closes it.
- det/slogdet 4x4 batches via cofactor closed form: 1.6-3.8x (L2 cliff
  at 100k wants the cholesky chunk treatment).
- genfromtxt -> loadtxt routing for clean simple-kwargs calls: 3.8x
  bit-equal.
- np.ma: domained ops (divide/log/sqrt) are 6.2x plain divide with a
  credible fused-mask path to ~2-3x vs stock ma (miner: 8 full-array
  passes where 3 suffice); add/multiply 2.3-2.6x (fixed Python
  overhead: errstate CM, redundant m.any(), _update_from dict churn);
  ma.std 3.0x.
- histogramdd 3-D uniform: 3.1-3.9x naive - edge-correction unpriced,
  the hist1d withdrawal (batch 8) warns exactly here.

## Declined with measured proof

piecewise (my 10-15x cell was an algorithm-change artifact - the
honest faithful candidate replicates stock's per-branch gather/call/
scatter, and the semantically clean constants-only regime measures
1.20-1.36x; the panel additionally confirmed compressed-call warning
semantics and last-write-wins overlap ordering); round-vs-rint 1.02x
(numpy fixed the 2021 report); savetxt via char.mod 0.48-0.59x;
kron 0.90-0.92x; trim_zeros 1.19x; delete/insert 1.2-1.4x;
np.average 1.1-1.5x; apply_over_axes 0.64-0.98x; ma.median 0.84-1.0x;
np.char legacy ops 0.88-0.91x vs listcomp (numpy 2.x routes them to C
ufuncs); lstsq 1-2 col closed form 0.5-1.9x (stock's small-system
gelsd is far cheaper than assumed); np.roots deg-2 (10x margin but
stock's root ORDER is ~50/50 desc/asc across random inputs - LAPACK
geev order is not reproducible, contract dead).

## Mined upstream classes judged out of reach

bincount-sparse-large-values (numpy#11863, ~70000x reported - but the
dense output array IS the contract); np.random.choice single element
(numpy#11476, ~100x - random-stream fidelity cannot be preserved by a
different algorithm); int+int32 scalar mixing (numpy#12496, 25-50x -
operator dispatch, not an interceptable function); str(ndarray)
(numpy#18098, 29-50x - requires reproducing dragon4 formatting
exactly; research, not a probe); object-dtype __array__ construction
(numpy#28651 - upstream bug class, not dispatchable).

## Shipping order proposed for batch 11+

1. apply_along_axis (the 100x headline; guards fully specified).
2. vectorize gated core (bit-identical 20-101x; needs the class-patch
   design call first).
3. pinv 2x2/3x3 + svdvals/norm2/cond family (one shared closed-form
   core, the band machinery already exists in-repo).
4. pad narrow core; ma.apply_along_axis after its own verify pass.
5. det4x4 extension, genfromtxt routing, ma fused-mask arithmetic as
   the sub-10x tail.
