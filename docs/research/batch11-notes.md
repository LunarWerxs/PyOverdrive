# Batch 11: the two order-of-magnitude paths, shipped (2026-08-25)

Batch 11 shipped the two >=100x leads the batch-11 shortlist identified,
taking the project to 41 always-on families / 67 registered paths. Both
are BIT-IDENTICAL: neither changes any arithmetic, both remove a Python
loop that stands between the caller and NumPy's own C kernels.

## 1. apply_along_axis with a known reducer (OPP-000054)

`np.apply_along_axis` is a documented convenience wrapper whose body is
a Python loop over an ndindex, calling `func1d` once per 1-D slice. When
`func1d` IS one of NumPy's own reductions, that loop reproduces, one
slice at a time and in Python, exactly what the reduction's `axis=`
argument does in a single vectorized call.

Idle-box OPP-000054 cells (fp 9bbe7063c555, 0% load): mean 96.5x at 256
slices, 148.4x at 1000, 177.8x at 20_000; sum 91.9-131.1x; max
36.9-75.0x; median 14.9-29.9x. Dev box agrees (mean up to 258.6x).
End-to-end through the dispatcher: 216.7x idle / 264.4x dev.

THE FINDING THAT SHAPED THE PATH: bit-identity does NOT hold for every
axis. For the LAST axis each 1-D slice is contiguous and both routes
accumulate in the same order. Off the last axis NumPy's reduction walks
whole rows across the buffer instead of one strided column at a time,
and floating-point addition is not associative - measured last-ulp
disagreement on (40, 500) axis=0 for sum, mean, std and var. So the
served set is split by ORDER-SENSITIVITY: max/min/median/any/all/ptp/
argmax/argmin are exact (integer or boolean comparisons, or a partition
over the same values) and are served on ANY axis; mean/sum/std/var/prod
are served on the LAST axis only. The differential suite proves the
split in both directions wherever it runs.

Refused, each read from numpy's implementation and then tested: zero
length dimensions (stock raises ValueError on a zero non-axis dim where
the axis= form returns empty; a zero AXIS dim makes stock emit one
warning per slice); np.matrix, where stock's own answer is shape-wrong
because matrix slices never become 1-D - the axis= form is therefore
different-and-arguably-better, exactly what a transparent accelerator
must not be; other subclasses; object and complex dtypes; ndim < 2
(one slice, nothing to win); extra args/kwargs; unmatched callables.

## 2. vectorize wrapping a ufunc (OPP-000055)

NumPy's own docs say vectorize "is provided primarily for convenience,
not for performance. The implementation is essentially a for loop."
Wrapping one of NumPy's OWN ufuncs in it is therefore pure loss.

Idle-box OPP-000055 cells: sqrt 35.8x at n=100 rising to 112.0x at
10_000 and 59.6x at 1M; exp 20.1-26.0x; sin 12.7-20.5x; log 19.1-23.3x;
tanh 15.1-16.3x. Dev box: sqrt 101.1x at 1M, and a BATCH11-CAL rint
cell at 986.7x. End-to-end: 185.8x idle / 260.6x dev.

THE SAFETY ARGUMENT IS A MEASUREMENT. NumPy's scalar loop (what
vectorize's object loop calls) and its array loop are separate
implementations that may differ in the last ulp, which would make this
path a silent value-changer. All 34 candidate unary float64 ufuncs were
verified bit-identical between the two loops over an adversarial sweep
on BOTH machines (34/34 on numpy 2.4.5 AMD Zen 4 and numpy 2.5.2 Intel
hybrid), and the differential suite RE-VERIFIES every member wherever it
runs - so a future NumPy build that breaks the property fails CI on that
platform instead of shipping wrong values.

## The dispatcher grew a class-patch mechanism

np.vectorize is a CLASS and the slow work lives on the instance's
`__call__`, so wrapping the name with a function would make
`isinstance(v, np.vectorize)` raise TypeError outright. The gearbox now
has ClassPath: it installs a SUBCLASS of the stock class, so
construction, attributes, isinstance, `type(v) is np.vectorize`, and
`__name__`/`__qualname__` all keep working, exactly one method is
overridden, and `disable()` restores the original class object. Its kill
switch is LIVE (the subclass consults the flag per call) because, unlike
a FastPath, it cannot simply leave an active list. selfcheck was
extended to cover class paths, so this path is verified on the user's
own machine like every other.

## Two defects the project's own machinery caught

Worth recording because both were invisible to ordinary testing:

1. **The identity lookups were dead on arrival.** Both paths match
   NumPy callables by identity - "is this func1d np.mean?". PyOverdrive
   PATCHES np.mean and np.sin, so once enabled the caller's np.mean IS
   PyOverdrive's wrapper, and a table built at import time stopped
   matching: the paths would never have fired in production. Every
   value test still PASSED, because when the path refuses, both sides
   run stock and the comparison is vacuous. SELFCHECK caught it, by
   asserting the path actually dispatches. The fix is a lookup rebuilt
   on a gearbox patch-generation counter, mapping both the wrapper and
   the stock object to the STOCK reducer.
2. **A dead 1-D branch.** apply_along_axis's contract says stock returns
   a 0-d ndarray for 1-D input where the axis= form returns a scalar, so
   the first build re-wrapped it. A 1-D array is exactly ONE slice and
   can never clear the slice floor, so that branch was unreachable. The
   test agent found it; the path now refuses ndim < 2 outright and says
   so.

## Bench after batch 11

Remaining from the batch-11 shortlist, in the order proposed there:
pinv 2x2/3x3 + the svdvals/norm2/cond family (20-34x for pinv via
adjugate with the conditioning band priced; matrix_rank excluded by the
panel), the pad narrow core (14-15x), ma.apply_along_axis (64.8x, needs
its own masked-semantics verify pass), then the sub-10x tail (det 4x4,
genfromtxt routing, ma fused-mask arithmetic). The I/O family still
needs an owner call on a disk-bound evidence standard.
