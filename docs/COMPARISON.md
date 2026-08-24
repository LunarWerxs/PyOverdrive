# How PyOverdrive relates to the other "faster NumPy" tools

PyOverdrive is not the only way to speed up NumPy-adjacent code, and it does
not try to replace the tools below. This page describes what each one
actually is, what adopting it costs you in code changes, and where its scope
starts and ends, so you can pick the right tool for the job rather than
guessing. No performance numbers for other projects appear here: only their
well-known, verifiable mechanics. PyOverdrive's own measured numbers are in
`README.md`.

## NumPy itself

NumPy keeps improving on its own, and PyOverdrive is built around that fact
rather than around it. ProsPyctor, PyOverdrive's mining system, reads
NumPy's own issue and PR tracker for optimization opportunities, and every
opportunity is re-checked against the current NumPy release before it
becomes a fast path. Some opportunities turn out to already be fixed
upstream: the classic `ufunc.at` slowness (tracked internally as
OPP-000003) was resolved in NumPy itself in 2023 and PyOverdrive marks it
stale rather than shipping a redundant path for it.

Several shipped fast paths are not reimplementations at all. They dispatch
into machinery NumPy already ships but does not always select by default,
for example routing `np.unique` through a stable radix sort
(`kind='stable'`) or `np.einsum` through NumPy's own `optimize=True`
contraction planner. In those cases PyOverdrive's contribution is choosing
the faster code path NumPy already has, on your behalf, when the measured
shape and dtype say it will win.

## numexpr

numexpr accelerates NumPy-style expressions by compiling an expression
given as a string into a chunked virtual machine that can run across
threads and avoid the intermediate arrays a normal chained NumPy expression
allocates. Adopting it means rewriting the computation as a string passed
to `numexpr.evaluate(...)` instead of ordinary Python operators, and it
targets expressions that chain several elementwise operations over large
arrays, not isolated calls.

## Numba

Numba just-in-time compiles Python functions to machine code, triggered by
decorating them (`@jit`, `@njit`). Adopting it means restructuring the
target code into functions Numba can compile, typically explicit loops and
a constrained subset of Python and NumPy, and accepting a compilation
delay the first time a given function signature is called (caching can
amortize this across runs).

## Cython and Pythran

Both compile annotated Python ahead of time into a C or C++ extension
module, as a build step before the code ever runs. Cython accepts a
superset of Python with optional static type annotations; Pythran compiles
a restricted, NumPy-aware subset of Python directly. Both require writing
(or annotating) the target code in their accepted dialect and adding a
compilation step to your build.

## bottleneck

bottleneck provides a set of nan-aware reduction functions, such as
alternatives to `np.nanmean` and `np.nansum`, in its own separate
namespace. Adopting it means changing call sites from the `numpy.nanX`
form to bottleneck's equivalent import and call.

## JAX and PyTorch

Both are separate array libraries with their own array types, dtype and
device model, and (for JAX) a tracing and XLA compilation model. Adopting
either means porting code to construct and operate on their array objects
instead of `numpy.ndarray`. They are not accelerators for existing NumPy
code in place; they are a different runtime you move your computation to.

## SciPy

SciPy has broader scope than NumPy, with functionality NumPy does not
provide at all (signal processing, sparse arrays, optimization, and more).
Where SciPy overlaps with NumPy, for example `scipy.signal.fftconvolve`
alongside `numpy.convolve`, using it means importing and calling SciPy
explicitly. Reaching for SciPy is usually a scope decision as much as a
speed one.

## What makes PyOverdrive different

The axis PyOverdrive optimizes for is adoption cost, not raw novelty:

- **Zero code change.** `import pyoverdrive; pyoverdrive.enable()` and your
  existing `np.*` call sites are what run. Calls that qualify for a fast
  path get one; everything else executes stock NumPy, untouched.
- **Per-call dispatch, not a blanket rewrite.** Each patched function
  checks the actual dtype, shape, contiguity, and size of the arguments in
  front of it against a measured threshold before deciding whether the
  fast path is worth taking, so a call that would lose does not take it.
- **Bit-identical results, or a documented tolerance.** Most fast paths
  return output identical to stock NumPy. The handful that do not
  (`inner_tensordot`, float `fftconvolve`, `nanquantile_masked`,
  `einsum_optimize`) are named as such in the README, each with a measured
  numeric tolerance rather than a vague "should be close" claim.
- **Instant rollback.** `pyoverdrive.disable()` restores every patched name
  to the original object; `pip uninstall pyoverdrive` leaves nothing else
  changed on the system.
- **Every number has receipts.** Each measured speedup in the README table
  ties back to a committed benchmark result stamped with a hardware
  fingerprint, not a one-off number quoted without its run.

## When not to use PyOverdrive

- **Tiny calls in a hot loop.** The dispatch check itself costs roughly
  300 nanoseconds per patched call; on very small inputs that overhead can
  outweigh the win (a 10-element `np.add` or `np.sin` in a loop can end up
  slower than stock).
- **Operator syntax.** `a + b` calls `ndarray.__add__` directly and is not
  reachable by a function-level patch; only the explicit function form
  (`np.add(a, b)`) can dispatch.
- **Code you're already willing to restructure.** If you can afford to
  annotate or rewrite a hot function, a JIT-based tool has more room to
  optimize than a dispatch layer that must keep the original call
  signature and semantics intact.
- **Pre-1.0 maturity.** PyOverdrive currently covers thirty-four fast-path
  families, verified on two architectures (AMD Zen 4 / numpy 2.4, Intel
  Alder Lake / numpy 2.5) with thresholds calibrated per machine. It is not a general
  claim of coverage across the NumPy API or across hardware.

## Using PyOverdrive alongside these tools

For most of the tools above, yes: PyOverdrive operates at the level of
individual `numpy.*` function calls, while numexpr, Numba, Cython, Pythran,
JAX, and PyTorch operate at the level of whole expressions, functions, or
entire array libraries, so the two layers generally do not conflict. The
one case worth watching is any tool that itself replaces NumPy functions
wholesale (monkey-patches or shadows `numpy.*` the way PyOverdrive does);
two tools patching the same names can interact in ways neither one tested
against, so that combination specifically should be verified before relying
on it in production.
