"""OPP-000025: np.zeros_like(a) vs np.zeros(shape=a.shape) allocation fast
path.

numpy/numpy#9909 (opened 2017-10-23, still open at ingestion) reports
np.zeros_like(a) at 2.18 ms vs np.zeros(shape=a.shape) at 40.7 us on a
(1000, 1000) float64 array -- a DERIVED 53.56x (the thread never states the
ratio; both inputs are the reporter's own %timeit numbers). Mechanism: the
then-current zeros_like was empty_like(a) + copyto(res, 0), which physically
writes a zero into every element, while np.zeros can take calloc-style
lazily-zeroed OS pages and pay nothing until first touch. The claim regime
is therefore ALLOCATION-ONLY at one shape and one dtype.

This reproducer measures:

  - baseline: np.zeros_like(a), exactly as a user writes it (subok and
    order left at their defaults).
  - candidate "zeros_from_shape": np.zeros(a.shape, dtype=a.dtype,
    order=<'F' if a is F-and-not-C-contiguous else 'C'>) -- the route the
    reporter proposes in the issue body, with dtype and order inherited
    from a per tadeu's 2017-10-24 caveats. The flag inspection that
    resolves the order is INCLUDED in the timed call, so the measured
    speedup is net of the dispatch check's own cost. The candidate never
    imports pyoverdrive and never calls np.zeros_like (the op under
    reproduction), so it cannot recurse if zeros_like is patched.

Three regimes, per the record's measurement-honesty note (an alloc-only
number cannot be interpreted alone, because np.zeros defers the memory
touch cost that zeros_like pays up front):

  1. Alloc-only size sweep: (100,), (100, 100), (1000, 1000),
     (4000, 4000) float64. (1000, 1000) is THE claim regime; the sweep
     brackets the allocator's lazy-zero threshold (below it both paths
     memset and the gap should collapse; (4000, 4000) is 128 MB, far
     above it).
  2. Dtype and layout spread at the claim shape: int32, complex128, and
     an F-contiguous float64 input (candidate must route to order='F').
  3. Alloc-plus-touch at (1000, 1000) and (4000, 4000) float64: create
     then fully write (r[:] = 1) and create then fully read (r.sum()),
     both legs. This prices the deferred page faults and yields the honest
     end-to-end ratio; expect it to be far below the alloc-only number.

Correctness: exact, as befits an allocation op. Array results must match
stock zeros_like in values (np.array_equal), dtype, shape, contiguity
flags, and writeability. The read-touch variant returns a scalar sum,
exactly 0.0 on both legs, checked for exact equality.

NOT measured: non-contiguous slice inputs and ndarray subclasses (items
the research doc lists for fallback verification). The candidate route's
predicate leaves both on stock zeros_like -- np.zeros cannot reproduce
arbitrary strides, and subclass preservation requires *_like semantics --
so there is no candidate route to time there; timing stock against itself
is noise. Verifying that a dispatch layer actually falls back is a
pyoverdrive-layer test, and this reproducer never imports pyoverdrive.

Sizes are NOT shrunk: even the 128 MB (4000, 4000) legs cost tens of
milliseconds per call, so the full non-smoke battery lands well under the
~90 s budget.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 9909
SMOKE = "--smoke" in sys.argv


def zeros_from_shape(a):
    """The route from the issue body: np.zeros with shape, dtype, and order
    taken from a. Order resolution from the flags is deliberately inside
    the timed call (it is part of what a real dispatch would pay). Never
    calls np.zeros_like."""
    if a.flags.f_contiguous and not a.flags.c_contiguous:
        return np.zeros(a.shape, dtype=a.dtype, order="F")
    return np.zeros(a.shape, dtype=a.dtype, order="C")


def zeros_like_write(a):
    r = np.zeros_like(a)
    r[:] = 1
    return r


def zeros_from_shape_write(a):
    r = zeros_from_shape(a)
    r[:] = 1
    return r


def zeros_like_read(a):
    return np.zeros_like(a).sum()


def zeros_from_shape_read(a):
    return zeros_from_shape(a).sum()


def exact_check(cand, base):
    """Exact equality for an allocation op: no tolerance is defensible when
    both legs are defined to produce identical zero-filled (or identically
    touched) arrays. Also enforces dtype, shape, contiguity flags, and
    writeability, per the record's correctness list."""
    if isinstance(base, np.ndarray):
        return (
            isinstance(cand, np.ndarray)
            and cand.dtype == base.dtype
            and cand.shape == base.shape
            and cand.flags.c_contiguous == base.flags.c_contiguous
            and cand.flags.f_contiguous == base.flags.f_contiguous
            and cand.flags.writeable == base.flags.writeable
            and np.array_equal(cand, base)
        )
    # touch-read variant: scalar sums, exactly equal (0.0 + ... + 0.0)
    return bool(cand == base)


def make_input(rng, shape, dtype, order="C"):
    """Seeded random contents. zeros_like never reads a's values, only its
    shape/dtype/layout, but seeding keeps the convention and makes the
    input pages genuinely touched memory rather than lazy zeros."""
    if dtype == "complex128":
        a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            np.complex128
        )
    elif dtype == "int32":
        a = rng.integers(-1000, 1000, size=shape, dtype=np.int32)
    else:
        a = rng.uniform(size=shape).astype(dtype)
    if order == "F":
        a = np.asfortranarray(a)
    return a


def shape_tag(shape):
    return "x".join(str(s) for s in shape)


suite = BenchSuite(
    "OPP-000025",
    "np.zeros_like vs np.zeros(shape, dtype, order) allocation fast path",
)

rng = np.random.default_rng(SEED)

if SMOKE:
    alloc_cases = [((100,), "float64", "C"), ((100, 100), "float64", "C")]
    touch_shapes = [(100, 100)]
    samples_for = lambda shape: 3
else:
    alloc_cases = [
        ((100,), "float64", "C"),
        ((100, 100), "float64", "C"),
        ((1000, 1000), "float64", "C"),  # THE claim regime of numpy/numpy#9909
        ((4000, 4000), "float64", "C"),
        ((1000, 1000), "int32", "C"),
        ((1000, 1000), "complex128", "C"),
        ((1000, 1000), "float64", "F"),
    ]
    touch_shapes = [(1000, 1000), (4000, 4000)]
    samples_for = lambda shape: 5 if int(np.prod(shape)) >= 16_000_000 else 7

# Regime 1 + 2: alloc-only sweep, dtype spread, F-order layout
for shape, dtype, order in alloc_cases:
    a = make_input(rng, shape, dtype, order)
    case = f"alloc_{shape_tag(shape)}_{dtype}" + ("_F" if order == "F" else "")
    suite.measure(
        case=case,
        params={
            "dtype": dtype,
            "shape": list(shape),
            "order": order,
            "regime": "alloc_only",
        },
        baseline=("numpy.zeros_like", lambda a=a: np.zeros_like(a)),
        candidates={"zeros_from_shape": lambda a=a: zeros_from_shape(a)},
        check=exact_check,
        samples=samples_for(shape),
    )
    del a

# Regime 3: alloc-plus-touch (full write, full read), float64 C-order
for shape in touch_shapes:
    a = make_input(rng, shape, "float64")
    samples = samples_for(shape)
    suite.measure(
        case=f"touch_write_{shape_tag(shape)}_float64",
        params={
            "dtype": "float64",
            "shape": list(shape),
            "order": "C",
            "regime": "alloc_then_full_write",
        },
        baseline=("numpy.zeros_like+write", lambda a=a: zeros_like_write(a)),
        candidates={"zeros_from_shape+write": lambda a=a: zeros_from_shape_write(a)},
        check=exact_check,
        samples=samples,
    )
    suite.measure(
        case=f"touch_read_{shape_tag(shape)}_float64",
        params={
            "dtype": "float64",
            "shape": list(shape),
            "order": "C",
            "regime": "alloc_then_full_read",
        },
        baseline=("numpy.zeros_like+sum", lambda a=a: zeros_like_read(a)),
        candidates={"zeros_from_shape+sum": lambda a=a: zeros_from_shape_read(a)},
        check=exact_check,
        samples=samples,
    )
    del a

if not SMOKE:
    suite.save()
