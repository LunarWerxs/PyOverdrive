"""Fast path: numpy.vectorize wrapping a plain unary ufunc, called
directly instead of through the object loop.

Provenance (OPP-000055): np.vectorize is documented as a convenience
wrapper "provided primarily for convenience, not for performance" whose
implementation is essentially a Python loop. People nonetheless wrap
NumPy's own ufuncs in it - inside generic code, in teaching material,
and wherever a function is passed around and vectorized defensively -
and in that case the loop is pure loss: the wrapped callable is already
a fully vectorized C ufunc.

Measured (dev box, fp 8f8198d9abab, float64, n=1M): sqrt 101.1x, exp
36.6x, sin 20.7x; the earlier survey cell showed 17.8x for sin at
default settings and 14.1x on the idle box (fp 9bbe7063c555). Results
are BIT-IDENTICAL on the served set (below), because the two routes run
the same ufunc over the same values.

WHY THIS IS A CLASS PATCH, NOT A FUNCTION WRAPPER. np.vectorize is a
class and the slow part is the INSTANCE's __call__. Replacing the name
with a function would make ``isinstance(v, np.vectorize)`` raise
TypeError outright. Installing a SUBCLASS instead keeps isinstance,
type(v) is np.vectorize, attribute access, pickling of the class name,
and every documented attribute working, and overrides exactly one
method. See ClassPath in the dispatcher.

THE SERVED SET IS MEASURED, NOT ASSUMED. NumPy's scalar loop and its
array loop are separate implementations, and for some math they can
differ in the last ulp - which would make this path a silent
value-changer. So the path serves only ufuncs verified bit-identical
between the two loops over an adversarial input sweep (tiny, huge,
negative, exact halves, domain edges), on both benchmark machines AND
re-verified by the differential suite on every machine that runs the
tests: if a future NumPy build breaks the property for any member, CI
fails loudly on that platform rather than shipping wrong values.

The remaining wrapper edges, each read out of numpy's implementation,
tested, and refused here rather than assumed:

- Scalar/0-d input: stock returns a 0-d ndarray, the raw ufunc returns
  a NumPy scalar. Same value, different type -> refused.
- Empty input: stock RAISES ValueError ("cannot call `vectorize` on
  size 0 inputs unless `otypes` is set"); the raw ufunc happily returns
  an empty array -> refused, so the error survives.
- float32 and other dtypes: stock infers the output dtype from a trial
  call on the FIRST element and runs an object loop, which for float32
  computes in float64 and casts - genuinely different bits from the
  ufunc's native float32 SIMD loop. float64 only.
- otypes, signature, excluded, cache: each changes what stock computes
  or how; any of them refuses the instance to the stock __call__.

Correctness contract: an instance constructed as vectorize(f) where f
is a served unary ufunc and no otypes/signature/excluded/cache were
given, called with exactly one positional argument that is a plain
float64 ndarray with ndim >= 1 and size > 0, returns exactly
``f(arr)``. Every other construction and every other call shape runs
stock's own __call__ unchanged.

Comparison mode: bit-identical (spec section 9). Kill switch:
vectorize_ufunc_direct (live: the installed subclass consults the
path's enabled flag on every call).
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, ClassPath

_F64 = np.dtype(np.float64)

# Unary float64 ufuncs measured bit-identical between NumPy's scalar
# loop (what vectorize's object loop calls) and its array loop (what the
# direct call uses), over an adversarial sweep on both benchmark
# machines. compatibility/differential/test_vectorize_ufunc_differential
# re-verifies EVERY member wherever the suite runs, so this list cannot
# rot silently across NumPy versions, builds, or CPU dispatch targets.
_SERVED_NAMES = (
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "arcsinh", "arctanh",
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
    "sqrt", "cbrt", "square", "reciprocal", "absolute", "negative",
    "fabs", "rint", "floor", "ceil", "trunc", "sign",
    "degrees", "radians", "spacing", "conjugate",
)


# id(callable) -> the STOCK ufunc to call. Rebuilt whenever the gearbox
# patches or unpatches: while PyOverdrive is enabled the caller's np.sin
# is OUR threaded wrapper, so a table built before patching would stop
# matching and this path would silently never fire.
#
# The value is deliberately the STOCK ufunc: stock vectorize would call
# the wrapped callable once per ELEMENT, where every PyOverdrive ufunc
# path declines (scalar input), so stock's own answer is the stock
# ufunc's. Calling it directly is therefore exactly right, and avoids a
# second dispatch.
_LOOKUP: dict[int, object] = {}
_LOOKUP_GENERATION = -1


def _lookup() -> dict:
    global _LOOKUP, _LOOKUP_GENERATION
    gen = GEARBOX.generation
    if gen != _LOOKUP_GENERATION:
        table: dict[int, object] = {}
        for name in _SERVED_NAMES:
            op = f"numpy.{name}"
            try:
                stock = GEARBOX.stock_fn(op)
            except (AttributeError, ValueError):  # pragma: no cover
                continue
            if not (isinstance(stock, np.ufunc) and stock.nin == 1 and stock.nout == 1):
                continue
            table[id(stock)] = stock
            live = getattr(np, name, None)
            if live is not None:
                table[id(live)] = stock
        _LOOKUP, _LOOKUP_GENERATION = table, gen
    return _LOOKUP


def _served_ufunc(pyfunc):
    """The stock ufunc this callable resolves to, or None."""
    return _lookup().get(id(pyfunc))


def _instance_ok(v):
    """The stock ufunc to call for this instance, or None to stay on stock."""
    uf = _served_ufunc(getattr(v, "pyfunc", None))
    if uf is None:
        return None
    if getattr(v, "otypes", None):  # '' / None both mean "infer"
        return None
    if getattr(v, "signature", None) is not None:
        return None
    if getattr(v, "excluded", None):
        return None
    return None if getattr(v, "cache", False) else uf


def _call_ok(args: tuple, kwargs: dict) -> bool:
    if len(args) != 1 or kwargs:
        return False
    a = args[0]
    if type(a) is not np.ndarray or a.dtype != _F64:
        return False
    # 0-d returns a different TYPE from stock; empty RAISES in stock
    return a.ndim >= 1 and a.size > 0


def _applicable(args: tuple, kwargs: dict) -> bool:
    """For explain(): would constructing with these arguments accelerate?"""
    pyfunc = args[0] if args else kwargs.get("pyfunc")
    if _served_ufunc(pyfunc) is None:
        return False
    if kwargs.get("otypes") or kwargs.get("excluded"):
        return False
    if kwargs.get("signature") is not None or kwargs.get("cache", False):
        return False
    # positional otypes/doc/excluded/cache/signature after pyfunc
    return len(args) <= 1


def _make(stock_cls: type) -> type:
    """Build the installed subclass over the stock np.vectorize."""

    class OverdriveVectorize(stock_cls):  # type: ignore[misc, valid-type]
        """np.vectorize that calls a wrapped ufunc directly when it can.

        Identical to numpy.vectorize in construction, attributes and
        every unserved call; see pyoverdrive.fastpaths.vectorize_ufunc.
        """

        __pyoverdrive__ = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # decided once, at construction, from the resulting instance
            # (robust to numpy's argument spellings and defaults)
            try:
                self.__pyoverdrive_direct__ = _instance_ok(self)
            except Exception:  # noqa: BLE001 - never break construction
                self.__pyoverdrive_direct__ = None

        def __call__(self, *args, **kwargs):
            direct = getattr(self, "__pyoverdrive_direct__", None)
            if direct is not None and PATH.enabled and _call_ok(args, kwargs):
                return direct(args[0])
            return super().__call__(*args, **kwargs)

    OverdriveVectorize.__name__ = stock_cls.__name__
    OverdriveVectorize.__qualname__ = stock_cls.__qualname__
    return OverdriveVectorize


PATH = ClassPath(
    name="vectorize_ufunc_direct",
    op="numpy.vectorize",
    make=_make,
    applicable=_applicable,
    provenance={
        "opportunity": "OPP-000055",
        "source": "numpy.lib._function_base_impl.vectorize (documented convenience wrapper)",
        "license": "calls the user's own wrapped ufunc; no third-party code",
        "comparison_mode": "bit-identical",
    },
)


def register(gearbox) -> None:
    gearbox.register_class(PATH)
