"""Gearbox: PyOverdrive's adaptive dispatcher.

Table-driven and testable independently from kernels (spec section 10.1).
For each patched operation Gearbox walks the registered fast paths in priority
order; the first whose ``applicable`` predicate accepts the call runs. Anything
else, and any fast-path exception, falls back to stock NumPy.

Invariants:

- Deterministic: same environment + inputs => same decision.
- Conservative: no registered path applicable => stock NumPy, always.
- Kill-switchable: every fast path can be disabled by name (API or the
  ``PYOVERDRIVE_DISABLE`` environment variable, read at import time).
- Explainable: debug mode prints the decision and reason for every dispatch;
  ``explain()`` reports the decision without executing the operation.

Dispatch overhead is a benchmarked quantity, not an assumption: see
benchmarks/historical/opp_000000_noop_overhead.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable


class StockRaised(BaseException):
    """Raised by a fast path's run to surface an exception STOCK raised.

    Some paths fall back to stock inside their run (the graceful-fallback
    pattern) and that stock call can legitimately raise, e.g. nanargmax
    on all-NaN input. That is not a fast-path failure: the path
    faithfully reproduced stock's behavior. The dispatch wrappers
    re-raise the wrapped original unchanged, with no fast-path-failure
    RuntimeWarning and no second stock call. BaseException on purpose so
    a path's own ``except Exception`` cannot swallow it by accident.
    """

    def __init__(self, original: BaseException) -> None:
        super().__init__(repr(original))
        self.original = original


@dataclass
class FastPath:
    """One registered accelerated implementation.

    ``applicable(args, kwargs)`` must be cheap, side-effect free, and
    conservative: return True only when ``run`` is known-safe and expected
    profitable for this call. ``provenance`` records where the idea came from
    (spec section 4.7): opportunity id, source URL, license notes.
    """

    name: str
    op: str  # e.g. "numpy.unique"
    applicable: Callable[[tuple, dict], bool]
    run: Callable[..., Any]
    provenance: dict = field(default_factory=dict)
    priority: int = 0  # higher runs first
    enabled: bool = True


class Gearbox:
    def __init__(self) -> None:
        self._paths: dict[str, list[FastPath]] = {}
        # Per-op list of ENABLED paths in priority order. The list object is
        # shared with that op's wrapper closure and mutated in place, so the
        # dispatch hot path never does a dict lookup or an `enabled` check.
        self._active: dict[str, list[FastPath]] = {}
        self._stock: dict[str, Callable] = {}
        self._debug = os.environ.get("PYOVERDRIVE_DEBUG", "") not in ("", "0")
        self._warned_paths: set[str] = set()
        self.patched = False
        killed = os.environ.get("PYOVERDRIVE_DISABLE", "")
        self._killed_at_import = {k.strip() for k in killed.split(",") if k.strip()}

    # -- registration -----------------------------------------------------

    def register(self, path: FastPath) -> None:
        if path.name in self._killed_at_import:
            path.enabled = False
        ops = self._paths.setdefault(path.op, [])
        if any(p.name == path.name for p in ops):
            raise ValueError(f"duplicate fast path name: {path.name}")
        ops.append(path)
        ops.sort(key=lambda p: -p.priority)
        self._rebuild_active(path.op)

    def _rebuild_active(self, op: str) -> None:
        active = self._active.setdefault(op, [])
        active[:] = [p for p in self._paths.get(op, ()) if p.enabled]

    def supported_operations(self) -> list[str]:
        return sorted(self._paths)

    def set_path_enabled(self, name: str, value: bool) -> None:
        for op, paths in self._paths.items():
            for p in paths:
                if p.name == name:
                    p.enabled = value
                    self._rebuild_active(op)
                    return
        raise KeyError(f"no fast path named {name!r}")

    # -- dispatch ---------------------------------------------------------

    def dispatch(self, op: str, stock: Callable, args: tuple, kwargs: dict):
        self._active.setdefault(op, [])
        return self._run_active(op, self._active[op], stock, args, kwargs)

    def _run_active(self, op: str, active: list, stock: Callable, args: tuple, kwargs: dict):
        for path in active:
            try:
                if not path.applicable(args, kwargs):
                    continue
            except Exception:
                continue  # a broken predicate must never break the call
            try:
                result = path.run(*args, **kwargs)
            except StockRaised as sr:
                raise sr.original
            except Exception as exc:
                self._on_path_error(op, path, exc)
                continue
            if self._debug:
                print(f"[gearbox] {op} -> {path.name}")
            return result
        if self._debug:
            print(f"[gearbox] {op} -> stock (no applicable path)")
        return stock(*args, **kwargs)

    def _on_path_error(self, op: str, path: FastPath, exc: BaseException) -> None:
        if path.name not in self._warned_paths:
            self._warned_paths.add(path.name)
            import warnings

            warnings.warn(
                f"PyOverdrive fast path {path.name!r} raised "
                f"{exc!r}; falling back to stock NumPy. Disable "
                f"it with pyoverdrive.disable_path"
                f"({path.name!r}).",
                RuntimeWarning,
                stacklevel=4,
            )
        if self._debug:
            print(f"[gearbox] {op} -> {path.name} RAISED {exc!r}; falling back to stock")

    def decide(self, op: str, args: tuple, kwargs: dict) -> tuple[str, str]:
        """(chosen implementation, reason code) without executing the op."""
        if op not in self._paths:
            return "stock", "operation-not-supported"
        for path in self._paths[op]:
            if not path.enabled:
                continue
            try:
                if path.applicable(args, kwargs):
                    return path.name, "predicate-accepted"
            except Exception:
                return "stock", f"predicate-error:{path.name}"
        return "stock", "no-applicable-path"

    # -- activation (experimental patching route; spec section 10.5) ------

    def patch(self, operations: list[str] | None = None) -> None:
        import numpy

        targets = operations or self.supported_operations()
        for op in targets:
            if op in self._stock:
                continue  # already patched; idempotent
            module, attr = self._resolve(numpy, op)
            stock = getattr(module, attr)
            self._stock[op] = stock
            wrapper = self._make_wrapper(op, stock)
            setattr(module, attr, wrapper)
        self.patched = bool(self._stock)

    def unpatch(self) -> None:
        import numpy

        for op, stock in self._stock.items():
            module, attr = self._resolve(numpy, op)
            setattr(module, attr, stock)
        self._stock.clear()
        self.patched = False

    def stock_fn(self, op: str) -> Callable:
        """The unpatched implementation (valid while patched)."""
        if op in self._stock:
            return self._stock[op]
        import numpy

        module, attr = self._resolve(numpy, op)
        return getattr(module, attr)

    @staticmethod
    def _resolve(numpy_module, op: str):
        parts = op.split(".")
        if parts[0] != "numpy":
            raise ValueError(f"unsupported operation root: {op}")
        obj = numpy_module
        for p in parts[1:-1]:
            obj = getattr(obj, p)
        return obj, parts[-1]

    def _make_wrapper(self, op: str, stock: Callable):
        import functools

        gearbox = self
        active = self._active.setdefault(op, [])  # shared list, mutated in place
        on_error = self._on_path_error

        # The body of _run_active, inlined: this closure IS the hot path, and
        # every call level costs ~100 ns on a call that may itself be 450 ns.
        @functools.wraps(stock)
        def overdrive_wrapper(*args, **kwargs):
            for path in active:
                try:
                    if not path.applicable(args, kwargs):
                        continue
                except Exception:
                    continue
                try:
                    result = path.run(*args, **kwargs)
                except StockRaised as sr:
                    raise sr.original
                except Exception as exc:
                    on_error(op, path, exc)
                    continue
                if gearbox._debug:
                    print(f"[gearbox] {op} -> {path.name}")
                return result
            if gearbox._debug:
                print(f"[gearbox] {op} -> stock (no applicable path)")
            return stock(*args, **kwargs)

        overdrive_wrapper.__pyoverdrive__ = True
        _mirror_ufunc_attributes(stock, overdrive_wrapper)
        return overdrive_wrapper

    def set_debug(self, value: bool) -> None:
        self._debug = bool(value)


# Patching a ufunc (np.sin, np.exp, ...) replaces a np.ufunc object with a
# Python function. Duck-typed callers read ufunc attributes and call ufunc
# methods on the public name (np.sin.at(...), np.add.reduce, .nin/.types);
# mirroring them keeps those callers working. `isinstance(np.sin, np.ufunc)`
# still goes False while patched: that is a documented limit of the
# experimental patching route (spec 10.5), not something to paper over.
_UFUNC_MIRRORED_ATTRS = (
    "nin", "nout", "nargs", "ntypes", "types", "identity", "signature",
    "at", "reduce", "reduceat", "accumulate", "outer", "resolve_dtypes",
)


def _mirror_ufunc_attributes(stock, wrapper) -> None:
    try:
        import numpy

        if not isinstance(stock, numpy.ufunc):
            return
    except ImportError:  # pragma: no cover
        return
    for attr in _UFUNC_MIRRORED_ATTRS:
        if hasattr(stock, attr) and not hasattr(wrapper, attr):
            setattr(wrapper, attr, getattr(stock, attr))


GEARBOX = Gearbox()


def explain(op: str, *args, **kwargs) -> tuple[str, str]:
    return GEARBOX.decide(op, args, kwargs)


def set_debug(value: bool) -> None:
    GEARBOX.set_debug(value)


def enable_path(name: str) -> None:
    GEARBOX.set_path_enabled(name, True)


def disable_path(name: str) -> None:
    GEARBOX.set_path_enabled(name, False)
