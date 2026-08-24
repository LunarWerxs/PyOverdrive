"""PyOverdrive: an independent adaptive accelerator for NumPy.

NumPy at full throttle.

    import numpy as np
    import pyoverdrive

    pyoverdrive.enable()

Supported operations are routed through the Gearbox dispatcher, which selects
a verified fast path or falls back to stock NumPy. Unsupported, ambiguous, or
unprofitable cases always fall back.

Controls:

- ``pyoverdrive.disable()`` restores stock NumPy exactly.
- ``PYOVERDRIVE_DISABLE=name1,name2`` (env) or ``pyoverdrive.disable_path(name)``
  kill-switches individual fast paths.
- ``PYOVERDRIVE_DEBUG=1`` or ``pyoverdrive.set_debug(True)`` prints one line per
  dispatch decision with a reason code.
- ``PYOVERDRIVE_THREADS=N`` caps the PyRallel thread pool (default
  ``min(cpu_count, 16)``); ``PYOVERDRIVE_THREADS=1`` switches every parallel
  fast path off while leaving the serial ones active.
- ``pyoverdrive.explain(op, *args, **kwargs)`` reports which path Gearbox would
  choose without executing anything.
- ``pyoverdrive.status()`` / ``report()`` show what is registered, enabled and
  patched; ``configure(threads=, disable=, enable=, debug=)`` sets the knobs
  in one call; ``selfcheck()`` (or ``python -m pyoverdrive --selfcheck``)
  proves every fast path against stock NumPy on this machine.
- ``pyoverdrive.calibrate()`` (or ``python -m pyoverdrive --calibrate``)
  probes the calibration-gated fast paths on this machine and enables only
  those that measured a win here, persisted per machine fingerprint. With
  no calibration file the gated paths stay off.

Activation currently patches module-level NumPy functions (the experimental
route; spec section 10.5). The public extension-hook route is under
investigation and monkey-patching is not the intended permanent architecture.
PyOverdrive is not affiliated with or endorsed by NumPy or NumFOCUS.
"""

from __future__ import annotations

from .dispatcher.gearbox import (
    GEARBOX,
    FastPath,
    disable_path,
    enable_path,
    explain,
    set_debug,
)
from . import fastpaths as _fastpaths

__version__ = "0.0.1.dev0"

_fastpaths.register_all(GEARBOX)

from . import calibration as _calibration  # noqa: E402

_calibration.apply(GEARBOX)  # per-machine gates; no file -> gated paths stay off

from .calibration import calibrate  # noqa: E402
from .diagnostics import configure, report, selfcheck, status  # noqa: E402


def enable(operations: list[str] | None = None) -> None:
    """Activate PyOverdrive for the given operations (default: all supported)."""
    GEARBOX.patch(operations)


def disable() -> None:
    """Deactivate PyOverdrive and restore stock NumPy functions exactly."""
    GEARBOX.unpatch()


def enabled() -> bool:
    return GEARBOX.patched


def supported_operations() -> list[str]:
    return GEARBOX.supported_operations()


__all__ = [
    "enable",
    "disable",
    "enabled",
    "explain",
    "set_debug",
    "enable_path",
    "disable_path",
    "supported_operations",
    "calibrate",
    "configure",
    "status",
    "report",
    "selfcheck",
    "FastPath",
    "__version__",
]
