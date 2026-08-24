"""System CPU busy fraction, dependency-free (Windows + Linux).

Why this exists: the first PyRallel calibration battery (2026-08-23) ran while
other agent sessions held the machine at 65-96% CPU. Single-thread baselines
were unaffected (one core was always free) but every multi-thread candidate
was starved, so thread-scaling speedups came out ~2x lower than the quiet-box
OPP-000008 run. A Dyno result must carry the load it was taken under, or a
reader cannot tell a real crossover from a contended one.
"""

from __future__ import annotations

import sys
import time


def _windows_times() -> tuple[float, float, float] | None:
    import ctypes
    from ctypes import wintypes

    class FILETIME(ctypes.Structure):
        _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]

    idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
    ok = ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    )
    if not ok:
        return None

    def val(ft: FILETIME) -> float:
        return float((ft.hi << 32) | ft.lo)

    # kernel time includes idle time on Windows
    return val(idle), val(kernel), val(user)


def _linux_times() -> tuple[float, float] | None:
    try:
        with open("/proc/stat", encoding="ascii") as fh:
            first = fh.readline().split()
    except OSError:
        return None
    if not first or first[0] != "cpu":
        return None
    vals = [float(v) for v in first[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)  # idle + iowait
    return idle, sum(vals)


def cpu_busy_fraction(interval: float = 0.5) -> float | None:
    """Fraction [0, 1] of all logical CPUs busy over ``interval`` seconds.

    Measures the WHOLE machine, including this process; call it before the
    timed work starts and after it ends, so the reading is foreign load.
    Returns None where unsupported (macOS without psutil), never raises.
    """
    try:
        if sys.platform == "win32":
            a = _windows_times()
            time.sleep(interval)
            b = _windows_times()
            if a is None or b is None:
                return None
            idle = b[0] - a[0]
            total = (b[1] - a[1]) + (b[2] - a[2])
            return max(0.0, min(1.0, 1.0 - idle / total)) if total > 0 else None
        a = _linux_times()
        time.sleep(interval)
        b = _linux_times()
        if a is None or b is None:
            return None
        idle = b[0] - a[0]
        total = b[1] - a[1]
        return max(0.0, min(1.0, 1.0 - idle / total)) if total > 0 else None
    except Exception:
        return None
