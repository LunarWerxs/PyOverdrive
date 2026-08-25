"""Which logical CPUs are fast ones, measured rather than assumed.

WHY THIS EXISTS. A speedup is a ratio, and on a hybrid CPU the denominator
is a coin flip. This box (i7-12700K: 8 performance cores + 4 efficiency
cores) was asked for the same single-threaded np.sin float64 n=1e5 in 25
fresh processes and answered in two populations:

    15 runs at 344 us      10 runs at 497 us      1.44x apart

with almost no spread inside either. The scheduler places a single-threaded
process on a P-core or an E-core and it stays there. A THREADED candidate
spans several cores and averages over the difference, so the quirk moves one
side of the ratio and not the other.

That is not a small error bar, it is the whole measurement. The committed
PYRALLEL-CAL evidence for that exact cell records stock at 490.8 us and
pyrallel_4t at 307.8 us, a 1.59x win, on a run whose own conditions block
says the box was idle. Re-measured a day later the candidate was unchanged
at ~302 us and stock was 344 us - the same cell, 1.10x. Nothing regressed;
the earlier baseline had simply been handed an E-core. Thresholds derived
from that evidence dispatch at sizes where the path does not pay.

So the lab pins its measurements to one class of core and RECORDS which,
and the classification is measured on the machine in front of us rather than
read off a hardcoded model table - hybrid layouts differ per vendor and per
generation, and a wrong assumption here is invisible and total.

DO NOT PIN THE MEASUREMENT ITSELF. The obvious repair - set the process
affinity mask to the fast class and measure inside it - fixes the baseline
and destroys the candidate, measured on this box at n=3e5 with 4 threads:

    mask                      stock        4-thread     ratio
    unpinned (all 20)        1220.5 us      894.9 us    1.36x
    fast class (0-15)        1299.7 us     1395.5 us    0.93x
    one per P-core (evens)   1302.3 us     1390.2 us    0.94x

Stock barely moves and the THREADED side loses its parallelism outright -
1390 us across four workers on eight distinct physical cores is the serial
time. Windows does not spread pool threads sensibly once a process affinity
mask is set, so a pinned ratio is wrong in the opposite direction and would
have raised every threshold far too high.

What works instead is to leave affinity alone and REJECT the runs that drew
a slow core: a fresh process stays on the class it was given (that is what
the bimodal 344/497 split means), so a cheap probe at process start says
which class this one is on, and a cell measured on a fast one is both
reproducible and conservative. ``pin_to`` remains because CLASSIFYING the
CPUs requires it; measuring a candidate does not.

Used by the lab tools AND by ``pyoverdrive --calibrate``, which has the same
problem in the user's hands: its probes compare a threaded candidate against
a single-threaded baseline, so on a hybrid machine a probe process that drew
an efficiency core would enable or refuse a path on a coin flip. Nothing
here runs during normal import or dispatch, and PyOverdrive NEVER sets
affinity for a caller - ``pin_to`` exists only so ``measure_cpus`` can time
one CPU at a time while classifying.
"""

from __future__ import annotations

import ctypes
import os
import platform
import statistics
import timeit

import numpy as np

_PROBE_N = 40_000
_ROUNDS = 5


def can_pin() -> bool:
    return hasattr(os, "sched_setaffinity") or platform.system() == "Windows"


def pin_to(cpus: list[int]) -> bool:
    """Restrict this process to ``cpus``. True if it took effect."""
    if not cpus:
        return False
    if hasattr(os, "sched_setaffinity"):  # Linux
        try:
            os.sched_setaffinity(0, set(cpus))
            return True
        except OSError:
            return False
    if platform.system() != "Windows":
        return False
    mask = 0
    for c in cpus:
        mask |= 1 << c
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    return bool(kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask))


def _probe_us() -> float:
    """A small compute-bound kernel; whatever core we are pinned to runs it."""
    x = np.linspace(0.0, 6.0, _PROBE_N)

    def fn():
        np.sin(x)

    for _ in range(10):
        fn()
    return min(timeit.timeit(fn, number=40) / 40 for _ in range(_ROUNDS)) * 1e6


def measure_cpus(passes: int = 2) -> dict[int, float]:
    """Time the probe pinned to each logical CPU in turn, best of `passes`.

    Two passes, keeping the BEST time per CPU, because one pass cannot tell
    a slow core from a busy one. Measured on the AMD reference box while it
    was loaded: a single pass reported CPU 1 as 1.52x slower than the other
    31 and the machine as HYBRID, which is false - it is a uniform Zen 4 and
    reports `spread 1.061x` when idle. A genuinely slower CORE is slow in
    every pass; a contended one recovers in whichever pass the interference
    is not there, and taking the best of two is the estimator that separates
    them. Cheap insurance: the whole classification is a couple of seconds.

    Must run in a process that is free to change its own affinity, and it
    restores the original mask before returning.
    """
    n = os.cpu_count() or 1
    original = list(range(n))
    if hasattr(os, "sched_getaffinity"):
        original = sorted(os.sched_getaffinity(0))
    out: dict[int, float] = {}
    for _ in range(max(1, passes)):
        for cpu in range(n):
            if not pin_to([cpu]):
                continue
            t = _probe_us()
            if cpu not in out or t < out[cpu]:
                out[cpu] = t
    pin_to(original)
    return out


def classify(times: dict[int, float] | None = None) -> dict:
    """Split the logical CPUs into a fast class and, if present, a slow one.

    Two populations are declared only when the largest gap between
    consecutive sorted timings exceeds 15%; below that the machine is treated
    as uniform, which is the correct answer for a non-hybrid CPU and keeps
    this from inventing a split out of ordinary noise.
    """
    times = measure_cpus() if times is None else times
    if not times:
        return {"hybrid": False, "fast": [], "reason": "affinity unavailable"}
    ordered = sorted(times.items(), key=lambda kv: kv[1])
    vals = [v for _, v in ordered]
    gaps = [(vals[i + 1] / vals[i], i) for i in range(len(vals) - 1)]
    ratio, idx = max(gaps) if gaps else (1.0, 0)
    n_slow = len(vals) - (idx + 1)
    # A CLASS of cores is a design decision, so it comes in plausible sizes.
    # One slow CPU out of 32 is not a class, it is a busy CPU: the AMD
    # reference box reported exactly that (CPU 1, 1.47x slower, "HYBRID")
    # while another session had work pinned there, and it is a uniform Zen 4.
    # Every real hybrid layout has at least two efficiency cores and they are
    # a real fraction of the machine - 4 of 20 on the Intel box, 4 of 8 on an
    # M1 - so anything smaller is treated as interference and the machine is
    # reported uniform. Erring this way is safe: a missed class costs some
    # measurement noise, an invented one tells the user a false thing about
    # their hardware and makes every probe re-draw away from a good CPU.
    implausible = n_slow < 2 or n_slow * 8 < len(vals)
    if ratio <= 1.15 or implausible:
        return {
            "hybrid": False,
            "fast": sorted(times),
            "spread": round(vals[-1] / vals[0], 3),
            "outlier_cpus": (sorted(c for c, _ in ordered[idx + 1:])
                             if ratio > 1.15 else []),
            "probe_us": {c: round(t, 1) for c, t in ordered},
        }
    fast = sorted(c for c, _ in ordered[: idx + 1])
    slow = sorted(c for c, _ in ordered[idx + 1:])
    return {
        "hybrid": True,
        "fast": fast,
        "slow": slow,
        "class_ratio": round(statistics.median(vals[idx + 1:])
                             / statistics.median(vals[: idx + 1]), 3),
        "probe_us": {c: round(t, 1) for c, t in ordered},
    }


def fast_cutoff(info: dict) -> float | None:
    """Probe time separating the fast class from the slow one, or None on a
    uniform machine. A process whose ``probe_us()`` comes in under this drew
    a fast core."""
    if not info.get("hybrid"):
        return None
    fast = [t for c, t in info["probe_us"].items() if c in set(info["fast"])]
    slow = [t for c, t in info["probe_us"].items() if c in set(info["slow"])]
    if not fast or not slow:
        return None
    return (max(fast) + min(slow)) / 2


def probe_us() -> float:
    """This process's probe time, for comparison against ``fast_cutoff``."""
    return _probe_us()


def describe(info: dict) -> str:
    if not info.get("fast"):
        return "cpu classes: affinity unavailable; measurements are unpinned"
    if not info["hybrid"]:
        note = ""
        if info.get("outlier_cpus"):
            note = (f"; {len(info['outlier_cpus'])} slow outlier(s) "
                    f"{info['outlier_cpus']} look contended, not a core class")
        return (f"cpu classes: uniform ({len(info['fast'])} logical CPUs, "
                f"spread {info.get('spread', 1.0)}x{note})")
    return (f"cpu classes: HYBRID - {len(info['fast'])} fast CPUs "
            f"{info['fast']}, {len(info['slow'])} slow CPUs {info['slow']}, "
            f"{info['class_ratio']}x apart")


if __name__ == "__main__":
    got = classify()
    print(describe(got))
    for cpu, t in got.get("probe_us", {}).items():
        print(f"  cpu {cpu:3d}  {t:8.1f} us")
