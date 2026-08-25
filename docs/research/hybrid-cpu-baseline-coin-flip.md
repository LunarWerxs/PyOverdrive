# The single-threaded baseline is a coin flip on a hybrid CPU

**Status:** measured, acted on. Found 2026-08-24 while re-deriving the PyRallel
thresholds end to end.

## The number that did not reproduce

`benchmarks/results/PYRALLEL-CAL/9bbe7063c555.json`, recorded 2026-08-24 with
its own `conditions` block reporting the box idle (`cpu_busy_before` 0.3%,
`contended: false`), says for `sin_float64_n100000`:

| variant | median |
| --- | --- |
| `numpy.sin` (baseline) | 490.8 us |
| `pyrallel_4t` | 307.8 us |

a **1.59x** win, and `SUPPORTED["sin"][float64] = 100_000` is set from it.

Re-measured on the same machine, same NumPy, same Python, one day later:

| variant | median |
| --- | --- |
| `numpy.sin` | 344 us |
| `pyrallel_4t` | ~302 us |

**1.10x.** The candidate is unchanged to within 2%. The entire difference is
the baseline, and it is 43% faster than the committed number.

Nothing regressed and nothing improved. The first run drew a different kind of
CPU core than the second.

## The mechanism

The box is an i7-12700K: 8 performance cores (16 logical, hyperthreaded) plus
4 efficiency cores. `tools/probe_baseline_stability.py` asked 25 **fresh
processes** for the same single-threaded `np.sin` and got two populations, not
one noisy one:

```
min 343.8 us   median 344.5 us   max 498.2 us   spread 1.45x
344 344 344 344 344 344 344 344 344 344 344 344 344 345 345 | 492 492 493 494 497 497 497 497 498 498
```

Fifteen runs at 344, ten at 497, **1.44x apart, with essentially no spread
inside either group**. A process is placed on one class of core and stays
there.

`lab/dyno/cpuclass.py` times a fixed kernel pinned to each logical CPU in turn
and finds the classes directly rather than assuming a vendor layout:

```
HYBRID - 16 fast CPUs [0..15], 4 slow CPUs [16,17,18,19], 2.378x apart
  cpu 0-15    134.3 - 138.9 us
  cpu 16-19   325.0 - 326.3 us
```

**Why it hits one side of the ratio only:** a single-threaded baseline is one
thread on one core, so it inherits that core's class for the whole run. A
threaded candidate spreads over several cores and averages across the
difference. So the same hardware quirk moves the denominator of every speedup
and leaves the numerator alone.

The error is not small and it is not symmetric: it is up to **1.44x, always in
our favour**, on exactly the measurement that decides whether a fast path
ships.

## Affinity is not the fix

The obvious repair - pin the process to the fast class and measure inside it -
fixes the baseline and destroys the candidate. Measured at n=3e5, 4 threads:

| affinity mask | stock | 4-thread | ratio |
| --- | --- | --- | --- |
| unpinned (all 20) | 1220.5 us | 894.9 us | 1.36x |
| fast class (0-15) | 1299.7 us | 1395.5 us | 0.93x |
| one per P-core (evens) | 1302.3 us | 1390.2 us | 0.94x |

Stock barely moves; the threaded side loses its parallelism outright. 1390 us
across four workers on eight distinct physical cores *is* the serial time.
Windows does not distribute pool threads sensibly once a process affinity mask
is set, so a pinned ratio is wrong in the opposite direction and would have
pushed every threshold far too high.

## What is done instead

Leave affinity alone and **reject the draws that landed on a slow core**. A
fresh process keeps the class it was given, so a cheap probe at process start
says which one this is; if it is slow, the cell is declined and the parent
spawns again. Both `tools/calibrate_dispatch.py` and
`tools/verify_no_pessimization.py` do this, and both record it.

Judging on the **fast** class is also the conservative choice, not merely the
reproducible one: stock is quickest there, so a threaded path has the least to
offer, and a cell that clears its target on a P-core clears it everywhere.

The classifier declines to invent a split when there is not one - the second
calibration box (AMD Zen 4, 32 logical CPUs) reports `uniform, spread 1.061x`,
so the 15% gap rule does not false-positive on ordinary noise.

## What it invalidated

Re-derived end to end on fast cores only, the shipped PyRallel table was wrong
in **15 of its 16 rows**, and `sqrt` never reached the 1.3x it is supposed to
guarantee at any measured size on either dtype - at its own shipped floor of
1e6 float64 it runs at **1.05x**, a dispatch that cannot pay for itself. Only
`exp` float64 survived unchanged. Per-row detail is in
`docs/research/opportunities/OPP-000008.md`; the 96 raw cells are in
`benchmarks/results/PYRALLEL-DISPATCH-CAL/9bbe7063c555.json`.

The threaded BINARY family (`np.add` and friends) came from the same battery
and fared worse. At its old 1e6 floors it delivered **1.04-1.20x** against a
promised 1.3x, and `subtract` float32 at its 3e6 floor ran at **0.97x** - a
dispatched loss. It also needed a second rule: those wins are bandwidth, they
cross 1.3x only between 1e7 and 3e7 elements, and the run-to-run spread there
is as wide as the margin (one sweep read `subtract` float64 at 1.23x, 1.14x,
1.33x on consecutive sizes - non-monotone). So its floors come from **two
independent sweeps with the worse reading kept per cell**. Every float32 row
and all of `np.divide` failed that standard and left the family.

There is a lesson in the pairing. The coin flip made the numbers wrong; the
noise made them *unstable*, and those need different repairs. Rejecting
slow-core draws fixes the first. Only re-measuring fixes the second, and it
is the mirror of "make the red reproduce": **a threshold is a green, and a
green whose margin is inside the noise has to reproduce too.**

## The general lesson

A benchmark can be idle, repeatable, low-variance, and still wrong, because
*repeatable within a process* is not *repeatable across processes*. Any ratio
between a single-threaded and a multi-threaded measurement on a hybrid CPU
inherits a per-process coin flip that no amount of in-process re-measurement
can see. Hybrid layouts are now the default on consumer Intel, Apple Silicon
and much of ARM, so this is not an exotic configuration.

Sibling findings, same shape - the number was measured on something adjacent
to what the user experiences: `time-the-consumed-result-not-the-call` (the
call, not the consumed result), and the guard-inside-the-predicate regressions
that `tools/verify_no_pessimization.py` exists to catch (the kernel, not the
dispatched route).
