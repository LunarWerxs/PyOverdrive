# Batch 14: five shipped losses, and the reason none of them were caught (2026-08-25)

Batch 14 shipped no new fast path. It re-derived two whole calibration
tables, fixed five cells that were dispatching into a LOSS, and closed the
structural gap that let all five hide. 45 always-on families / 69 registered
paths (down from 71: `pyrallel_sqrt` and `pyrallel_divide` left the project).

Everything below was measured end to end through the public API with the
result consumed, one cell per process, on the idle Intel box
(fp `9bbe7063c555`) unless stated otherwise.

## 1. The threaded thresholds were set by a coin flip

The whole PyRallel table - both families - rested on a measurement artifact.

On a hybrid CPU the scheduler places a single-threaded process on a P-core
or an E-core and it stays there. Asked for the same `np.sin` float64 n=1e5
baseline in 25 fresh processes, the box answered **344 us fifteen times and
497 us ten times**, 1.44x apart, with almost no spread inside either group.
A THREADED candidate spans cores and averages over the split, so the flip
moves the denominator of every speedup and never the numerator - up to
1.44x, always in the candidate's favour.

What makes it nasty is that it is perfectly reproducible *within* a process.
More samples, medians, interleaved sides and idle-machine checks all miss
it. Two other suspects were ruled out by measurement first: candidate vs
dispatched agree, and blocked vs interleaved timing agree.

Full write-up: `docs/research/hybrid-cpu-baseline-coin-flip.md`.

**Affinity is not the fix.** Pinning to the fast class strips the candidate
of its parallelism (1390 us across four workers on eight physical cores IS
the serial time), so a pinned ratio is wrong in the other direction. The
working repair is to leave affinity alone and REJECT the draws that landed
on a slow core, re-drawing the process until one lands fast.

Re-derived with `tools/calibrate_dispatch.py`: **15 of 16 unary rows moved**,
only `exp` float64 survived unchanged. `np.sqrt` left the family - bandwidth
bound, clears 1.3x at no measured size, and had been shipping at 1.05x at
its own floor. The binary family fared worse: every float32 row and all of
`np.divide` failed, and what survives clears the bar by 0.01-0.08x, so its
floors come from TWO independent sweeps with the worse reading kept.

A second unary sweep moved no floor, which is the evidence that the
corrected method is itself stable, and it reached 3e7 so `cos`/`exp`/`log`
float32 came back at 1e7.

## 2. `--calibrate` now covers the threaded families

One machine is not enough for numbers this hardware-dependent, and the
answer is not to promise a second box some day. `python -m pyoverdrive
--calibrate` re-times every threaded row at its own floor on the machine it
is run on and drops the ones that do not pay there. It only ever removes
rows - finding where a row STARTS paying needs the full sweep, not a probe.

Two guards, both required:

- Each cell runs in a FRESH SUBPROCESS that re-draws until it lands on a
  fast core. Every other probe in `calibration.py` compares single-threaded
  against single-threaded, so the core class cancels out of the ratio; that
  cancellation is what makes an in-process probe sound, and it fails the
  moment one side is threaded.
- A row is dropped only when TWO readings AGREE. There is no portable way to
  ask whether a machine is busy, and disagreement is exactly what a busy one
  produces, so disagreement keeps the shipped row.

Verified live on both boxes: nothing dropped on either.

`_cpuclass` moved from `lab/` into the package (lab is not in the wheel) and
had to be hardened - it called the uniform Zen 4 box "HYBRID, 1 slow CPU"
because another session had work pinned to CPU 1. A slow CLASS must now be
at least 2 CPUs and at least 1/8 of them; anything smaller is reported as
contended outliers, by name.

## 3. The guard that cost more than the work it guarded

`np.linalg.inv` computed the determinant in its PREDICATE for the
conditioning check, and the run computed it again - plus a separate
finiteness scan of the whole stack. At batch 4096 that guard cost
**128.5 us against 25.8 us** for the entire 2x2 inverse it was protecting.

Fused into the run, the way det/slogdet/solve already are. The finiteness
test now falls out of the conditioning scale for free: `np.max` propagates
NaN and keeps inf, so max|a| is non-finite exactly when some entry is.

Two more wins fell out of the same area:

- det/slogdet/solve computed that scale TWICE per call - the shared helper
  returned it, the caller discarded it, the guard recomputed it.
- max|entry| folded over the entry views beats reducing over the last two
  axes by 4.6-11x at scale, because numpy handles a multi-axis reduction
  over a tiny trailing shape badly.

**That last one has a crossover and I walked into it**: folding
unconditionally regressed det 3x3 at its own floor from 1.00x to 0.78x,
because at small batches the 2*d*d-1 numpy calls are the whole cost. Caught
by A/B-ing against the committed version on the idle box before shipping.
The crossover is now a measured per-dimension table (`_FOLD_FROM`).

Net at the floors: det 2x2 0.97x -> 1.37x, det 3x3 1.00x -> 1.23x,
slogdet 2x2 1.13x -> 1.41x, slogdet 3x3 1.01x -> 1.19x, inv 2x2 3.2x ->
11.5x at batch 10k.

## 4. Five cells were losing where nothing was looking

This is the structural finding, and it is the one to carry forward.

`tools/verify_no_pessimization.py` proved no path is slower than stock. It
did that on **one canonical input per path**, and every canonical input sits
near the BOTTOM of what its path accepts. So:

- **Upward**, three cells were losing at the TOP of their window: det 3x3
  1.01x at 1e5, slogdet 3x3 0.84x at 1e5, slogdet 4x4 0.85x at 3e4. det 3x3
  and slogdet 3x3 had no upper cap at all; the two 4x4 caps came down from
  3e4 to 1e4.
- **Downward**, two paths had no floor at all and kept accepting: `np.inner`
  on stacked operands ran at **0.38x** (2.6x slower than stock), and
  `np.histogram2d` at **0.75x**.

**A path can be honest where it is checked and lose where it is not.**

`np.inner` deserves its own note. Its canonical input, `(4,5,64) x (32,64)`,
turned out to be *the smallest shape in the sweep that wins* - every shape
below it lost. Nobody chose that adversarially; whoever picked the fixture
tried a few shapes, kept one where the thing clearly worked, and that
fixture became the sample the safety check ran on forever. Treat a
hand-picked fixture as evidence about the author, not about the range.

Its wins and losses also INTERLEAVE - `(4,256,512)` is 0.43x while
`(20,16,512)` is 1.23x - so no function of volume, output size or
contraction length separates them. When a regime will not separate, the gate
has to be restrictive rather than clever: it admits only the corner where
every measured cell won (1.27x-6.98x) and leaves the rest on stock. That
forfeits genuine wins tangled up with the losses, and that is the right
trade - dispatching into 0.38x is worse than declining a 3x.

`np.histogram2d` is the same bug in a different shape: **a threshold on the
wrong axis**. It gated on BIN count alone, but its cost scales with the bins
it allocates and clears while stock's scales with the samples it walks. Few
samples into many bins was its losing corner and the gate was not measuring
in that direction at all. Sample floor added at 2000 (1.57-2.02x there; 200
samples measured 0.75-0.81x).

## 5. What the sweep covers now, and what it still does not

`tools/verify_no_pessimization.py`:

- one cell PER DTYPE where a path's table is dtype-keyed (77 cells, was 68
  paths) - this closed the hole that let `pyrallel_subtract` pass at 1.13x
  on float64 while its float32 row ran at 0.97x;
- `--sizes` judges every cell at 3x/10x/30x/100x its canonical size AND at
  1/3, 1/10, 1/30, 1/100 of it (693 cells);
- a red is re-measured in a second process and only reported if it
  reproduces;
- only on a fast core, on a hybrid CPU.

**Still not covered, and stated in the tool itself:** only the LEADING AXIS
is scaled, uniformly across operands that share it. A loss that needs a
particular aspect ratio - a long contraction against few rows, which is
exactly what `np.inner`'s bad corner looked like - will not appear from
scaling alone. The instrument for that is a shape sweep of the specific
path, not this.

## Audit method, and what it cost

The five losses came out of an audit of all 44 fast-path modules against the
defects already found in three of them (guard-in-predicate, candidate-level
threshold, unconsumed-allocation flattery, extrapolated floor, margin inside
the noise, one-dtype coverage). Six findings survived adversarial
refutation; **measurement then confirmed two and refuted three**:

- `np.vectorize` needs no floor - it wins 3.1x even at size 1, because stock
  is slow at every size;
- `intersect1d`'s 1.19x at its floor was a contended reading - 1.50x idle;
- `svd_small_batch`'s "PROVISIONAL" calibration label was pessimistic, not
  wrong - 2.38x at BATCH_MIN.

Both notes now carry the idle-box number. The lesson worth keeping is that
an audit's job is to produce CANDIDATES and only measurement closes them:
half of these were wrong.

## State at the end of the batch

2207 tests here / 2208 on the Intel box, 69 registered paths (67 always-on),
693 sweep cells all >= 1.0x, all six CI jobs green, both repos identical
(`TREE DIFF: CLEAN`).
