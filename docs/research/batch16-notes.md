# Batch 16: the calibration TABLE is the coverage unit, not the path (2026-08-25)

Batch 14 found five shipped losses and named the reason none were caught:
the safety sweep judged one canonical input per path, and every canonical
input sits near the bottom of what its path accepts. Batch 15 closed the
aspect-ratio half of that. This batch closes the half neither touched, and
it is the one that was hiding the most.

A path's dispatch gate is usually a TABLE - a floor per matrix dimension, a
window per dtype, a work floor per convolution mode - and every row of it
is a separately measured, separately shipped promise. The sweep built its
input from the path's ONE fixture, so it judged whichever row that fixture
happened to land on and no other.

**Every small-matrix linalg fixture in this project is d=3.** det, slogdet,
solve, inv, cholesky, qr, pinv, svdvals and norm2 therefore shipped their
d=2 and d=4 rows with no end-to-end measurement behind them at all - and
those are different closed-form kernels, not the same code at another size.
The dtype tables were the same story: unique_sort's small-int rows take a
radix route the int64 fixture never touches, and searchsorted's int64 row
had never been measured end to end at all.

Everything below was measured through the public API with the result
consumed, one cell per process, on the idle Intel box (fp `9bbe7063c555`),
python 3.13.3 / numpy 2.5.2 unless stated otherwise.

## 1. The instrument: `--rows`

`tools/verify_no_pessimization.py --rows` emits one cell per shipped
calibration row, and the cells are derived FROM THE SHIPPED TABLE rather
than hand-written. A row added to a table becomes a cell with no edit to
the tool. 79 cells across 21 paths on the first run.

Where a row is a WINDOW, both ends are judged - `#3` at the floor and `#3^`
at the cap - because three of batch 14's five losses were at the TOP of a
window, which scaling up from the floor by a round factor need not reach.

`_table_audit()` is what makes it durable, and it is the part worth
copying elsewhere: every multi-row dict in the package must be claimed by
an axis or listed in `_EXEMPT_TABLES` with a written reason, or `--rows`
fails loudly. A coverage instrument that silently skips what it does not
understand reads exactly like one that found nothing. It immediately caught
two tables the first hand-written list had missed (`parallel_ufunc.SHIPPED`
and its binary twin), which is the whole argument for it.

Every cell dispatches, and every cell matches stock under its own
comparison mode - so what follows is a timing gap, never a correctness one.
(79 cells on the first run; 82 after the edge audit in section 5 moved
three cells onto their row's actual boundary and gave unique's third floor
branch its own.)

## 2. searchsorted's int64 row: a calibration that ROTTED

0.36x at its own floor, and worse as it grows: 0.30x, 0.33x, 0.27x, 0.20x
at 3x/10x/30x/100x. **It never wins anywhere in its admissible region.**

That is a flat contradiction of the row's own evidence comment, which
records 1.51x at exactly that query count. Two suspects: the haystack (the
mechanism it claims is a cache effect, and its gate asks only for a flat
10_000-element haystack, which is 80 KB and fits in L2), or the reading.

`tools/probe_searchsorted_haystack.py` says both, and neither is the
interesting part. Same box, same code, same day, two numpy versions:

| haystack | int64, numpy 2.4.5 | int64, numpy 2.5.2 |
| --- | --- | --- |
| 10k | 0.73x | 0.26x |
| 100k | 1.06x | 0.29x |
| 300k | 1.90x | 0.37x |
| 1M | 2.46x | 0.46x |
| 3M | 3.97x | 0.74x |

**The row was not measured wrong. It rotted.** numpy 2.5 made stock
searchsorted substantially faster and took the entire int64 margin with it.
float64 survived the same change with a thinner margin - 1.77-4.41x on
2.4.5, 1.28-1.97x on 2.5.2 - so the dtype is what separates them: stock's
integer comparison is now cheap enough that the argsort can never be
repaid, while float64 still leaves room.

The row is withdrawn. A supported numpy that loses is a loss, not a case
for a version-conditional gate; the package claims numpy >= 2.0 and 2.5 is
what CI calls latest.

> Corrected later the same day, in section 7: the package no longer claims
> numpy >= 2.0, because that claim turned out to be false in the other
> direction too. The principle stands unchanged - a supported numpy that
> loses is a loss - but "supported" is a floor this project gets to set
> honestly, and setting it is a different act from gating a path on the
> version at runtime, which is still not done anywhere.

Two things the haystack table also settles: the original 1.51x reading used
a LARGER haystack than the query count it was filed under (at matched sizes
2.4.5 gives 1.06x), and the haystack does matter - int64 climbs 0.26x to
0.74x across it - it just never reaches break-even on current numpy.

**This is a failure class this project had not seen.** Every previous loss
was a measurement that was wrong when it was taken. This one was right when
it was taken and became wrong underneath us. Nothing in the repo detects
it, because a sweep only ever runs against whatever numpy the box has. The
cheap part of the answer is shipped: the sweep now prints the python and
numpy versions in its header, so a green can be compared with the next
green. The expensive part - re-sweeping every row against the oldest and
newest supported numpy, or stamping each calibrated row with the version
that measured it - is not, and is the batch's main recommendation.

## 3. relayout: a whole battery measured one shape family

`relayout_blocked#float32` at 0.70x on its own floor. The other two rows
were losing too; their fixture just sat on a size where they did not.

Every case in that path's calibration battery is a power-of-two square or
a power-of-two pair: 256x256, 512x512, 1024x1024, 2048x2048, 4096x4096,
8192x1024, 1024x8192. That is not a neutral sample. A power-of-two leading
dimension is the cache-associativity corner where stock's strided copy
conflict-misses worst, so it is precisely where a tiled copy wins most. The
floors were then written as plain element counts, which generalised a
measurement of one shape family to every shape.

One element off a power of two is all it takes:

| side | float32 | note |
| --- | --- | --- |
| 2047 | 1.76x | |
| **2048** | **3.58x** | power of two |
| 2049 | 1.75x | |
| 886 | 1.00x | |
| **1024** | **2.69x** | power of two |
| 1254 | 1.28x | |

A hypothesis that looked better than it was, recorded because the A/B is
the point: the matched-byte grid put every cell at or above 4M elements at
3.02-5.16x and everything below at 0.89-2.69x, which lines up exactly with
`threads_for()` switching from 8 threads to 16 at 4M. Straddling that
switch by ONE ELEMENT refuted it outright - 2047 (8 threads) 1.76x against
2049 (16 threads) 1.75x, and the same null for float64 and int64. The
apparent thread boundary was the n=2048 spike sitting on top of it.

Walking non-power-of-two sides gives the honest crossing, worst of the
three dtypes at each size: 2.56M 1.21x, 3.24M 1.39x, 4.19M 1.44x, 4.84M
2.25x, 5.76M 2.67x, 6.76M 2.91x, 9.00M 3.23x. All three floors move to 4M
elements (from 262_144 for the two float rows and 1_048_576 for int64),
where the worst measured shape is 1.44x and the typical one is 2-5x.

That **forfeits the genuine 1.5-2.7x that power-of-two squares win between
1M and 4M**, deliberately. It is the np.inner trade from batch 14 made
again: the spike is one machine's cache geometry and the losses around it
are not, so a gate that admits only the corner where every measured shape
wins beats a clever one that tries to keep the spike.

The selfcheck fixture moved one element off the exact square root for the
same reason - `sqrt(4M)` is 2048, and a fixture on the spike reports a
path's best shape as its typical one. So does the row cell. This is the
batch-14 lesson about hand-picked fixtures arriving one level up: it is not
only the person who picks a fixture who can pick a flattering one, it is
also a `sqrt` that lands on a round number.

## 4. What the row sweep did NOT find

Worth stating, because a sweep that only ever reports finds is a sweep
nobody trusts. 77 of 79 rows were clean on the first run, including every
one of the unmeasured linalg dimensions that motivated the instrument:

- det/slogdet d=2 and d=4 at both ends of their windows: 1.42-2.57x.
- cholesky d=2 1.76x, qr d=2 3.37x, pinv d=2 5.16x, svdvals d=2 2.45x,
  norm2 d=2 3.36x - the d=2 kernels are all comfortably fine.
- inv's never-measured float32 row: 7.26x.
- unique_sort's eight dtype rows, radix route included: 1.31-22.40x.
- intersect_sorted's eight: 1.60-11.41x.
- both convolution keyword modes, both char kinds, both int matmul widths,
  both complex widths.

The prior going in was that the linalg dimensions were the risk, since
that family produced three of batch 14's five losses. They were fine, and
the two losses came from paths nobody suspected. An audit produces
candidates; only measurement closes them, and it closes them in both
directions.

## 5. Two more holes the batch closed, and the ones it did not

**Rot now has a detector.** `tools/verify_across_numpy.py` runs the whole
row sweep once per supported numpy and fails on any cell that loses on any
of them. It works only because this package is pure Python: a bare venv
holding nothing but numpy imports it straight from `src/` and runs. It also
flags cells that MOVED more than 25% between versions while still passing,
which is rot in progress rather than rot arrived. A version with no wheel
for the running interpreter is reported SKIPPED with the reason - numpy
2.0.2 has no cp313 wheel, so on a 3.13 box that leg belongs to CI's 3.12
runner and this tool says so rather than implying coverage it lacks.

**A pair of 1-D operands has an aspect ratio too**, and `--shapes` was
skipping every such cell for want of a second axis to move. This batch is
the argument: searchsorted's whole regime lives on haystack-vs-queries, and
every cell of every axis held those two operands in step. 24 cells across
six paths now move the ratio instead. Two guards were needed, and the
second is the interesting one: elementwise operands are ratio-LOCKED, so
moving their ratio is a broadcast error rather than a shape - and the first
cut generated 40 such cells, which reported as "no-dispatch", i.e. exactly
what a well-gated path reports. The parent now asks with a tiny 8-vs-2
probe pair, and the sweep says **invalid-input** where it used to say
no-dispatch. That distinction is the machine-checked version of the
by-hand skip audit batch 15 had to do.

Auditing the row builders against their own tables also caught three cells
sitting in the comfortable middle of the row they were meant to test
(matmul_int at 100x100 when its floor is 50, split_complex nowhere near its
three limits, unique_char_view at 10_000 for a branch whose floor is 300),
and one gate axis `_table_audit()` structurally cannot see: unique's floor
depends on which RETURN was asked for, and those are three scalar constants
rather than a dict. All three branches have cells now.

**Cardinality got an axis too** (`--values`), since that was the obvious
next one and the modules document the sensitivity themselves: every integer
operand is redrawn from 16 distinct values, and from the dtype's whole
range. Floats are deliberately left alone - a float array in these fixtures
is usually a matrix whose VALUES are load-bearing (positive-definite, well
conditioned, a monotone grid), and redrawing it makes a different call
rather than a different distribution.

That axis exposed a trap in the sweep itself: the canonical cell was
re-added to a run only because `SIZE_MULTS` happens to contain 1, so any
axis used on its OWN - `--shapes`, `--values` - silently dropped every
canonical input and returned a green that said nothing about them. The base
cell is always judged now; `--sizes` is unchanged at 693.

### The limit the relayout loss actually exposes

Worth stating plainly, because it is a limit of the whole instrument rather
than a missing axis: **the derived cells inherit the fixture's alignment.**
`--sizes` multiplies by 3/10/30/100 and `--shapes` by 4/16, so a
power-of-two base dimension stays highly 2-aligned through every cell the
sweep can generate. No amount of running it would have found the relayout
spike; a probe that walked n one element at a time did.

The scope of that risk was checked rather than assumed, in both directions:

- Across the 26 calibration batteries, relayout's is the outlier - 80% of
  its size literals are powers of two. The next three (percentile 70%,
  syrk_gram 67%, quantile 62%) are 1-D length sweeps or an unshipped
  candidate, where a 2-D stride cannot be the mechanism.
- Across the sweep's own cells, only 6 have a 2-D-or-deeper input with a
  power-of-two axis at all, and they are BLAS and sort paths (inner,
  split_complex, percentile, quantile) where the packing is the library's
  problem, not a hand-written tiled copy's.

So an "off-by-one alignment" axis was considered and NOT built: it would
touch six cells for a mechanism the evidence says is concentrated in the
one path already repaired. Recorded here so the next person weighs the same
evidence instead of rediscovering it.

Still not covered:
- Layout beyond F-contiguity: every constructed cell is C- or
  F-contiguous, never strided.
- The row cells judge a row's ENDS. A row whose middle loses while both
  ends win would pass `--rows`; `--rows --sizes` samples between them, but
  only at 3x/10x/30x from the floor.
- Scalar gate constants generally. `_table_audit()` enforces coverage of
  every multi-row dict, which is what caught two tables a hand-written list
  had missed - but a threshold that is a bare `int` at module scope is
  invisible to it, and char_view's three unique floors were exactly that.

## 6. The four-axis run that was lost, and the hop that lost it

The batch's closing evidence is one run of all four axes together -
`--rows --sizes --shapes --values`, every cell the instrument can make. It
was started on the idle Intel box over a single ssh hop, in the
foreground, and roughly two hours later THIS box restarted. The run did
not fail. It was never written down: no log, no partial table, no
verdict, nothing to resume from. The only record that it had ever run was
a sentence saying it was running.

That is a defect in the harness, not bad luck, and it has two halves.

**The output was never on disk.** It lived in the pipe back to the
driving machine, so the driving machine WAS the recording medium. Every
long-running instrument here now writes stdout, stderr and its exit code
to files on the far side before anything reads them, and `--attach` reads
them back from any session - including one that never launched the run.
An in-flight run reports as exit 2, so "not finished yet" can never be
mistaken for "passed".

**And the process could not have survived anyway.** Windows OpenSSH puts
the remote command in a JOB OBJECT and tears the whole tree down when the
connection ends, so `Start-Process -PassThru` over ssh is not detached at
all. That was measured here rather than reasoned about: the first
relaunch used it, the hop returned, and the sweep was gone within
seconds, leaving a zero-byte log and a zero-byte stderr - which reads
exactly like a command that never started, and would have been diagnosed
as one. `Win32_Process.Create` spawns outside the job and survives.

Worth stating because it generalises past this repo: **an ssh hop is not
a place to keep a two-hour measurement.** The same shape would have eaten
any of the calibration batteries, and the cost is not the compute, it is
that a lost run leaves nothing to distinguish it from a run that was
never started.

The relaunch also carries `-v`. A two-hour run whose entire output is one
verdict line is not evidence anyone can check; per-cell ratios on disk
are, and they cost nothing.

## 7. The floor was a false claim: numpy>=2.0 becomes numpy>=2.3

Section 2 found a row that rotted as numpy got FASTER. The cross-version
tool built in response was then run for the first time across the whole
supported range, and found the opposite failure: two families that were
never good on the versions at the BOTTOM of the range, and had shipped
that way for as long as they had existed.

The cliff is between 2.2 and 2.3, in both families, with nothing ambiguous
on either side of it:

```
cell                          2.0.2   2.1.3   2.2.6 | 2.3.5   2.4.6   2.5.2
matmul_split_complex          0.05x   0.05x   0.05x | 2.31x   2.07x   2.03x
matmul_split_complex#c128     0.02x   0.02x   0.02x | 1.36x   1.43x   0.70x
matmul_split_complex#c64      0.01x   0.01x   0.01x | 0.90x   0.91x   0.66x
intersect_sorted#int32        0.52x   0.53x   0.53x | 1.21x   1.23x   1.39x
intersect_sorted#int64        0.62x   0.63x   0.62x | 1.32x   1.26x   1.42x
intersect_sorted#uint32       0.53x   0.53x   0.52x | 1.22x   1.23x   1.39x
intersect_sorted#uint64       0.63x   0.61x   0.62x | 1.32x   1.28x   1.56x
```

**The matmul one has a named upstream cause, and it is exact.** numpy
2.3.0: "Enable using BLAS for `matmul` even when operands are
non-contiguous by copying if needed" (gh-23752). That path multiplies
`C.real` and `C.imag`, which are stride-2 views. Below 2.3 those two
matmuls never reached BLAS at all - they fell to numpy's own loop - so the
"two real GEMMs instead of one complex GEMM" argument the whole path rests
on was simply not what was running. 0.01x is not a slightly worse kernel;
it is a different kind of code.

It also means the path's own docstring was wrong about its mechanism: it
claimed BLAS consumed the stride-2 views "natively, no copy anywhere",
which was never true on any numpy. 2.3+ copies them contiguous and then
calls BLAS. The win survives - two copies plus two real GEMMs still beat
upcasting R to complex - but a false mechanism is how a path gets
generalised to a case it cannot serve, so it is corrected in place.

The intersect one is less certain and is recorded as such: numpy 2.3 also
rewrote `np.unique` to try a hash table before sorting (gh-26018), which
is the likeliest thing to have moved stock's side, and the four losing
rows are exactly the WIDE integer ones (the narrow rows take the radix
route and win 3.5-7.7x on every version). Worth knowing that this row's
margin now depends on a stock implementation upstream is actively
changing.

**The decision: move the floor, do not withdraw the paths.** Section 2's
principle is unchanged - a supported numpy that loses is a loss - but
which numpys are supported is a claim this project makes, and it was
making a false one. `numpy>=2.0` was never measured; it was inherited from
the scaffold. Withdrawing instead would forfeit 1.2-2.3x on matmul and
1.2-1.6x on four intersect rows for every user on a current numpy, to keep
a compatibility promise to versions that were being actively harmed by it.
Nothing is gated on the version at runtime, then or now.

This is cheap TODAY and would not be later: the package is 0.1.0 and
unpublished, so the floor costs nothing to move. After release it would
cost a major version.

### What correctness CI cannot see

The floor job in ci.yml carries the comment "if this job breaks, the claim
is wrong and the floor moves". It never broke. Every one of these versions
passes the entire suite, because being 100x slower than stock is not a
wrong answer - it is a correct answer nobody wants. A gate that can only
see wrongness cannot see this class at all, which is the argument for the
cross-version tool existing at all.

### Left open, then closed

`matmul_split_complex#complex64` read 0.90x and 0.91x on 2.3.5 and 2.4.6
and 0.66x on 2.5.2 on the AMD box, which would have been a two-machine-law
failure rather than a version one. It was left unresolved rather than
settled from contended numbers, and that was the right call: the four-axis
sweep on the IDLE Intel box judged the same two row cells at **1.19x
(complex128) and 1.29x (complex64)**. The AMD readings were four other
agent sessions on the box, nothing more. The path is sound above the new
floor; only the floor was wrong.

## 8. The four-axis run, at last: 23 losses, and 19 of them are one thing

The run section 6 lost, re-run detached on the idle Intel box (python
3.13.3, numpy 2.5.2, fp `9bbe7063c555`): **877 cells judged in their own
processes, 1156 skipped, 23 below 1.0x.** Every loss reproduced in a second
independent process before being reported; one candidate (`unique_values_
sort#uint64%d`, 0.93x then 1.17x) did not reproduce and is correctly absent.

Nothing here is a NEW defect. These are cells the instrument could not
build until this batch, on paths that have shipped for weeks.

### The one that is not like the others: isin, on the ratio axis

```
isin_string_hash>16   0.05x        isin_string_hash>4   0.35x
isin_object_hash>16   0.09x        isin_object_hash>4   0.46x
```

0.05x is the worst reading in the package - twenty times slower than the
call it replaces - and it is not a worse kernel, it is the wrong algorithm
for that corner. `>16` means the ratio moved 16x toward MANY elements
against FEW test_elements at constant combined size. numpy's `in1d`
switches method on exactly that ratio: with a small enough test_elements it
evaluates a chain of vectorized equalities in C. The hash route pays a
Python-level hash and lookup PER ELEMENT of the big operand, because these
are object and StringDType arrays. So the fewer things there are to test
against, the better stock gets and the worse we do.

**Both paths gate on `element.size + test.size >= 300` and nothing else.**
A single combined-size number cannot express a ratio, so no value of it
could have caught this. `tools/probe_isin_ratio.py` walks the axis to put
the crossing somewhere measured.

### The other 19: the sort-based set family, in two regimes at once

`unique_sort`, `unique_values_sort` and `intersect_sorted` account for all
the rest, and they lose in exactly two places:

- **Low cardinality** (`%d`, integer operands redrawn from 16 distinct
  values): intersect at 0.68x/0.68x/0.73x/0.74x on int64/uint64/uint32/
  int32 and 0.93-0.94x on the 16-bit rows; unique_values at 0.95x on
  int64/int32/uint32.
- **64-bit at 10x and 30x their canonical size**: 0.80-0.98x across all
  three paths, worst on uint64.

Two regimes, one cause - **and that reading was WRONG. See section 10**,
which put the cardinality axis through every supported numpy and found two
different failures wearing the same number. The original text is left
below because the correction is only legible against it.

> ~~Two regimes, one cause, and it is the cause section 7 already named.
> numpy 2.3 rewrote `np.unique` to try a HASH TABLE before sorting
> (gh-26018). A hash table is at its best precisely when there are few
> distinct values, and it keeps improving relative to a comparison sort as
> n grows. This whole family's advantage is a bet that stock dedups by
> sorting, and upstream is in the middle of taking that bet away. That
> makes it the same failure as section 2's, caught earlier.~~

### What the axes were worth

Every one of the 23 lives on a cell that could not be built before this
batch: `>` and `<` (the 1-D pair ratio) and `%d`/`%u` (cardinality) are
new here, and the `*10`/`*30` losses are row cells crossed with the size
axis, which is also new - `--sizes` alone judged the path's one fixture,
never a table row at 30x. Batch 14 asked what the sweep was not looking
at, and the answer keeps being "an axis", not "a size".

## 9. isin, fixed: the gate now counts what actually decides the winner

The 0.05x and 0.09x cells are closed, and the shape of the fix is worth
keeping because it is not the shape the sweep's cell NAME suggested.

The losing cells are aspect cells - `>4`, `>16` - so the obvious reading is
"the gate needs a ratio". It does not. Stock's `in1d`, given few enough
test_elements, evaluates a chain of vectorized equalities: **one full C
pass per test element**. These paths pay a Python-level hash and lookup per
element of the big operand, because the operands are object and StringDType
arrays. So stock's cost is (number of test elements) passes and ours is one
expensive pass, and what decides the winner is HOW MANY test elements there
are - not how they compare in size to the array.

The probe settles it, because a ratio mechanism and a count mechanism make
different predictions and the object table matches only one of them:

```
elements     1      2      3      4      6      8     12     16     24
   300     0.25x  0.36x  0.50x  0.60x  0.86x  1.08x  1.53x  1.96x  2.78x
 1,000     0.18x  0.29x  0.44x  0.57x  0.88x  1.11x  1.58x  1.94x  2.98x
10,000     0.17x  0.37x  0.51x  0.64x  0.99x  1.26x  1.82x  2.27x  3.52x
100,000    0.16x  0.33x  0.48x  0.61x  0.94x  1.23x  1.77x  2.29x  3.72x
```

The crossing sits at 8 across four orders of magnitude of element count. A
ratio would have moved it by a factor of 300. So one number is the honest
gate: `TEST_FLOOR = 12`, the first column winning everywhere with a margin.

**The string path does not share that number, and assuming it would have
been wrong.** Its crossing walks with the array - 16, 24, 32, 32 - and 32
is still only 1.05-1.10x at the top, so its floor is 48 (worst 1.53x, the
same margin the object floor carries). Two paths, one mechanism, two
numbers 4x apart, because the per-element cost differs between Python
objects and StringDType. Copying numpy's own crossover formula would have
got both wrong for the same reason: it is tuned for fixed-width numeric
dtypes.

A fitted curve for the string path was considered and rejected: four points
do not justify an exponent, and a flat number every measured cell clears is
a better promise than a curve that is right on average. It forfeits the
1.42x available at 24 test elements against 300 elements, deliberately.

Verified on the idle Intel box, `--only isin` across all four axes: the
four losing cells now REFUSE (stock answers, so there is nothing to be
slower than), everything still dispatching wins 1.53x-23.35x, and the run
reports no dispatching path below 1x. Note `<4` and `<16` - the opposite
skew, many test_elements against few elements - were never in danger and
still win 14.31x and 23.35x; the fix costs none of that.

### Still open after this batch

The other 19 losses from section 8 - the `unique_sort` /
`unique_values_sort` / `intersect_sorted` family at low cardinality and at
64-bit large sizes - are NOT fixed here, and the isin fix does not
generalise to them. Worth saying exactly why, because the difference
decides what the next batch can even attempt.

**isin was fixable because its deciding quantity is FREE TO READ.**
`test.size` is an attribute; the gate costs nothing and knows the answer
before dispatching. **Cardinality is not.** Counting distinct values is the
work itself, so a gate cannot ask "are there only 16 distinct values here?"
without paying for the operation it is deciding whether to perform.

That rules out the obvious fix and most of the fallbacks:

- A size threshold cannot separate the regimes: the sizes that lose at low
  cardinality are the same sizes that win at high cardinality. There is no
  value of a size floor that keeps the wins and drops the losses.
- Withdrawing the affected dtype rows would forfeit the high-cardinality
  wins, which are large (`unique_sort#int16` at 17x, `intersect_sorted`
  narrow rows at 3.5-7.7x) and are much of why the family exists.

What is left is a **sampled cardinality estimate** - read a small random
subsample, count distinct values in it, refuse when the estimate says the
input is low-cardinality. That is not a new idea here: the shipped
`searchsorted_sortqueries` gate already samples its queries for disorder
and refuses sorted-looking ones, so the machinery and the precedent both
exist. It needs its own calibration (how large a sample, what estimate,
what error rate is acceptable in both directions) and a demonstration that
the sample cost stays inside the margin.

The 64-bit large-size half may be simpler - that one IS on an observable
axis - but it should not be re-derived before the family's story on the
next numpy is decided, since section 8's evidence is that upstream's
hash-table `unique` is still moving these numbers underneath us.

## 10. The family's story, settled: two failures, and only one is rot

Section 8 called all 19 remaining losses one mechanism - upstream's
hash-table `np.unique` eroding a family that bets on stock sorting - and
recommended pointing the cross-version tool at it. Doing that turned the
hypothesis over: **the tool the recommendation asked for is the tool that
refuted the recommendation.**

The cross-version run needed a new axis to answer it at all. The version
bisect in section 7 ran `--rows` only, at whatever cardinality each fixture
happens to draw, so it had never visited the low-cardinality corner. With
`--values` carried through per version (`--only` keeps it affordable), the
loss count barely moves: **13 cells under 1.0x on numpy 2.1.3, 13 on 2.2.6,
15 on 2.3.5, 12 on 2.4.5.** A mechanism that arrived in 2.3 cannot produce
that. But the per-cell table shows why the total is flat while the contents
are not:

```
cell                          2.1.3   2.2.6 | 2.3.5   2.4.5   2.5.2
intersect_sorted#int16%d      4.32x   4.48x | 0.68x   0.86x   0.95x
unique_sort#int16%d           5.06x   5.59x | 0.75x   0.95x   1.08x
unique_values_sort#int16%d    5.04x   5.54x | 0.75x   0.96x   1.07x
unique_values_sort#uint16%d   1.30x   1.21x | 0.75x   1.01x   1.02x
---------------------------------------------------------------
intersect_sorted#int64%d      0.71x   0.70x | 0.63x   0.67x   0.70x
intersect_sorted#uint64%d     0.72x   0.72x | 0.62x   0.64x   0.68x
intersect_sorted#int32%d      0.68x   0.67x | 0.67x   0.69x   0.74x
unique_sort#int64%d           1.04x   1.04x | 1.02x   1.00x   0.99x
unique_values_sort#int64%d    1.06x   1.06x | 0.98x   0.98x   0.96x
```

**Above the line is rot, and it is severe: a 5x win became a 0.68x loss at
numpy 2.3.** That is the hash-table rewrite, and section 8's mechanism is
right for exactly these rows - the NARROW (16-bit) ones, where our radix
route was beating a sorting stock by 4-5x at low cardinality and now
competes with a hash table instead. Note they are recovering as upstream
tunes it (0.68x -> 0.95x by 2.5.2), which is its own warning: these rows
will keep moving.

**Below the line is not rot at all. Those rows never won.**
`intersect_sorted`'s wide-integer rows measure 0.62-0.75x on every numpy
this project has ever supported, 2.1 through 2.5, and the unique pair sits
at 0.96-1.06x throughout - no margin on any version, ever. Nothing took
these away; they were shipped this way, and no instrument had looked. The
calibration battery behind them drew high-cardinality operands, and the
comment in `intersect_sorted` even says low-cardinality inputs are
"marginal-to-losing" - a sentence that turns out to have been literally
true and never acted on.

So the honest count is not 19 of one thing. It is **six rows of genuine
rot** (the 16-bit family, from 2.3) and **thirteen rows that were never
measured in the regime where they lose**, plus the marginal `*10`/`*30`
size cells discussed below.

### And the fix I recommended in section 9 is the wrong one

Section 9 argued the route was a sampled cardinality estimate, on the
grounds that cardinality is not free to read. That reasoning still holds,
but it stops short of the fact that decides the matter: **the losing
regime is SMALL arrays.** These are ROW cells, and a row cell is measured
at its row's own floor - 64 elements for `unique_sort`'s 32/64-bit rows,
400 combined for `intersect_sorted`'s. Measured directly: `unique` at 16
distinct values is 0.98x at n=64, 1.02x at 400, 1.12x at 1000 and **2.01x
by n=10,000**; `intersect1d` is 0.65x at 400 and 0.90x at 1000. At
n=100,000 the same call wins 2.11x.

A sampling gate cannot be paid for there. Estimating cardinality means
touching a sample, and at n=64 or n=400 any sample large enough to
distinguish 16 distinct values from 400 is a sizeable fraction of the
array - the estimate costs what the operation costs. Sampling works for
`searchsorted`'s disorder gate precisely because that path's floor is
10,000 queries, where 4096 strided pairs are cheap. Here the floor IS the
problem.

**So the fix is the floors, not an estimator.** Every one of these rows has
its floor set from a high-cardinality battery, and the floor has to hold at
the WORST cardinality the row accepts, because at these sizes nothing can
cheaply tell the two apart. `tools/probe_cardinality.py` walks size against
cardinality per dtype to place them.

That is the same shape as this batch's relayout repair: a threshold
measured on one distribution, generalised to all of them, and repaired by
moving it up to where every measured distribution wins - forfeiting the
region in between deliberately.

## 11. The floors move, and what is deliberately left standing

Two floors raised, both because the old number was measured at ONE
cardinality and the row loses at another, and nothing in a gate can see
cardinality cheaply - counting distinct values IS the work.

- **`intersect_sorted`, 32/64-bit: 400 -> 10,000 combined.** At the old
  floor int64 reads **0.69x** - a third slower than the call it replaces -
  and 0.75x at 1,000, 0.99x at 3,000. 10,000 is the first size whose worst
  cardinality clears 1.0x (1.11x int64, 1.77x int32). This forfeits the
  2.0-5.8x that high-cardinality inputs win between 400 and 10,000,
  deliberately, on the same reasoning as relayout's floor in section 3.
- **`unique_sort` / `unique_values_sort`, 32/64-bit: 64 -> 1,000.** At 64
  the worst cardinality is a wash (1.00-1.01x) and at 400 int64 actually
  loses (0.96x). At 1,000 both widths clear 1.0x everywhere measured and
  int32 clears the 1.3x min-win at 1.38x.

Thirty-four tests failed the moment those floors moved, every one a fixture
sized against the old number rather than a behaviour change. The intersect
fixtures now derive their sizes FROM `SIZE_THRESHOLD` so they cannot drift
out from under the gate again, and one unique test that read
`SIZE_THRESHOLD * 4` to mean "below the int8 floor" - true when the 32/64
floor was 64, false at 1,000 - now names the floor it actually means.

### What is left, and why it is not withdrawal

Two residues survive the floors, and neither can be gated away:

- **int64 in the 256-distinct band**: 0.88x at n=10,000 and ~0.97x at
  n=100,000, in both independent grids. It is a band in the MIDDLE of the
  cardinality axis, not a tail, so no floor and no cap removes it.
- **the 16-bit rows at low cardinality**: 0.89-0.92x for unique and
  0.83-0.84x for intersect, at every size their floors admit.

The rule-following move is withdrawal - the project's non-negotiable is
that a path is never slower than stock, and no threshold rescues these.
**I am not withdrawing them, and this is a judgement call rather than a
reading of the data.** The reasoning:

The trade is a 3-12% loss in a narrow cardinality band against 4-9x wins
(int64) and up to 18x (int16) everywhere else on the axis, including the
regime int64 arrays are usually in - an int64 column with 256 distinct
values is a categorical that wanted a smaller dtype. Withdrawing returns
every one of those callers to stock to spare a corner that costs them a
tenth of one call. That is a worse outcome for users than the rule it would
satisfy.

It is also inside the measurement's own resolution. Two grids of 240 cells
agreed within 15% on 232 of them, and 5 of the 8 disagreements moved
15-21%. A 0.88x reading and a 1.0x bar are not far enough apart, at that
scatter, to justify deleting a path that wins 9x elsewhere - and three of
the disagreements were verdict FLIPS caused by an artifact this session
only found because it went looking (section 12).

What would change the answer: a cheap cardinality signal (none exists at
these sizes - see section 10), or evidence the band is wider than measured.
The band is bounded above and below by measured wins, so it is not a
slope.

**Recorded as the owner's to overrule.** The numbers are here, the trade is
stated, and `%d` cells for these rows will keep the sweep honest about the
cost either way.

## 12. The instrument bit: a cell's first measurement is not evidence

Found while deriving section 11's floors, and it nearly set them wrong.

Measured six times in a row on the IDLE Intel box, `unique`/int64 at
n=100,000 with 16 distinct values reads **0.91x once and then 1.73x five
times**, with almost no spread. At 256 distinct: 0.84x, then 1.25x twice.
The AMD box reads 2.04-2.09x for the same cell three times running. One
reading in each sequence disagrees with everything around it and with the
other machine, and it is always the first. Best explanation: our own 800 KB
allocation paying first-touch page faults, while stock's hash table for 16
distinct values is a few hundred bytes and pays none.

The probe's `--repeat` keeps the MINIMUM, deliberately - so without a
warm-up it latched onto that artifact every time. Run with `--warmup`, the
same cell reads 2.05x, 2.04x, 2.08x.

**But the first version of this note was wrong about how far that reaches,
and the correction matters more than the finding.** It claimed the artifact
would have made the floor proposer call four rows unrescuable that were
not. Running the second grid refuted that: 232 of 240 cells reproduce
within 15%, and exactly **three** flip their verdict - all three are int64
at n=100,000, the largest allocation in the grid, which is precisely where
the artifact was measured and nowhere else. Most of the low readings are
the path's real behaviour.

The sweep's own cell settles it independently: `unique_sort#int64%d` reads
0.9198x, 0.9233x, 0.9206x, 0.9272x, 0.9184x across five fresh processes.
As reproducible as a number gets, and losing.

That is the second time in this batch a tidy mechanism was stated before
the run that could refute it had finished, both times in the same
direction. `tools/compare_probe_tables.py` exists so the next such claim
has to survive an arithmetic check first: it reports which cells reproduce,
which move, and loudest, which two runs disagree about the SIGN of the
verdict - the ones that would flip a floor.

## 13. The size axis does not measure size. It TILES.

The floors in section 11 dropped the loss list from 19 cells to 7, and the
survivors are almost all `uint64` `*10`/`*30`/`*100` cells - which sent me
looking for a signed/unsigned difference that does not exist. What is
actually there is a property of the instrument, and it changes how every
`*N` reading in this project should be read.

`_scaled()` grows an array with `np.concatenate([a] * mult)`. **Tiling adds
no new distinct values.** So `unique_values_sort#uint64*30` is not a
30,000-element input: it is the row's 1,000-element fixture repeated thirty
times - cardinality still ~1,000, and now perfectly periodic. The size axis
and the cardinality axis are entangled by construction, and `--sizes` has
been quietly measuring "same values, more repetition" wherever a path's
cost depends on the distribution.

The periodicity is not incidental either. Same values, same count, same
dtype, three orderings:

```
                          n=30,000, ~1,000 distinct
  tiled (what --sizes builds)     int64 1.31x    uint64 1.29x
  the same values shuffled              2.33x          1.85x
  redrawn at random                     1.95x          1.82x
```

Tiling costs 0.5-1.0x against the same multiset in a different order. So a
`*30` cell reports a number that a caller only sees if they passed
`np.tile(...)`, which is a legitimate input but not the one the axis is
named for.

**This is why `uint64` looked like the odd dtype out.** It is not: the
signed and unsigned fixtures differ in span (`_ints` clamps `lo` to 0 for
unsigned, so uint64 draws from half the range int64 does at the same
nominal span), which shifts cardinality slightly, and the tiled cells sit
exactly where that shift matters. Measured directly at equal cardinality
and equal spread, uint64 tracks int64 within scatter - 7.6x against 9.9x at
n=30,000 all-distinct, both comfortable wins.

### What to do about it, and why not tonight

The faithful fix is for the size axis to REDRAW rather than tile, so size
moves independently of cardinality. That is a change to what every existing
`*N` number in the repo means, including the ones batch 14 and 15 acted on,
so it deserves its own batch and a re-run of the calibration it touches -
not a quiet edit at the end of this one. Recorded here as the next
instrument job.

Until then, read every `*N` cell as "this many copies of the canonical
input", not "this many elements".

### And one more caveat on the same cells

The `*N` losses are also the cells most exposed to section 12's cold-first
measurement: every sweep cell is one measurement in a fresh process, so it
is always the cold reading. The same `uint64*30` cell reads 0.89-0.93x
across three fresh processes and 1.29x warm in-process. Both are real - a
caller's first call is cold and their thousandth is warm - but a marginal
`*N` reading is carrying two effects at once, and neither of them is the
one the cell name advertises.

## 14. The size axis, repaired - and what it did to the record

Section 13 named the defect; this is the repair and the audit that had to
come with it, because changing the sweep changes what every `*N` number in
this repo means.

**The change.** A cell whose maker can build its input at a size is now
REBUILT at that size, so distinct values grow with n the way a caller's
array does; `_takes_scale` asks the maker rather than keeping a list beside
it, so a maker that gains the parameter is used with no second edit.
Everything else grows by a seeded resample WITH replacement from its own
values, which keeps the distribution, breaks the periodicity, and re-sorts
anything that arrived sorted. Neither is a tile.

**The sortedness half was a live bug, not a refinement.** `_scaled` tiling
a sorted array produces a sawtooth, so `searchsorted_sortqueries*N` was
handing the path a "haystack" that was not sorted at all - and that path's
gate samples the disorder of the QUERIES, not of the haystack, so it
dispatched happily on it. Verified after the repair: the haystack is sorted
through scaling and the queries are not, which is the input the cell was
always supposed to be.

**What it is worth, on identical code.** The pre-change tool and the
post-change tool run against the same source, same box, back to back:

```
unique_values_sort#uint64*30    old 0.94x    new 8.35x
```

Nine-fold, and entirely the instrument. Four of the seven cells still red
after section 11's floor raises were this and nothing else.

### The audit: do batches 14 and 15 still stand?

They acted on `*N` cells, so the question is not rhetorical. Structurally
they should be immune - batch 14's upward findings are batches of MATRICES,
whose cost does not depend on the value distribution, and its downward ones
(`np.inner` 0.38x, `np.histogram2d` 0.75x) use division, which slices and
never tiled. But that is reasoning, so it was measured, old tool against
new on the same code:

```
det_small_batch#3*100        old 4.89x   new 3.94x    same verdict
slogdet_small_batch#3*100    old skip    new skip     (cap came down in 14)
slogdet_small_batch#4*30     old skip    new skip
hist2d_uniform*30            old 1.57x   new 1.71x    same verdict
inner_tensordot*10           old 2.48x   new 2.60x    same verdict
```

**No decision from either batch flips.** Worth stating precisely because
the alternative was expensive: had the tiling flattered a threshold, every
floor those batches set would have needed re-deriving.

The numbers do MOVE where the verdicts do not - det 3x3 at 100x its floor
is 24% different - so any `*N` figure quoted in the batch 14 and 15 notes
should be read as approximate under the current instrument. The verdicts
are what those batches acted on, and the verdicts hold.

`tools/compare_sweep_logs.py` exists so the next instrument change is
landed the same way: whole sweep before, whole sweep after, diffed cell by
cell, with a cell that PASSED before and fails now called out loudest -
that is the direction where the old green was hiding a real loss.

## 15. A precondition numpy documents and never checks

Section 14 fixed the size axis so `searchsorted`'s scaled cells stop being
handed a sawtooth. That repaired the MEASUREMENT and left the real question
open: the sweep was only ever able to feed the path an unsorted haystack
because the path accepts one.

It does, and on the numpy this box runs it does not agree with stock about
it. numpy documents `a` as needing to be sorted and does not verify it, so
an unsorted haystack yields a meaningless answer rather than an error - but
meaningless is not arbitrary, it is a deterministic function of the array,
and **this path returned a different one: 43,407 of 50,000 positions
disagreed with stock** on a 50,000-element unsorted float64 haystack, and
it dispatched without hesitation.

**That measurement is version-conditional, and it was already known.** CI
said so before this section was first written: the divergence is a numpy <
2.5 effect, the differential battery already carried a strict xfail marking
exactly that boundary (17,122/20,000 on 2.4.5, 0/20,000 on 2.5.2), and the
module docstring already described the whole mechanism. A previous batch
measured this, understood it, and decided to keep dispatching. The claim as
first written here - a flat 86.8% divergence, no version, no prior art -
was wrong on both counts, and the container leg failed it within the hour.

So what follows is a REVERSAL of a considered decision, not the discovery
of an oversight.

The cause is the path's own mechanism. Sorting the queries is order-neutral
only if each query is answered independently, and numpy narrows the search
range as it walks a sorted query list - valid on a sorted haystack, wrong
on an unsorted one. **The optimisation the path exists for is exactly what
makes it diverge here.**

**Why reverse it.** The old call was not unreasonable: this is undefined
behaviour on numpy's side either way, and on the newest numpy there is no
divergence at all. Two things moved. The floor went to numpy>=2.3 in
section 7, which means the divergent versions - 2.3 and 2.4 - are
SUPPORTED versions rather than a legacy tail. And this project had already
ruled the opposite way on the identical shape: `isin_string_hash` refuses
lone-NUL strings so stock keeps answering for a class where stock is
quirky. Two opposite decisions about undefined behaviour is one too many,
and the cheap one to move is the one that costs 0.1-0.5%.

`_haystack_sorted` is a full O(n) non-decreasing check - not sampled,
because a probabilistic check that passes an unsorted array leaves the
divergence in place - and it runs last, after the 4096-point query-disorder
sample, so it is only paid by an input that would otherwise have
dispatched.

It is free at the scale it guards: **3 us against a 593 us call at
n=10,000, 250 us against 203 ms at n=1e6 - 0.1-0.5%**, which cannot reach
the 1.28-1.97x margin. Non-decreasing rather than strictly increasing,
since searchsorted is defined on haystacks with repeats. A NaN anywhere in
the haystack reads as unsorted and goes to stock, which is conservative on
purpose and costs a dispatch, never an answer.

The two tests that CI failed are now the two that carry the history. The
old xfail became a REFUSAL test that explains what it used to assert and
why it changed, kept rather than deleted because it is the exact shape that
used to dispatch. And the divergence itself is asserted in BOTH directions
against the numpy boundary - divergent below 2.5, identical at or above -
so if that boundary ever moves again the suite says which way it went,
instead of an xfail quietly flipping to a pass.

**The lesson is not about searchsorted.** A claim was written into these
notes and a commit message before the run that could check it had finished,
for the third time in this batch, and the same instrument that keeps
catching it caught it again. What made the difference each time was running
the thing on more than one numpy and more than one machine - which is what
sections 7 and 10 built the tooling for. Use it before writing the sentence,
not after.

### The same question, asked of every other path

One path having an unchecked precondition is a bug; the class is worth a
sweep. Of the 59 ops served, four have a documented precondition numpy does
not verify:

- **`searchsorted`** - `a` must be sorted. Was broken, now guarded.
- **`interp`** - `xp` must be increasing. Already safe, and not by luck:
  the uniformity guard requires every `diff(xp)` to match the first within
  1e-9 relative, which enforces `dx > 0` as a side effect.
- **`intersect1d`** - `assume_unique=True` takes the caller's word. The
  path refuses that keyword outright and stock answers.
- **`isin`** - `assume_unique=True` is ACCEPTED by both hash paths, so it
  was tested rather than assumed: with the promise deliberately false
  (duplicates in both operands), dispatched and stock agree exactly, in
  both the True and False cases. Membership is idempotent under
  duplication, so there is nothing for the promise to change. Recorded
  because a negative result nobody wrote down gets re-investigated.

## 16. Two conventions become checks, and a contradiction hunt

Three times in this batch a claim was written before the run that could
refute it, always in the same direction, and every time the thing that
caught it was measuring somewhere ELSE - another numpy, another machine.
"Run it first" is a habit, and habits are what those three slipped through,
so it goes in front of the claim instead.

**`tools/verify_evidence_cited.py`, wired into `ci.yml`.** One doctrine,
two halves: numpy's version belongs in the EVIDENCE and never in the
BEHAVIOUR.

- A speed claim that does not name its numpy cannot be shown to have
  rotted, which this batch watched happen three times (sections 2, 7, 10).
  Machine provenance was already the convention - most modules name a
  fingerprint - but numpy is the axis that moves under a threshold with
  nobody touching the repo.
- A path that BRANCHES on the numpy version has made its behaviour
  version-conditional, which section 2 ruled against. That one was
  VERIFIED rather than assumed, which is the whole point of this section:
  the only uses of `np.__version__` under `src/` are diagnostics, the
  calibration record and the demo banner. Reporting, not deciding.

**19 of 48 modules carry ratios with no recoverable numpy provenance**, so
the rule is a ratchet rather than a wall - the list may shrink and may
never grow. Those entries are deliberately NOT fixable by editing a
comment: nobody knows which numpy those numbers came from, and typing in a
plausible one would be inventing evidence. Each leaves by being
re-measured, which is a real batch of work and is now visible instead of
invisible.

Proven to bite rather than merely to pass: an unqualified claim and a
version branch were each introduced deliberately, and each exits 1.

### The contradiction hunt

Undefined-numpy-behaviour was handled two opposite ways until section 15
(refuse for `isin`'s NUL strings, accept for `searchsorted`'s unsorted
haystack). That is a class a sweep finds and a reader does not, so the rest
of the doctrine got swept.

**One more found.** `intersect_sorted` REFUSES `assume_unique=`; both isin
hash paths ACCEPT it. Same keyword, same family, opposite answers, and the
justification existed nowhere. It IS one rule: `assume_unique` is a promise
numpy does not verify, and for isin the promise cannot change the answer
because membership is idempotent under duplication, while for `intersect1d`
it can - stock with `assume_unique=True` skips its `unique()` calls
entirely and takes a different route, so a false promise yields a specific
wrong result a fast path would have to reproduce rather than merely be
correct about. Now written on all three sides and pinned by a test on each
isin path with the promise deliberately FALSE.

**The negatives are worth as much as the find**, because each is a place a
future sweep would otherwise re-investigate:

- All 68 registered paths declare a comparison mode and carry a kill
  switch. (`nanpercentile_masked` reads as missing one only to a regex
  looking for the docstring header; it declares its mode in the registry
  metadata, which is the source of truth.)
- The only sampled - probabilistic - gate in the package is
  `searchsorted`'s query-disorder estimate, and that is a PERFORMANCE gate.
  So the line drawn in section 15, never sample a correctness gate, is
  contradicted nowhere.
- `hist2d_uniform`'s "SAMPLES" is a data-size floor, not a sampled gate. A
  false positive worth naming so the next sweep does not re-flag it.

## 17. The re-derivation: 23 losses to 4, and nothing was hiding

The whole four-axis sweep, re-run on the idle Intel box after the size-axis
repair and the batch's gate changes, diffed against the pre-repair run cell
by cell with `tools/compare_sweep_logs.py`:

```
before  875 measured / 1156 skipped     23 below 1x
after   859 measured / 1174 skipped      4 below 1x
```

**Zero cells BROKEN.** That is the number that mattered and the reason the
comparer names that direction loudest: a cell that PASSED under tiling and
fails under honest sizing would mean the old green was hiding a real loss.
There were none. The instrument was pessimistic, never flattering.

**16 FIXED**, in two groups that say different things. Nine were the tiling
artifact and were never path losses at all - `unique_sort#int64*30` reads
9.37x where it read 0.89x, `unique_values_sort#uint64*30` 6.94x where it
read 0.80x. Seven were fixed by section 11's floor raises, which is the
floors doing their job: `intersect_sorted#int64%d` 0.68x -> 1.37x,
`#int32%d` 0.74x -> 1.95x.

**The 4 that remain are the residue section 11 argued for keeping**, and
they are the same four families named there - `intersect_sorted`'s 16-bit
rows at 0.92x and 0.94x, `unique_sort`/`unique_values_sort`'s uint64 rows at
0.94x and 0.95x, all of them `%d` low-cardinality cells. No `*N` losses
survive anywhere. The judgement call is unchanged and still the owner's to
overrule; what changed is that it is now the ONLY thing standing between
this sweep and green.

### The diff also caught my own regression, which is the point of it

18 cells stopped dispatching. Twelve are gates working: `intersect_sorted`'s
divided cells fall under its raised floor, `unique_sort/100` likewise, and
the four `isin_*>4`/`>16` cells are the TEST_FLOOR guard refusing the
corner it was built for (they read 0.05-0.46x before).

Six were mine. `relayout_blocked`'s size cells went from 1.78-2.68x to
no-dispatch because the resample built its array with
`np.ascontiguousarray`, forcing C order, and that path's fixture is a
TRANSPOSED, F-contiguous view its predicate requires. Tiling preserved the
layout by accident; an explicit copy does not unless asked. Fixed, verified
back at 1.49-1.78x with `f_contiguous` True.

Worth stating plainly: relayout_blocked is a path whose floors moved
earlier in this same batch, so the sweep would have been silently blind to
the size axis of a gate that had just been re-derived. **A cell that stops
dispatching reads exactly like a cell with nothing to say**, which is why
the comparer reports it separately from a ratio that merely moved.

## State at the end of the batch

2205 tests here / 2206 on the Intel box (two int64 searchsorted dispatch
tests became refusal tests), 68 registered paths' worth of rows across 155
row+dtype cells all >= 1.0x, plus the 693-cell size sweep and the shape
sweep. 69 registered paths, unchanged - the searchsorted path still ships,
with one fewer dtype.
