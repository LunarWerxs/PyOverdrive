# Batch 13: np.pad, and the benchmark that lied by 400x (2026-08-25)

Batch 13 shipped one path (OPP-000057) and is mostly a lesson about
measurement. 45 always-on families / 71 registered paths.

## Shipped: 1-D constant-mode np.pad

Same vein as roll_concat (OPP-000032): np.pad is a Python driver whose
fixed preamble is the entire cost on a small array. Stock takes ~6.4 us at
n=64 whatever the dtype or size. For 1-D constant mode the actual work is
one allocation plus one assignment.

Consumed margins through the shipped route, idle box, output lengths 14 to
16006: no constant 4.57x down to 1.45x, constant given 2.56x down to 1.61x.
Every measured dtype lands in the same band at n=64 (3.6-4.7x), which is
what a fixed-machinery saving should look like.

## The bare number was wrong by more than two orders of magnitude

The no-constant route allocates with np.zeros, which is calloc, so the
pages it returns have not been faulted in yet. Timing only the pad call
therefore banks work the caller has not paid for:

    n=8, pad=1_000_000     bare 342x      consumed 0.86x
    n=8, pad=120_000       bare  58x      consumed 0.86x
    n=64, pad=3            bare 14.8x     consumed 4.46x

The first row is a REGRESSION that would have shipped as the largest
speedup in the project. Nothing about it looks suspicious from the bare
timing: it is monotone, it reproduces, it is stable across runs, and it
gets *better* as the pad grows, which reads like a scaling win rather than
an artifact.

Two consequences, both now permanent. Every margin in the record is the
CONSUMED one - the padded array is summed before the clock stops. And
OUTPUT_CAP is set on the length of the RESULT rather than of the input,
because a tiny array with an enormous pad is exactly the shape that looks
best bare and performs worst used; a cap on input size alone would have
admitted every one of those cells. BATCH13-CAL keeps the refused cells,
measured both ways, so the crossing stays evidence rather than a claim.

## The shortlist figure had nothing behind it

The batch-11 panel recorded "13.9x dev / 15.2x idle at n=64" for this
opportunity. There was no results cell and no bench script anywhere in the
repo - it was an exploratory number that had been written into a document
and then read back as though it were evidence.

Re-measuring reproduced 13.9x exactly, which is the uncomfortable part: the
figure was not wrong, it was just measuring a bare timing of a raw route,
which is neither what ships nor what a user experiences. The honest figure
for the shipped path is 4.46x. Any shortlist number without a committed
fingerprinted cell behind it is a lead, not evidence, and the remaining
entries in that shortlist should be treated the same way.

## The guard is part of the cost

The raw route measured 7.7x consumed at n=64. The shipped route measured
3.5x. The gap is the predicate: normalization runs twice per call, once for
`applicable` and once inside `run`, and a single np.asarray costs about as
much as the pad it is preparing. Fast-pathing the two spellings that
dominate real traffic - a plain int and a 2-tuple of ints, no numpy at all -
took it back to 4.46x with nothing about the accepted or refused set moving.

A guard that is cheap to *describe* is not automatically cheap to *run*,
and on a path whose entire premise is "stock's preamble is the cost", a
preamble of one's own is the obvious way to lose.

## Three traps np.array_equal cannot see

The differential suite for this path compares raw bytes, which is unusual
here and load-bearing:

1. `constant_values=-0.0` fills with NEGATIVE zero; np.zeros fills with
   positive zero; np.array_equal calls those equal. So the np.zeros route
   is gated on constant_values being ABSENT, not on it being zero.
2. The constant must become a NUMPY SCALAR, exactly as stock's `_as_pairs`
   produces. A 0-d array agrees with stock on wrapping -1 into uint8 and
   DISAGREES on NaN into an integer array, where it silently writes INT_MIN
   under a RuntimeWarning instead of raising ValueError. An earlier build
   used np.asarray and turned a stock exception into a wrong answer.
   Hypothesis found it; no hand-written case would have.
3. A zero-width pad on both sides writes the constant only into empty
   slices, so the cast never happens and an invalid constant quietly
   succeeds. That degenerate call is refused.

And one that only a value comparison catches: a string array is padded by
stock with the STRING `"0"`, not with the empty string np.zeros produces.
Shapes and dtypes agree. That is why the dtype allowlist is an allowlist
rather than a kind check.

Worth noting how much of this batch was found by the comparators being
wrong rather than the code. Five separate times a comparison reported a
difference that was not there, or hid one that was: NaN against NaN in a
list compare (twice), signed zero under array_equal, object arrays under
tobytes (pointers, not values), and a tolerance that went NaN. The code was
right in four of those five and wrong in the fifth in a way only the strict
comparator revealed.

## Bench after batch 13

Still queued from the batch-11 shortlist, and now all of them carry the
caveat above: det/slogdet on 4x4 batches (1.6-3.8x), genfromtxt routed to
loadtxt for clean simple calls (3.8x, bit-equal), and the np.ma fused-mask
arithmetic family. The I/O family still needs an owner call on a disk-bound
evidence standard.
