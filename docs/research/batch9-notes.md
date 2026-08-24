# Batch 9: bench-item execution - fused cholesky, eigvalsh split, einsum ellipsis (2026-08-24)

No sweep this batch: batch 9 executed the three numeric leads the
batch-8 bench left open, all refinements of shipped families (no new
OPP records; notes appended to OPP-000018/000047/000048/000049).

## 1. cholesky_small_batch: fused guard-in-run, chunked, capless (OPP-000047)

The observation that unlocked it: the shipped path computed the pivot
guard in `_applicable` and then recomputed the same arithmetic in
`_run` - the guard pivots ARE the factorization's intermediates
(p2 = a22 - l21², p3 = a33 - l31² - l32²). Fusing guard into the
factorization pass and running it in cache-sized chunks removes the
full-stack temporaries that caused the old 3x3 L2 cliff (the bistable
5000 cell that forced the 3000 cap).

Honest-candidate correction first (the batch-8 lesson applied): the
bench note's projected ~5x at batch ≥5000 was raw unguarded Crout
(69.7µs vs guarded ~350µs at 3x3/5000). The honest fused number is
1.6–1.9x there - still a clean win, a fraction of the projection.

Chunk size is measured, not guessed. Both machines show a narrow
chunk-4096 resonance around batch 10_000 for 3x3 only (1.11x idle /
1.23x dev, recovered by 16_384; 2x2 never dips), while chunk 1024 is
smooth there but ~25% slower ≥30_000. Policy: 2x2 always 4096; 3x3
uses 1024 below 30_000 (the smallest cell where 4096 is verified
clean on BOTH boxes), 4096 from there up.

BATCH9-CAL idle-box verdict (fp 9bbe7063c555, 0% load, dev box
re-probed at 1% load agreeing at every cell): 2x2 1.86–2.30x at
1000–1M (the old weakest admitted cell, 1.29x at 20_000, now 2.30x);
3x3 1.59–1.90x at 1000–10_000, 1.66–1.88x at 16_384–1M. **Both
windows: floor 1000, no cap.** Guard refusal now hands the whole call
to stock mid-run (StockRaised pattern), so in-window non-PD /
non-finite input gets stock's exact behavior, unbranded.

## 2. eigvalsh_3x3_trig: split-and-recombine (OPP-000048)

The shipped path bailed the WHOLE stack to stock when any cell had
1 - r² < DEGENERACY_MIN - a 10k-cell batch was punished to ~1.0x for
1% degenerate cells. Now the failing cells are gathered, served by
stock, scattered back; past DEGEN_FRAC_MAX = 0.25 the whole stack
goes to stock in one batched call (the gathered subset pays stock's
per-matrix LAPACK dispatch, so past ~a quarter one batched call is
cheaper).

Idle-box cells (n=10_000): 3.90x at 0%, 3.70x at 1%, 3.03x at 10%,
2.33x at 25%, 0.92x at 50% - the ceiling protects exactly that loss.
f32 degenerate cells take stock's own f32 route and survive the
f32→f64→f32 roundtrip exactly.

## 3. einsum ellipsis admission (OPP-000018 + OPP-000049)

Probe surprise: implicit-output label chains ('ij,jk,kl') were
ALREADY served - the batch-8 bench note was wrong that both implicit
and ellipsis forms were refused. Only ellipsis was.

Both einsum paths now admit ellipsis spellings when every operand
carries '...' with equal ellipsis shapes and ≥1 real ellipsis dim
(mid-ellipsis 'i...j' included; extents pre-checked so malformed
calls raise from stock). Refused, deliberately: unequal ellipsis
shapes (numpy broadcasts them - legal, unmeasured), '...' in only
some operands, explicit '->' outputs lacking '...' (numpy's
broadcast-dim output rules get subtle), zero ellipsis dims (that's
just the label spelling, already served).

Floors: the two-operand form reuses PROJECTED_FLOOR = 10_000
(idle-box floor cell 3.16x; 4.25x at B=64 n=32; 1.22x below floor).
The ellipsis CHAIN crosses later than the label spelling: 0.62x at
volume 20_736, 1.29x/1.49x across runs at 65_536 (a straddling cell,
bistable-cell rule), 2.20x at 76_832, 3.38x at 131_072, 162.76x at
67.1M. So it gates against its own CHAIN_ELLIPSIS_VOLUME_FLOOR =
76_832, the first cell clearing the bar in every run on both machines.

## Still on the bench (unchanged from batch 8)

Each needs an owner-level infrastructure call before further work:
np.fromfile/np.save/np.load I/O family (disk-bound evidence standard,
since dyno measures CPU), np.linalg.qr gufunc demand (numpy#7179,
C-feasibility research). The einsum planner-shapes item is now CLOSED
by the ellipsis admission; remaining refused einsum forms are refused
by measurement or by semantic risk, with the evidence recorded above.
