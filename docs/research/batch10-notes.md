# Batch 10: qr_small_batch - the bench's last open lead executed (2026-08-24)

The batch-8 bench left np.linalg.qr as "C-level feasibility research";
batch 10 executed it. The research answer: values-only R DOES have a
closed form once the sign convention is pinned, and so does Q - the
Householder factorization for d=2 (one reflector) and d=3 (two
reflectors) unrolls into vectorized arithmetic in LAPACK's own dlarfg
convention, reproducing stock's exact per-column sign choices. One new
family: qr_small_batch (OPP-000053, numpy#7179 provenance), 65 paths
total.

## The sign contract

qr's factors are unique only up to per-column signs, so serving qr
means reproducing LAPACK's choices, not producing "a valid QR". dlarfg
pins them: beta = -sign(alpha)*hypot(alpha, xnorm) with a nonzero
below-diagonal part; tau = 0, beta = alpha (identity reflector) when
it is exactly zero. Unrolling exactly those formulas matched stock to
|dQ| <= 8.7e-15 / |dR| <= 9.3e-16 across random, negative-lead,
triangular, zero-column, and all-zero inputs on the first probe.

## The second-reflector determinism band (the real find)

The d=3 second reflector's inputs are COMPUTED (not raw bits), carrying
~eps*scale noise, and dlarfg is discontinuous at b32 == 0 and in
sign(b22). Rank-deficient or repeated-column input puts the trailing
block at noise grade, where the two routes return sign-flipped R rows
and O(1)-different Q columns - both valid factorizations, but the
contract is agreement with STOCK. The reviewer agent caught this on
rank-1 input; the QR_RTOL = 1e-6 band detects it mid-run.

The first fix (whole-stack bail) was REJECTED by the battery within
the hour: random Gaussian stacks trip the band with probability ~1e-5
per matrix, so a million-matrix stack almost surely contains one and
every large-n cell went CORRECTNESS-FAIL (the bail left np.empty
garbage in the battery's mirror) - and even done correctly it would
have collapsed the large-n regime to stock. The shipped design is the
eigvalsh batch-9 split-and-recombine: trippers gathered, served by
stock, scattered back; whole stack to stock past QR_BAD_FRAC_MAX =
0.25. The d=2 path needs no band at all - its single reflector reads
raw input bits, which both routes see identically.

## Calibration (BATCH10-CAL, both fingerprints)

Idle box (fp 9bbe7063c555, 0-1% load): reduced-mode 2x2 4.08x at
batch 300 rising to 7.7-11x, 3x3 2.21x at 300, 3.0-4.2x at 1000-1M;
R-only 1.67-8.2x. Floors: 300 for both shapes - 3x3 n=100 loses
outright (0.83x), and 2x2 n=100 clears on the idle box (1.46x) but
never on the dev box in mode='r' (1.04-1.29x across three runs), so
the two-machine law holds it at 300. CHUNK = 4096 uniform: a few
mid-size reduced cells prefer 1024 by ~15-20% (the batch-10_000
resonance again), but every such cell still reads 3-6x, so the
cholesky-style adaptive chunk is not warranted.

## Bench state after batch 10

The np.fromfile/np.save/np.load I/O family remains the only bench item,
still blocked on the owner call for a disk-bound evidence standard
(dyno measures CPU). Everything else the sweeps ever surfaced is now
shipped, extended, or declined with committed measurement.
