"""OPP-000030: batched 2x2 symmetric eigendecomposition, closed form vs LAPACK.

numpy/numpy#22158 (javierttgg, 2022, numpy 1.21.5): eigh on a batch of
1000 stacked 2x2 symmetric matrices is ~1.43x SLOWER than eig on the
same input (derived from the reporter's "gives ~ 0.7" ratio comment),
because both route each tiny matrix through a per-matrix LAPACK call
whose setup overhead dominates (seberg's unconfirmed hypothesis, the
only analysis in the thread). The thread never proposes a candidate;
the closed-form route below is from first principles.

What this reproducer measures:

  1. numpy.linalg.eigvalsh (values only) vs a closed-form quadratic:
     for a real symmetric [[a, b], [b, d]] read from the LOWER triangle
     (eigvalsh's default UPLO='L', which reads a[..., 1, 0] and ignores
     a[..., 0, 1]), eigenvalues are (a+d)/2 -/+ sqrt(((a-d)/2)^2 + b^2),
     returned ascending exactly as stock does. eigvalsh output is
     mathematically UNIQUE (no sign/basis freedom outside exact
     degeneracy), so the check is numeric allclose scaled to the
     matrix magnitudes.
  2. numpy.linalg.eigh (values + vectors) vs a closed-form route, for
     HEADROOM DOCUMENTATION ONLY: eigenvector SIGN is convention, not
     mathematics, so the check here is a CONTRACT check (eigenvalues
     match stock numerically; vectors satisfy A v = lambda v and
     orthonormality to a tolerance tied to stock's own residual), not
     elementwise equality. A fast path on eigh cannot ship without a
     comparison-contract mode (same blocker as np.partition, see
     OPP-000022 notes); eigvalsh has no such blocker.
  3. Batch-size sweep 100 .. 1_000_000 to locate the floor, plus a
     single-matrix case (the anti-regime: the thread itself shows eigh
     WINNING 3x at batch 1, so tiny batches must stay on stock).
  4. An ill-conditioned witness batch (eigenvalue ratios ~1e12) at one
     size, correctness-checked with the same scaled tolerance: the
     closed-form quadratic loses relative accuracy on the SMALL
     eigenvalue via cancellation exactly where LAPACK is careful. If
     the check fails there, that is a FINDING that narrows the regime
     (well-separated moderate-condition matrices), not a script bug.

House rules: this script never imports pyoverdrive. The closed-form
candidates call only sqrt/stack/arithmetic, no linalg, so a patched
dispatch could not recurse.

Result JSON: benchmarks/results/OPP-000030/.
Run: .venv/Scripts/python benchmarks/historical/opp_000030_eigh_small_batch.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 22158
SMOKE = "--smoke" in sys.argv


def eigvalsh2x2(a):
    """Closed-form ascending eigenvalues, lower triangle (UPLO='L')."""
    a00 = a[..., 0, 0]
    a10 = a[..., 1, 0]
    a11 = a[..., 1, 1]
    mid = 0.5 * (a00 + a11)
    disc = np.sqrt((0.5 * (a00 - a11)) ** 2 + a10 * a10)
    return np.stack([mid - disc, mid + disc], axis=-1)


def eigh2x2(a):
    """Closed-form (values, vectors), lower triangle. For each eigenvalue
    w_k both null-space forms [b, w_k - a00] and [w_k - a11, b] are exact
    (their defect is b^2 - (w-a00)(w-a11) = 0 by the characteristic
    equation); pick per-k whichever has the larger norm so the chosen one
    never degenerates except for a truly diagonal matrix, which gets the
    identity basis. Vector sign is OUR convention, so only the contract
    check below is meaningful - eigh cannot ship on elementwise equality."""
    w = eigvalsh2x2(a)
    a00 = a[..., 0, 0, None]
    a10 = a[..., 1, 0, None]
    a11 = a[..., 1, 1, None]
    # branch 1: [b, w - a00]; branch 2: [w - a11, b]
    n1 = a10 * a10 + (w - a00) ** 2
    n2 = (w - a11) ** 2 + a10 * a10
    take1 = n1 >= n2
    vx = np.where(take1, a10, w - a11)
    vy = np.where(take1, w - a00, a10)
    # truly diagonal (both branches vanish for the diagonal's own entry):
    # eigenbasis is the identity, ordered by which diagonal entry is lo
    degen = (n1 == 0) & (n2 == 0)
    lo_first = a00 <= a11
    ex = np.where(lo_first, np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    ey = np.where(lo_first, np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    vx = np.where(degen, ex, vx)
    vy = np.where(degen, ey, vy)
    norm = np.sqrt(vx * vx + vy * vy)
    vx = vx / norm
    vy = vy / norm
    v = np.stack([vx, vy], axis=-2)  # v[..., :, k] is the k-th eigenvector
    return w, v


def sym_batch(rng, n, dtype=np.float64):
    a = rng.uniform(-1.0, 1.0, size=(n, 2, 2)).astype(dtype)
    return np.ascontiguousarray(a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(2, dtype=dtype))


def values_close(cand, base):
    if cand.dtype != base.dtype or cand.shape != base.shape:
        return False
    scale = np.maximum(1e-30, np.abs(base).max(axis=-1, keepdims=True))
    return bool((np.abs(cand - base) <= 1e-9 * scale).all())


def contract_check(cand, base):
    """eigh contract: eigenvalues match stock numerically; candidate's
    vectors reconstruct A as well as stock's do (within 10x), checked via
    residual norms. Elementwise vector equality is NOT required (sign)."""
    wc, vc = cand
    wb, vb = base
    if not values_close(wc, wb):
        return False
    # A is recoverable from stock: A = V diag(w) V^T
    a = vb @ (wb[..., None] * np.swapaxes(vb, -1, -2))
    res_c = np.abs(a @ vc - wc[..., None, :] * vc).max()
    res_b = np.abs(a @ vb - wb[..., None, :] * vb).max()
    ortho = np.abs(np.swapaxes(vc, -1, -2) @ vc - np.eye(2)).max()
    return bool(res_c <= 10.0 * max(res_b, 1e-14) and ortho < 1e-12)


suite = BenchSuite("OPP-000030", "batched 2x2 eigh/eigvalsh: closed form vs LAPACK loop")
rng = np.random.default_rng(SEED)

BATCHES = [1_000] if SMOKE else [1, 100, 1_000, 10_000, 100_000, 1_000_000]
SAMPLES = 3 if SMOKE else 9


def samples_for(n):
    return SAMPLES if n <= 100_000 else max(5, SAMPLES - 2)


for n in BATCHES:
    a = sym_batch(rng, n)
    suite.measure(
        case=f"eigvalsh_2x2_batch{n}",
        params={"batch": n, "d": 2, "op": "eigvalsh"},
        baseline=("numpy.linalg.eigvalsh", lambda a=a: np.linalg.eigvalsh(a)),
        candidates={"closed_form_values": lambda a=a: eigvalsh2x2(a)},
        check=values_close,
        samples=samples_for(n),
    )

for n in ([1_000] if SMOKE else [1_000, 100_000]):
    a = sym_batch(rng, n)
    suite.measure(
        case=f"eigh_2x2_batch{n}_contract",
        params={"batch": n, "d": 2, "op": "eigh", "check": "contract-not-elementwise"},
        baseline=("numpy.linalg.eigh", lambda a=a: tuple(np.linalg.eigh(a))),
        candidates={"closed_form_full": lambda a=a: eigh2x2(a)},
        check=contract_check,
        samples=samples_for(n),
    )

if not SMOKE:
    # ill-conditioned witness: eigenvalue ratio ~1e12; a check failure here
    # is a regime finding (see docstring), recorded either way
    n = 10_000
    q = rng.uniform(-1.0, 1.0, size=(n, 2, 2))
    qq, _ = np.linalg.qr(q)
    w_big = np.stack([np.full(n, 1e-12), np.full(n, 1.0)], axis=-1)
    a_ill = np.ascontiguousarray(qq @ (w_big[..., None] * np.swapaxes(qq, -1, -2)))
    a_ill = 0.5 * (a_ill + np.swapaxes(a_ill, -1, -2))
    suite.measure(
        case=f"eigvalsh_2x2_batch{n}_illconditioned",
        params={"batch": n, "d": 2, "op": "eigvalsh", "condition": 1e12},
        baseline=("numpy.linalg.eigvalsh", lambda a=a_ill: np.linalg.eigvalsh(a)),
        candidates={"closed_form_values": lambda a=a_ill: eigvalsh2x2(a)},
        check=values_close,
        samples=samples_for(n),
    )
    suite.save()
