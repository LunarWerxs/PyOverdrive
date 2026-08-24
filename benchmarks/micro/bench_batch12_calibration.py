"""Batch-12 calibration battery: the singular-value family on 2x2/3x3
batches (OPP-000056) - pinv, norm(ord=2), svd(compute_uv=False).

Each block drives the SHIPPED route, guards included, so the numbers are
what a user gets and not what a bare kernel could do. Three separate
accuracy standards are checked, because the three paths make three
different promises (see the module docstring): pinv relative to its own
magnitude, singular values ABSOLUTE against ||A|| (the standard LAPACK
itself guarantees, and the one the shipped eigvalsh paths use), the
2-norm relative.

The conditioning sweep is the point of this battery, not a footnote: it
is what located both bands. Forming the gram squares the condition
number, so the singular-value error grows like eps*cond^2, and d=3
carries a worse constant than d=2 because its trigonometric solution
amplifies error as the gram's eigenvalues cluster. The cells below
straddle each band so a future NumPy or CPU that moves the crossing
shows up here rather than in a user's results.

Result JSON: benchmarks/results/BATCH12-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_batch12_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from lab.dyno import BenchSuite

from pyoverdrive.fastpaths.svd_small_batch import (
    BATCH_MIN,
    PINV_SIGMA_RATIO_MIN,
    SVDVALS_SIGMA_RATIO_MIN,
    _run_norm2,
    _run_pinv,
    _run_svdvals,
)

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite(
    "BATCH12-CAL",
    "pinv / norm(ord=2) / svd(compute_uv=False) on 2x2-3x3 batches via the gram closed form",
)
rng = np.random.default_rng(20260825)


def rel_close(rtol):
    """Relative to the result's own magnitude (pinv, norm2)."""

    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        if c.dtype != b.dtype or c.shape != b.shape:
            return False
        scale = max(float(np.abs(b).max()), 1e-300)
        return bool(np.all(np.abs(c - b) <= rtol * scale))

    return _chk


def sv_close(rtol):
    """Absolute error against ||A|| = the largest singular value."""

    def _chk(c, b):
        c = np.asarray(c)
        b = np.asarray(b)
        if c.dtype != b.dtype or c.shape != b.shape:
            return False
        scale = np.maximum(b[..., :1], 1e-300)
        return bool(np.all(np.abs(c - b) <= rtol * scale))

    return _chk


def well_conditioned(n, d, seed):
    return np.ascontiguousarray(np.random.default_rng(seed).standard_normal((n, d, d)))


def with_condition(n, d, cond, seed):
    """An SVD-constructed stack with a known condition number."""
    local = np.random.default_rng(seed)
    u, _ = np.linalg.qr(local.standard_normal((n, d, d)))
    v, _ = np.linalg.qr(local.standard_normal((n, d, d)))
    s = np.geomspace(1.0, 1.0 / cond, d)
    return np.ascontiguousarray(u @ (s[None, :, None] * np.swapaxes(v, -1, -2)))


# --- 1. margin vs batch size, all three paths ------------------------------

NS = [BATCH_MIN, 10_000] if SMOKE else [BATCH_MIN, 300, 1_000, 10_000, 100_000]

for d in (2, 3):
    for n in NS:
        a = well_conditioned(n, d, seed=d * 1000 + n)
        suite.measure(
            case=f"pinv_{d}x{d}_n{n}",
            params={"op": "pinv", "d": d, "n": n, "batch_min": BATCH_MIN,
                    "sigma_ratio_min": PINV_SIGMA_RATIO_MIN},
            baseline=("numpy.linalg.pinv", lambda a=a: np.linalg.pinv(a)),
            candidates={"adjugate_banded": lambda a=a: _run_pinv(a)},
            check=rel_close(1e-9),
            samples=SAMPLES,
        )
        suite.measure(
            case=f"norm2_{d}x{d}_n{n}",
            params={"op": "norm2", "d": d, "n": n, "batch_min": BATCH_MIN},
            baseline=(
                "numpy.linalg.norm",
                lambda a=a: np.linalg.norm(a, ord=2, axis=(-2, -1)),
            ),
            candidates={"gram_sigma_max": lambda a=a: _run_norm2(a)},
            check=rel_close(1e-12),
            samples=SAMPLES,
        )
        suite.measure(
            case=f"svdvals_{d}x{d}_n{n}",
            params={"op": "svdvals", "d": d, "n": n,
                    "sigma_ratio_min": SVDVALS_SIGMA_RATIO_MIN[d]},
            baseline=("numpy.linalg.svd", lambda a=a: np.linalg.svd(a, compute_uv=False)),
            candidates={"gram_closed_form": lambda a=a: _run_svdvals(a)},
            check=sv_close(1e-9),
            samples=SAMPLES,
        )

# --- 2. the conditioning sweep that located the bands ----------------------

if not SMOKE:
    for d in (2, 3):
        for e in (2, 3, 4, 6, 8, 12):
            a = with_condition(4_000, d, 10.0**e, seed=e * 31 + d)
            suite.measure(
                case=f"svdvals_{d}x{d}_cond1e{e}",
                params={"op": "svdvals", "d": d, "n": 4_000, "cond": 10.0**e,
                        "sigma_ratio_min": SVDVALS_SIGMA_RATIO_MIN[d],
                        "band_cond": 1.0 / SVDVALS_SIGMA_RATIO_MIN[d]},
                baseline=("numpy.linalg.svd", lambda a=a: np.linalg.svd(a, compute_uv=False)),
                candidates={"gram_closed_form": lambda a=a: _run_svdvals(a)},
                check=sv_close(1e-9),
                samples=SAMPLES,
            )
            suite.measure(
                case=f"pinv_{d}x{d}_cond1e{e}",
                params={"op": "pinv", "d": d, "n": 4_000, "cond": 10.0**e,
                        "sigma_ratio_min": PINV_SIGMA_RATIO_MIN,
                        "band_cond": 1.0 / PINV_SIGMA_RATIO_MIN},
                baseline=("numpy.linalg.pinv", lambda a=a: np.linalg.pinv(a)),
                candidates={"adjugate_banded": lambda a=a: _run_pinv(a)},
                check=rel_close(1e-9),
                samples=SAMPLES,
            )

    # a stack that is MOSTLY band-trippers: the whole-call-to-stock arm
    for d in (2, 3):
        a = with_condition(4_000, d, 1e14, seed=d + 99)
        suite.measure(
            case=f"pinv_{d}x{d}_mostly_ill",
            params={"op": "pinv", "d": d, "n": 4_000, "cond": 1e14,
                    "expect": "whole call to stock (past BAD_FRAC_MAX)"},
            baseline=("numpy.linalg.pinv", lambda a=a: np.linalg.pinv(a)),
            candidates={"adjugate_banded": lambda a=a: _run_pinv(a)},
            check=rel_close(1e-9),
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
