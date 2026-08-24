"""nanpercentile_masked calibration: the shipped nanquantile route at q/100
vs stock np.nanpercentile.

OPP-000013 shipped nanquantile_masked (22-229x in its many-slice regime);
numpy implements nanpercentile as nanquantile at q/100, so the sibling
path reuses the parent's run verbatim after the same true_divide(q, 100)
scaling stock performs. House law: a path ships only on ITS OWN measured
cells, so this battery times the actual shipped run (the module's _run,
guards excluded, exactly like the parent's FASTNANQ-CAL protocol) against
stock np.nanpercentile at a subset of the FASTNANQ-CAL grid, plus the
known anti-regime rows the admission rule refuses.

Result JSON: benchmarks/results/NANPERCENTILE-CAL/.
Run: .venv/Scripts/python benchmarks/micro/bench_nanpercentile_calibration.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from lab.dyno import BenchSuite
from pyoverdrive.fastpaths import nanpercentile_masked

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("NANPERCENTILE-CAL", "shipped masked route at q/100 vs stock nanpercentile")
rng = np.random.default_rng(16575)


def close(c, b):
    c = np.asarray(c)
    b = np.asarray(b)
    return c.shape == b.shape and c.dtype == b.dtype and bool(
        np.allclose(c, b, rtol=1e-12, atol=0.0, equal_nan=True)
    )


# (shape, axis, nan_frac, q) - FASTNANQ-CAL subset + the refused anti-regime
CASES = [
    ((10, 30), 0, 0.1, 80.0),
    ((27, 100), 0, 0.1, 80.0),
    ((27, 100), 1, 0.1, 33.0),
    ((5, 500), 0, 0.1, 50.0),
    ((50, 100, 100), -1, 0.1, 90.0),
    ((500, 500), 1, 0.1, 25.0),
    ((200, 1000), 1, 0.0, 75.0),  # no-NaN branch: scan + vectorized quantile
    ((100_000, 5), 0, 0.01, 40.0),  # anti-regime the admission rule refuses
]
if SMOKE:
    CASES = CASES[:1]

for shape, axis, nan_frac, q in CASES:
    a = rng.standard_normal(shape)
    if nan_frac:
        a[rng.random(shape) < nan_frac] = np.nan
    admitted = nanpercentile_masked._applicable((a, q), {"axis": axis})
    suite.measure(
        case=f"s{'x'.join(map(str, shape))}_ax{axis}_nan{int(nan_frac * 100)}_q{q:g}"
        + ("" if admitted else "_REFUSED"),
        params={
            "shape": list(shape),
            "axis": axis,
            "nan_frac": nan_frac,
            "q": q,
            "admitted": bool(admitted),
        },
        baseline=(
            "numpy.nanpercentile",
            lambda a=a, q=q, ax=axis: np.nanpercentile(a, q, axis=ax),
        ),
        candidates={
            "masked_route": lambda a=a, q=q, ax=axis: nanpercentile_masked._run(
                a, q, axis=ax
            )
        },
        check=close,
        samples=SAMPLES,
    )

if not SMOKE:
    suite.save()
