"""OPP-000000: Gearbox dispatch overhead (Phase 0/3 gate metric).

Measures the cost of routing a call through the Gearbox wrapper against
calling stock NumPy directly, for both the fast-path-taken and the
fallback-taken routes. Dispatch overhead must not erase gains (spec Phase 3
gate); this file is the standing measurement of that overhead.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

import pyoverdrive

SMOKE = "--smoke" in sys.argv

stock_positive = np.positive
pyoverdrive.enable(["numpy.positive"])
pyoverdrive.enable_path("noop_positive")
wrapped_positive = np.positive
assert getattr(wrapped_positive, "__pyoverdrive__", False)

suite = BenchSuite("OPP-000000", "Gearbox dispatch overhead via no-op fast path")

sizes = [8] if SMOKE else [8, 1024, 1_000_000]
rng = np.random.default_rng(0)
for n in sizes:
    x = rng.standard_normal(n)
    out = np.empty_like(x)
    suite.measure(
        case=f"positive_float64_n{n}",
        params={"dtype": "float64", "n": n},
        baseline=("stock_positive", lambda x=x: stock_positive(x)),
        candidates={
            # predicate accepts -> runs the no-op fast path
            "gearbox_fastpath": lambda x=x: wrapped_positive(x),
            # kwargs present -> predicate rejects -> stock fallback route
            "gearbox_fallback": lambda x=x, out=out: wrapped_positive(x, out=out),
        },
        check=np.array_equal,
        samples=3 if SMOKE else 15,
    )

pyoverdrive.disable()
if not SMOKE:
    suite.save()
