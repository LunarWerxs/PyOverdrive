"""OPP-000055: numpy.vectorize wrapping a ufunc vs calling it directly.

NumPy's own documentation says vectorize "is provided primarily for
convenience, not for performance. The implementation is essentially a
for loop." When the wrapped callable is one of NumPy's OWN ufuncs, that
loop calls a fully vectorized C routine once per element.

Bit-identical comparison: the two routes run the same ufunc over the
same float64 values. (The served set is exactly the unary float64
ufuncs verified bit-identical between NumPy's scalar loop and its array
loop on both benchmark machines; the differential suite re-proves that
wherever it runs.)

House rules: never imports pyoverdrive.
Result JSON: benchmarks/results/OPP-000055/.
Run: .venv/Scripts/python benchmarks/historical/opp_000055_vectorize_ufunc.py [--smoke]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SMOKE = "--smoke" in sys.argv
SAMPLES = 3 if SMOKE else 7

suite = BenchSuite("OPP-000055", "vectorize(ufunc) object loop vs direct call")
rng = np.random.default_rng(55)


def exact(c, b):
    c = np.asarray(c)
    b = np.asarray(b)
    return (
        c.shape == b.shape
        and c.dtype == b.dtype
        and bool(np.array_equal(c, b, equal_nan=True))
    )


NS = [10_000, 1_000_000] if SMOKE else [100, 1_000, 10_000, 100_000, 1_000_000]
FUNCS = ("sin", "exp", "sqrt") if SMOKE else ("sin", "exp", "sqrt", "log", "tanh")

for n in NS:
    x = np.abs(rng.standard_normal(n)) + 1e-6
    for name in FUNCS:
        uf = getattr(np, name)
        v = np.vectorize(uf)
        suite.measure(
            case=f"{name}_n{n}",
            params={"ufunc": name, "n": n},
            baseline=("numpy.vectorize.__call__", lambda v=v, x=x: v(x)),
            candidates={"ufunc_direct": lambda uf=uf, x=x: uf(x)},
            check=exact,
            samples=SAMPLES,
        )

if not SMOKE:
    suite.save()
