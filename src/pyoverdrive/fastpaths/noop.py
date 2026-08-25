"""The no-op fast path: Phase 0 gate artifact.

Registers a passthrough implementation of ``numpy.positive`` whose candidate
is literally stock NumPy. It exists so the whole pipeline (register ->
dispatch -> benchmark -> differential-test -> package) can be exercised end to
end before any real optimization ships, and so Gearbox dispatch overhead is a
measured number (benchmarks/historical/opp_000000_noop_overhead.py).

Disabled by default outside tests and benchmarks: it accelerates nothing.
Enable with ``pyoverdrive.enable_path("noop_positive")``.
"""

from __future__ import annotations

import numpy as np

from ..dispatcher.gearbox import GEARBOX, FastPath


def _applicable(args: tuple, kwargs: dict) -> bool:
    return (
        len(args) == 1
        and not kwargs
        and isinstance(args[0], np.ndarray)
    )


def _run(x):
    # A fast path must NEVER call the operation it accelerates by its public
    # name: while patched that name is the Gearbox wrapper and the call
    # recurses. Always go through stock_fn.
    return GEARBOX.stock_fn("numpy.positive")(x)


def register(gearbox) -> None:
    gearbox.register(
        FastPath(
            name="noop_positive",
            op="numpy.positive",
            applicable=_applicable,
            run=_run,
            provenance={
                "opportunity": "OPP-000000",
                "note": "Phase 0 gate artifact; measures dispatch overhead.",
                # trivially true - the candidate IS stock - but declared
                # rather than left blank, because this was the one
                # registered path with no comparison mode recorded, which
                # contradicted the claim that every path records one
                "comparison_mode": "bit-identical",
            },
            enabled=False,
        )
    )
