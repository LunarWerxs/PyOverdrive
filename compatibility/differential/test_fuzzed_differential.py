"""Differential tests driven by the seeded fuzz specs, plus corpus replay.

Contract: for every path in fuzz_specs.SPECS, a fixed-seed sample of its
declared input space (strided, reversed and transposed views, ranks 0-3,
empty arrays, NaN/inf/-0.0, sizes straddling the path's floor or cap)
gives, through the patched public name, exactly what stock NumPy gives:
same type, dtype, shape and values under the path's comparison
discipline, or the same exception type. Each spec must also actually
reach its fast path at least once, so a recalibrated floor cannot quietly
turn the sample into a stock-vs-stock comparison.

Every file under corpus/<path>/ is a minimized mismatch that fuzz.py once
found; it must keep matching stock forever.
"""

from __future__ import annotations

import pytest

from fuzz_specs import SPECS
from fuzzing import activated, check_case, corpus_files, describe, load_case, run_spec

SEED = 20260925
ITERATIONS = 40


@pytest.mark.parametrize("name", sorted(SPECS))
def test_seeded_fuzz_matches_stock(name):
    spec = SPECS[name]
    with activated(spec):
        report = run_spec(spec, ITERATIONS, SEED, shrink_budget=150)
    assert not report.mismatches, "\n".join(
        f"{why}: {describe(case)}\n  keep it: python compatibility/differential/fuzz.py "
        f"--path {name} --seed {SEED} --iterations {ITERATIONS}"
        for case, why in report.mismatches
    )
    assert report.dispatched > 0, (
        f"{name}: none of {report.samples} seeded samples reached the fast path; "
        "widen its spec in fuzz_specs.py around the current predicate"
    )


def test_corpus_cases_still_match_stock():
    failures = []
    for file in corpus_files():
        meta, case = load_case(file)
        spec = SPECS.get(meta["path"])
        if spec is None:
            failures.append(f"{file.name}: no fuzz spec named {meta['path']!r} (path withdrawn?)")
            continue
        with activated(spec):
            outcome = check_case(spec, case)
        if outcome.mismatch:
            failures.append(f"{meta['path']}/{file.name}: {outcome.mismatch}; {describe(case)}")
    assert not failures, "\n".join(failures)
