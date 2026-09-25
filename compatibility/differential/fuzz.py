"""Fuzz mode for the differential suites: hunt for patched-vs-stock
divergences and keep every one found as a permanent corpus case.

WHY: the plain test run only replays a small seeded sample (and the
corpus). A long run with fresh seeds explores far more of each path's
input space; any mismatch it finds is shrunk to a small input and written
to compatibility/differential/corpus/<path>/<hash>.npz, which
test_fuzzed_differential.py replays on every run from then on - the bug
stays caught after the fuzzer has moved on.

    python compatibility/differential/fuzz.py                  # every spec
    python compatibility/differential/fuzz.py --path nanmean_scan --iterations 5000 --seed 7
    python compatibility/differential/fuzz.py --list
    python compatibility/differential/fuzz.py --replay         # corpus only

Exit status 1 when any mismatch was found (or replayed), else 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Always fuzz this checkout, even if another checkout was installed editable
# (the same rule pyproject.toml gives pytest).
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from fuzz_specs import SPECS  # noqa: E402
from fuzzing import (  # noqa: E402
    CORPUS_DIR, activated, check_case, corpus_files, describe, load_case, run_spec, save_case,
)


def _replay() -> int:
    failures = 0
    for file in corpus_files():
        meta, case = load_case(file)
        spec = SPECS.get(meta["path"])
        if spec is None:
            print(f"ORPHAN {file.relative_to(CORPUS_DIR)}: no spec named {meta['path']!r}")
            failures += 1
            continue
        with activated(spec):
            outcome = check_case(spec, case)
        status = "FAIL" if outcome.mismatch else "ok  "
        failures += bool(outcome.mismatch)
        print(f"{status} {file.relative_to(CORPUS_DIR)}  {outcome.mismatch or ''}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", action="append", choices=sorted(SPECS),
                    help="fast path to fuzz (repeatable; default: every spec)")
    ap.add_argument("--iterations", type=int, default=500, help="samples per path")
    ap.add_argument("--seed", type=int, default=0, help="first sample seed")
    ap.add_argument("--no-save", action="store_true", help="report mismatches without writing corpus files")
    ap.add_argument("--list", action="store_true", help="list the fuzz specs and exit")
    ap.add_argument("--replay", action="store_true", help="only replay the committed corpus")
    args = ap.parse_args(argv)

    if args.list:
        for name, spec in sorted(SPECS.items()):
            print(f"{name:20s} {spec.op}")
        return 0
    if args.replay:
        return _replay()

    found = 0
    for name in args.path or sorted(SPECS):
        spec = SPECS[name]
        with activated(spec):
            report = run_spec(spec, args.iterations, args.seed)
        print(f"{name:20s} samples {report.samples:6d}  dispatched {report.dispatched:6d}  "
              f"mismatches {len(report.mismatches)}")
        for case, why in report.mismatches:
            found += 1
            print(f"  MISMATCH {why}\n    {describe(case)}")
            if not args.no_save:
                target = save_case(spec, case, why)
                print(f"    saved {target.relative_to(HERE.parents[1]).as_posix()}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
