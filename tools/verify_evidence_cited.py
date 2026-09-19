"""Numpy's version belongs in the EVIDENCE and never in the BEHAVIOUR.

Two halves of one doctrine, and this checks both.

A speed claim that does not name the numpy it was measured on cannot be
shown to have rotted. A path that BRANCHES on the numpy version has made
its behaviour version-conditional, which this project ruled against when
searchsorted's int64 row went underwater: "a supported numpy that loses is
a loss, not a case for a version-conditional gate". The row was withdrawn
and later the package floor moved instead. Nothing enforced either half
until now - both were conventions, and conventions are what the three
misfires in batch 16 slipped through.

Batch 16 found the same failure three times in one night, and each time the
thing that caught it was running the measurement somewhere else. A ratio
recorded without its numpy version cannot be re-checked, cannot be shown to
have rotted, and reads as timeless when it is not:

- `searchsorted_sortqueries`' int64 row won 1.9-4.0x on numpy 2.4.5 and lost
  0.26-0.74x on 2.5.2. Same box, same code, same day. The row was not
  measured wrong; numpy got faster underneath it. Withdrawn.
- `unique_sort` and `intersect_sorted`'s 16-bit rows won 4-5x on numpy 2.1
  and 2.2 and lost 0.68-0.75x from 2.3, when np.unique started trying a hash
  table (gh-26018).
- `matmul_split_complex` ran at 0.01-0.05x below numpy 2.3, because the
  BLAS-for-non-contiguous-operands change (gh-23752) is what its whole
  mechanism depends on. The package claimed numpy>=2.0 at the time.

Machine provenance was already the convention here - most modules name a
fingerprint. Numpy provenance was not, and numpy is the axis that moves
under a threshold without anyone touching the repo.

THE RATCHET. Modules that already cite a version must keep citing one, and
a module that does not is only tolerated if it is on the dated legacy list
below. The list may shrink and may never grow: a NEW claim that names no
numpy fails this check.

The legacy entries are NOT fixable by editing a comment. Nobody knows which
numpy those numbers came from, and writing a plausible one in would be
inventing evidence, which is the one thing this project must never do. The
fix for each is a re-measurement, and until then "unknown provenance" is
the honest label.

    .venv/Scripts/python tools/verify_evidence_cited.py
    .venv/Scripts/python tools/verify_evidence_cited.py --list-legacy

Exit 0 = every claim outside the legacy list names a numpy version, and no
shipped path branches on the numpy version.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN = ("src/pyoverdrive/fastpaths", "src/pyoverdrive/parallel")

# "2.63x", "0.88x" - a measured ratio, which is what a calibration claim is
# made of. Deliberately not matching bare integers ("2x"), which are almost
# always prose about a factor rather than a measurement.
CLAIM = re.compile(r"\b\d+\.\d+x\b")
# "numpy 2.5.2", "numpy 2.4", "np 2.3" - enough to re-run the measurement.
CITES_NUMPY = re.compile(r"\bnumpy[ /]?[<>=]{0,2}\s?\d+\.\d+", re.IGNORECASE)

# The nineteen entries recorded 2026-08-26 were re-measured on 2026-09-19.
# Per-module evidence, named NumPy versions, hardware/load qualifications,
# and negative results: docs/research/2026-09-19-burndown.md. Historical
# ratios of unknown version were removed rather than relabelled. No new
# exemption may be added: this ratchet remains in force with an empty set.
LEGACY_UNQUALIFIED: set[str] = set()
# fftconvolve.py is the reason the version pattern requires digits on both
# sides of the dot: its only "numpy 2.x" is quoting an error message, not
# naming a measurement, and a looser pattern counted that as provenance.


def scan() -> tuple[list[str], list[str], list[str]]:
    """(unqualified-and-not-excused, legacy-still-unqualified, graduated)."""
    bad, legacy, graduated = [], [], []
    for root in SCAN:
        base = REPO / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not CLAIM.search(text):
                continue
            cites = bool(CITES_NUMPY.search(text))
            if cites:
                if path.name in LEGACY_UNQUALIFIED:
                    graduated.append(path.name)
            elif path.name in LEGACY_UNQUALIFIED:
                legacy.append(path.name)
            else:
                bad.append(f"{root}/{path.name}")
    return bad, legacy, graduated


# Reporting the running numpy is fine and these files do it on purpose.
_REPORTS_VERSION_OK = {"calibration.py", "diagnostics.py", "demo.py"}
VERSION_BRANCH = re.compile(r"NumpyVersion|np\.__version__|numpy\.__version__")


def scan_version_branching() -> list[str]:
    """Shipped paths that ASK which numpy they are running on.

    The distinction is decide-vs-report: stamping np.__version__ into a
    diagnostics blob or a calibration record is evidence, and evidence is
    the point. Reading it to choose a code path makes the package behave
    differently on two numpys that both pass the test suite, which is the
    thing the searchsorted ruling forbids.
    """
    out = []
    for path in sorted((REPO / "src" / "pyoverdrive").rglob("*.py")):
        if path.name in _REPORTS_VERSION_OK:
            continue
        for i, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if VERSION_BRANCH.search(line):
                rel = path.relative_to(REPO).as_posix()
                out.append(f"{rel}:{i}: {line.strip()[:70]}")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-legacy", action="store_true")
    args = ap.parse_args(argv[1:])

    bad, legacy, graduated = scan()
    branching = scan_version_branching()

    if args.list_legacy:
        print(f"{len(legacy)} module(s) still carrying ratios of unknown "
              f"numpy provenance:")
        for name in sorted(legacy):
            print(f"  {name}")
        print("\nEach leaves this list by being RE-MEASURED against a named "
              "numpy, never by having a version typed into it.")
        return 0

    if graduated:
        print("These now cite a numpy version and can come off "
              "LEGACY_UNQUALIFIED:")
        for name in sorted(graduated):
            print(f"  + {name}")
        print()

    if bad:
        print("SPEED CLAIMS WITH NO NUMPY VERSION, and not on the legacy "
              "list. A ratio that does not name its numpy cannot be shown to "
              "have rotted - which this project has watched happen three "
              "times (see this file's docstring):")
        for name in bad:
            print(f"  !! {name}")
        print("\nName the numpy the numbers were measured on. If that is not "
              "known, re-measure - do NOT write in a plausible version.")
        return 1

    if branching:
        print("SHIPPED CODE BRANCHES ON THE NUMPY VERSION. This project "
              "ruled against version-conditional behaviour when "
              "searchsorted's int64 row went underwater - the row was "
              "withdrawn and the package floor moved, rather than the path "
              "asking which numpy it was running on:")
        for line in branching:
            print(f"  !! {line}")
        print("\nReporting the version (diagnostics, calibration records) "
              "is fine; deciding on it is not.")
        return 1

    print(f"every speed claim outside the legacy list names a numpy version, "
          f"and no path branches on it ({len(legacy)} legacy module(s) still "
          f"to re-measure; --list-legacy to see them)")
    return 0 if not graduated else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
