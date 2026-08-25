"""Check the counts the README and CHANGELOG advertise against the registry.

Those documents make arithmetic claims - how many fast-path families, how
many registered paths, how many are bit-identical. Every one of them is
derivable from the live Gearbox, and every one of them was being maintained
by hand. They had drifted: the published "bit-identical on 48 of the 71
registered paths" counted two paths that are bit-identical only for INTEGER
dtypes and numeric for floats, and it counted a denominator that included a
disabled test artifact.

That is the kind of error nobody notices, because the number is plausible,
nobody re-derives it, and each release nudges it by one. So it is a check
now rather than a habit.

Usage:
    .venv/Scripts/python tools/verify_claims.py

Exit 0 = every advertised number matches the registry.
Exit 1 = drift, with the real numbers printed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pyoverdrive  # noqa: E402
from pyoverdrive.dispatcher.gearbox import GEARBOX  # noqa: E402


def registry_counts() -> dict:
    pyoverdrive.enable()
    fn = [p for lst in GEARBOX._paths.values()
          for p in (lst if isinstance(lst, list) else [lst])]
    cls = list(GEARBOX._class_paths.values())
    every = fn + cls

    live = [p for p in every if p.enabled]
    modes = [(p.name, p.provenance.get("comparison_mode") or "") for p in live]

    exact = [n for n, m in modes if m == "bit-identical"]
    partial = [n for n, m in modes if m != "bit-identical" and "bit-identical" in m]
    other = [n for n, m in modes if "bit-identical" not in m]
    undeclared = [p.name for p in every if not p.provenance.get("comparison_mode")]

    return {
        "registered": len(every),
        "disabled": sorted(p.name for p in every if not p.enabled),
        "live": len(live),
        "families": len({p.op for p in live}),
        "exact": len(exact),
        "partial": sorted(partial),
        "other": len(other),
        "undeclared": undeclared,
    }


def main() -> int:
    c = registry_counts()
    problems = []

    print(f"registered paths      : {c['registered']}")
    print(f"  disabled by default : {c['disabled']}")
    print(f"always-on paths       : {c['live']}")
    print(f"always-on families    : {c['families']}")
    print(f"  bit-identical       : {c['exact']}")
    print(f"  integer-only exact  : {len(c['partial'])} {c['partial']}")
    print(f"  numeric / set-equal : {c['other']}")

    if c["undeclared"]:
        problems.append(
            "paths with NO comparison_mode recorded, which the docs claim "
            f"cannot happen: {c['undeclared']}"
        )

    total = c["exact"] + len(c["partial"]) + c["other"]
    if total != c["live"]:
        problems.append(f"mode counts sum to {total}, expected {c['live']}")

    # The advertised sentence, in both documents, in one machine-checkable
    # shape. Keep the wording in step with these patterns when editing.
    want_exact = re.compile(
        r"[Bb]it-identical\s+(?:results\s+)?(?:to\s+stock\s+)?on\s+(\d+)\s+of\s+"
        r"the\s+(\d+)\s+always-on\s+paths"
    )
    for doc in ("README.md", "CHANGELOG.md"):
        text = (REPO / doc).read_text(encoding="utf-8")
        found = want_exact.findall(text)
        if not found:
            problems.append(
                f"{doc}: could not find the bit-identical claim in the "
                "checkable form 'bit-identical on N of the M always-on paths'"
            )
            continue
        for got_exact, got_total in found:
            if int(got_exact) != c["exact"] or int(got_total) != c["live"]:
                problems.append(
                    f"{doc}: claims {got_exact} of {got_total} always-on paths "
                    f"bit-identical; registry says {c['exact']} of {c['live']}"
                )

    # Only the path counts are checked, not the family count: "family" in
    # the docs is a curated grouping (the three singular-value paths are one
    # family, and they patch three different numpy names), so it is not
    # derivable from the registry and a check that guessed at it would be
    # worse than no check.
    # \s+ rather than a literal space throughout: these documents are hard
    # wrapped, so any of these numbers can land either side of a line break
    want_paths = re.compile(r"\((\d+)\s+always-on\s+paths\s+of\s+(\d+)\s+registered")
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    m = want_paths.search(text)
    if not m:
        problems.append(
            "CHANGELOG: could not find the path counts in the checkable form "
            "'(N always-on paths of M registered'"
        )
    else:
        if int(m.group(1)) != c["live"]:
            problems.append(
                f"CHANGELOG: claims {m.group(1)} always-on paths; "
                f"registry says {c['live']}"
            )
        if int(m.group(2)) != c["registered"]:
            problems.append(
                f"CHANGELOG: claims {m.group(2)} registered paths; "
                f"registry says {c['registered']}"
            )

    if problems:
        print("\nCLAIMS: DRIFTED")
        for p in problems:
            print("  !!", p)
        return 1
    print("\nCLAIMS: MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
