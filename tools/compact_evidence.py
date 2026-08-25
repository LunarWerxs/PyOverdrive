"""Rewrite committed benchmark evidence in its compact on-disk form.

The evidence files were 60% of this repository by bytes - 100 files, 3.4 MB -
and most of that was not evidence. They were pretty-printed, they stored
17-significant-digit floats for NANOSECOND timings whose own MAD is tens of
nanoseconds, and every case carried a `speedups` block that is a pure
function of the medians sitting beside it.

Nothing measured is lost. Whitespace carries no information; six significant
figures is a thousand times finer than the noise these numbers describe, so
the extra digits recorded the float representation rather than the
measurement; and a derived field is not evidence.

Because these files ARE the project's evidence, the rewrite verifies itself
rather than being trusted:

- recomputing each dropped speedup from the ORIGINAL medians must
  reproduce the stored value to 1e-9 relative, which is what proves the
  field held no information of its own;
- recomputing it from the ROUNDED medians must stay within what rounding
  can account for, which is a separate and much looser claim - conflating
  the two makes the check fail on its own rounding;
- every non-timing field must survive byte-identical;
- every timing must land within SIG_FIGS of its original.

Any failure aborts before a single file is written.

Idempotent: running it on already-compact files rewrites them to the same
bytes.

Usage:
    .venv/Scripts/python tools/compact_evidence.py [--check]

--check verifies and reports without writing. Exit 1 on any verification
failure, or (with --check) if any file is not already compact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lab.dyno import (  # noqa: E402
    SIG_FIGS,
    compact_case,
    is_correct,
    role_of,
    speedup_of,
)

RESULTS = REPO / "benchmarks" / "results"
REL_TOL = 1e-9          # "the dropped field was derivable" - essentially exact
ROUND_TOL = 1e-4        # "rounding moved it no more than rounding can"


def _verify(original: dict, compacted: dict, path: Path) -> list[str]:
    problems: list[str] = []
    for key in original:
        if key == "cases":
            continue
        if original[key] != compacted.get(key):
            problems.append(f"{path.name}: header field {key!r} changed")

    for i, (before, after) in enumerate(zip(original["cases"], compacted["cases"])):
        where = f"{path.name} case[{i}] {before.get('case')!r}"

        for key in before:
            if key in ("speedups", "variants"):
                continue
            if before[key] != after.get(key):
                problems.append(f"{where}: field {key!r} changed")

        # Two separate claims, checked separately because they have very
        # different tolerances and only the first one is about redundancy.
        #
        #   (1) the dropped field was DERIVABLE: recomputing it from the
        #       ORIGINAL medians must reproduce it essentially exactly.
        #   (2) rounding the medians perturbs that derived value by no more
        #       than the rounding itself, i.e. ~10^-(SIG_FIGS-1) relative.
        #
        # Deriving from the rounded medians and demanding 1e-9 conflates the
        # two and fails on the rounding, which is not a redundancy failure.
        stored = before.get("speedups") or {}
        for name, want in stored.items():
            exact = speedup_of(before, name)
            rounded = speedup_of(after, name)
            if want is None:
                if exact is not None:
                    problems.append(f"{where}: {name} was null, derives to {exact}")
                continue
            if exact is None:
                problems.append(f"{where}: {name} was {want}, derives to null")
                continue
            if abs(exact - want) > REL_TOL * max(abs(want), 1e-300):
                problems.append(
                    f"{where}: {name} was NOT derivable - stored {want!r}, "
                    f"recomputed {exact!r}"
                )
            if rounded is None or abs(rounded - want) > ROUND_TOL * max(abs(want), 1e-300):
                problems.append(
                    f"{where}: {name} moves more than rounding allows - "
                    f"{want!r} -> {rounded!r}"
                )

        for name, stats in (before.get("variants") or {}).items():
            new = after["variants"][name]

            # role and correct are no longer stored; they must RECONSTRUCT
            # to exactly what was there, which is the whole claim.
            if "role" in stats and role_of(after, name) != stats["role"]:
                problems.append(
                    f"{where}: {name}.role was {stats['role']!r}, "
                    f"reconstructs as {role_of(after, name)!r}"
                )
            if "correct" in stats and is_correct(new) != bool(stats["correct"]):
                problems.append(
                    f"{where}: {name}.correct was {stats['correct']!r}, "
                    f"reads back as {is_correct(new)!r}"
                )
            if stats.get("correct") is False and new.get("correct") is not False:
                problems.append(
                    f"{where}: {name} was a CORRECTNESS FAILURE and that is "
                    "evidence - it must stay written explicitly"
                )

            for k, v in stats.items():
                if k in ("role", "correct"):
                    continue  # handled above, by reconstruction
                nv = new.get(k)
                if isinstance(v, float) and v == v and abs(v) != float("inf"):
                    if nv is None or abs(nv - v) > 10 ** -(SIG_FIGS - 1) * max(abs(v), 1e-300):
                        problems.append(f"{where}: {name}.{k} {v!r} -> {nv!r}")
                elif v != nv:
                    problems.append(f"{where}: {name}.{k} {v!r} -> {nv!r}")
    return problems


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    files = sorted(RESULTS.glob("*/*.json"))
    if not files:
        print("no evidence files found")
        return 1

    plan: list[tuple[Path, str, int, int]] = []
    problems: list[str] = []
    foreign: list[Path] = []

    for path in files:
        raw = path.read_text(encoding="utf-8")
        original = json.loads(raw)
        # Not every file under results/ is a Dyno suite - tools/calibrate_dispatch.py
        # writes its own shape. Skip those LOUDLY rather than crashing on the
        # missing key, and never silently: a malformed Dyno suite would look
        # exactly like a foreign one, so both get named.
        if "cases" not in original:
            foreign.append(path)
            continue
        compacted = dict(original)
        compacted["cases"] = [compact_case(c) for c in original["cases"]]
        problems += _verify(original, compacted, path)
        text = json.dumps(compacted, separators=(",", ":"))
        plan.append((path, text, len(raw.encode()), len(text.encode())))

    for path in foreign:
        print(f"skipped (no 'cases' key, not a Dyno suite): "
              f"{path.relative_to(RESULTS.parent.parent)}")
    if not plan:
        print("no Dyno suite files found")
        return 1

    if problems:
        print(f"VERIFICATION FAILED ({len(problems)} problems), nothing written:")
        for p in problems[:20]:
            print("  !!", p)
        return 1

    before = sum(b for _, _, b, _ in plan)
    after = sum(a for _, _, _, a in plan)
    print(f"{len(plan)} files: {before / 1024:.0f} KB -> {after / 1024:.0f} KB "
          f"({100 * (1 - after / before):.1f}% smaller)")
    print(f"verification: stored speedups were derivable to {REL_TOL:g} relative "
          f"(so the dropped field held no information), rounding moves them by at "
          f"most {ROUND_TOL:g}, non-timing fields byte-identical, timings within "
          f"{SIG_FIGS} significant figures")

    stale = [p for p, text, b, _ in plan if p.read_text(encoding="utf-8") != text]
    if check_only:
        if stale:
            print(f"--check: {len(stale)} file(s) are not in compact form")
            return 1
        print("--check: every file already compact")
        return 0

    for path, text, _, _ in plan:
        path.write_text(text, encoding="utf-8")
    print(f"rewrote {len(stale)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
