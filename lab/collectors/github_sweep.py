"""Batch sweep of GitHub issue-search angles into a candidate JSONL.

The banked instrument behind the mining pipeline's first stage (batch-3
and batch-5 ran as throwaway scratchpad scripts; this is the permanent
version, extended for batch 6 with CLOSED-issue angles and adjacent
trackers). Uses the authenticated `gh` CLI (search tier: 30 req/min).

Dedup: candidates already present in lab/corpus/OPP-*.yaml or linked
from docs/research/batch*-shortlist.md are dropped (numpy/numpy
namespace; adjacent repos have no prior coverage).

Usage:
    python lab/collectors/github_sweep.py --out <dir> [--angles a,b,...]

Output: <dir>/candidates.jsonl (one candidate per line, deduped,
highest-signal angle wins) and a per-angle count summary on stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# (angle, repo, query). Query strings are the search-API q= value; the
# script adds repo: and is:issue itself so the table stays readable.
ANGLES = [
    # numpy CLOSED issues - fixed-partially, wontfix, and by-design gold
    ("n-closed-slower", "numpy/numpy", 'is:closed "slower" in:title'),
    ("n-closed-slow", "numpy/numpy", 'is:closed "slow" in:title'),
    ("n-closed-perf", "numpy/numpy", 'is:closed "performance" in:title'),
    ("n-closed-faster", "numpy/numpy", 'is:closed "faster" in:title'),
    ("n-times-faster", "numpy/numpy", '"times faster" in:body'),
    ("n-x-slower", "numpy/numpy", '"x slower" in:body'),
    ("n-overhead", "numpy/numpy", '"overhead" in:title'),
    ("n-quadratic", "numpy/numpy", "quadratic"),
    ("n-regression", "numpy/numpy", 'label:"06 - Regression" performance'),
    # numpy OPEN, re-sorted so different heads surface than batch 3/5
    ("n-open-comments", "numpy/numpy", "is:open performance", "comments"),
    ("n-open-react", "numpy/numpy", 'is:open "slow"', "reactions"),
    # adjacent trackers
    ("s-label-perf", "scipy/scipy", "label:performance"),
    ("s-numpy-slow", "scipy/scipy", "numpy slower"),
    ("p-numpy-slow", "pandas-dev/pandas", '"numpy" "slower" performance'),
    ("b-tracker", "pydata/bottleneck", "numpy faster"),
    ("x-tracker", "pydata/numexpr", "numpy faster"),
]


# shortlist prose names issues as "numpy#123", "scipy#123", etc.; map the
# shorthand back to the repo the sweep would rediscover it under
_SHORTHAND_REPOS = {
    "numpy": "numpy/numpy",
    "scipy": "scipy/scipy",
    "pandas": "pandas-dev/pandas",
    "bottleneck": "pydata/bottleneck",
    "numexpr": "pydata/numexpr",
}


def seen_numbers() -> set[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    for y in (REPO_ROOT / "lab" / "corpus").glob("OPP-*.yaml"):
        for m in re.finditer(r"number:\s*(\d+)", y.read_text(encoding="utf-8")):
            seen.add(("numpy/numpy", int(m.group(1))))
    for md in (REPO_ROOT / "docs" / "research").glob("batch*-shortlist.md"):
        text = md.read_text(encoding="utf-8")
        for m in re.finditer(r"github\.com/([\w.-]+/[\w.-]+)/issues/(\d+)", text):
            seen.add((m.group(1), int(m.group(2))))
        for m in re.finditer(r"\b(numpy|scipy|pandas|bottleneck|numexpr)#(\d+)", text):
            seen.add((_SHORTHAND_REPOS[m.group(1)], int(m.group(2))))
    # triage verdicts banked by past sweeps: a title-rejected candidate stays
    # rejected until someone deliberately revisits it, so every future sweep
    # only surfaces genuinely new ground
    rejects = REPO_ROOT / "lab" / "corpus" / "triage-rejects.jsonl"
    if rejects.exists():
        for line in rejects.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                seen.add((rec["repo"], int(rec["number"])))
    return seen


def search(repo: str, query: str, sort: str | None) -> list[dict]:
    cmd = [
        "gh", "api", "-X", "GET", "search/issues",
        "-f", f"q=repo:{repo} is:issue {query}",
        "-F", "per_page=100",
        "-H", "Accept: application/vnd.github+json",
    ]
    if sort:
        cmd += ["-f", f"sort={sort}", "-f", "order=desc"]
    for attempt in (1, 2):
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode == 0:
            return json.loads(proc.stdout).get("items", [])
        if attempt == 1:
            time.sleep(5)  # transient network/TLS hiccups deserve one retry
    print(f"  QUERY FAILED ({proc.stderr.strip()[:200]})", file=sys.stderr)
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--angles", default="")
    args = parser.parse_args()
    only = set(args.angles.split(",")) if args.angles else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = seen_numbers()
    print(f"dedup baseline: {len(seen)} previously-seen issues")

    kept: dict[tuple[str, int], dict] = {}
    for spec in ANGLES:
        angle, repo, query = spec[0], spec[1], spec[2]
        sort = spec[3] if len(spec) > 3 else None
        if only and angle not in only:
            continue
        items = search(repo, query, sort)
        fresh = 0
        for it in items:
            if "pull_request" in it:
                continue
            key = (repo, it["number"])
            if key in seen or key in kept:
                continue
            fresh += 1
            kept[key] = {
                "repo": repo,
                "number": it["number"],
                "state": it["state"],
                "title": it["title"],
                "comments": it.get("comments", 0),
                "reactions": (it.get("reactions") or {}).get("total_count", 0),
                "created_at": it.get("created_at"),
                "closed_at": it.get("closed_at"),
                "labels": [l["name"] for l in it.get("labels", [])],
                "url": it["html_url"],
                "angle": angle,
            }
        print(f"{angle:18s} {len(items):3d} hits, {fresh:3d} fresh")
        time.sleep(2.5)  # authenticated search tier: 30 req/min

    out = out_dir / "candidates.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for rec in kept.values():
            f.write(json.dumps(rec) + "\n")
    print(f"total fresh candidates: {len(kept)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
