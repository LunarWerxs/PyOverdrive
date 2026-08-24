"""Fetch GitHub issues (body + comments + timeline) into lab/corpus/raw/.

Uses the authenticated `gh` CLI so rate limits are the 5000/hr authenticated
tier. Raw dumps are gitignored (refetchable third-party content); only the
normalized opportunity records derived from them are committed.

Usage:
    python lab/collectors/github_fetch.py 31969 12778 ...
    python lab/collectors/github_fetch.py --repo scipy/scipy 12345

This is the seed of the ProsPyctor collector. Full incremental repo-wide sync
(resumable cursors, content hashing, link resolution; spec section 7.2) is
Phase 1 work tracked in docs/BUILD_SPEC.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "corpus" / "raw"


def gh_api(path: str, paginate: bool = False):
    cmd = ["gh", "api", path, "-H", "Accept: application/vnd.github+json"]
    if paginate:
        cmd += ["--paginate", "--slurp"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {proc.stderr[:500]}")
    data = json.loads(proc.stdout)
    if paginate:
        # --slurp wraps pages in an outer array; flatten page lists
        flat = []
        for page in data:
            flat.extend(page if isinstance(page, list) else [page])
        return flat
    return data


def fetch_issue(repo: str, number: int) -> Path:
    base = f"repos/{repo}/issues/{number}"
    record = {
        "repo": repo,
        "number": number,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "issue": gh_api(base),
        "comments": gh_api(f"{base}/comments", paginate=True),
        "timeline": gh_api(f"{base}/timeline", paginate=True),
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{repo.replace('/', '_')}_{number}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    state = record["issue"].get("state", "?")
    n_comments = len(record["comments"])
    print(f"fetched {repo}#{number} [{state}] {n_comments} comments -> {out.name}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("numbers", nargs="+", type=int)
    parser.add_argument("--repo", default="numpy/numpy")
    args = parser.parse_args()
    failures = 0
    for n in args.numbers:
        try:
            fetch_issue(args.repo, n)
        except Exception as exc:
            failures += 1
            print(f"FAILED {args.repo}#{n}: {exc}", file=sys.stderr)
        time.sleep(0.3)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
