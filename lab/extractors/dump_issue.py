"""Print a fetched raw issue dump as readable text for analysis.

Usage: python lab/extractors/dump_issue.py lab/corpus/raw/numpy_numpy_31969.json [...]

Prints title/state/labels, full body (truncated), every comment, and the
cross-referenced PRs/commits mined from the timeline. Deterministic parsing
belongs here; interpretation belongs to the analyst (spec section 7.3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MAX_BODY = 9000
MAX_COMMENT = 3500

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def dump(path: str) -> None:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    issue = d["issue"]
    labels = [lb["name"] for lb in issue.get("labels", [])]
    print("=" * 70)
    print(f"{d['repo']}#{d['number']}: {issue['title']}")
    print(
        f"state={issue['state']} created={issue['created_at']} "
        f"closed={issue.get('closed_at')} labels={labels}"
    )
    print(f"author={issue['user']['login']} retrieved={d['retrieved_at']}")
    print("-" * 70)
    print((issue.get("body") or "")[:MAX_BODY])
    for c in d.get("comments", []):
        print("-" * 70)
        print(f"COMMENT {c['user']['login']} {c['created_at']}:")
        print((c.get("body") or "")[:MAX_COMMENT])
    xrefs: list[str] = []
    for ev in d.get("timeline", []):
        event = ev.get("event")
        if event == "cross-referenced":
            src = ev.get("source", {}).get("issue") or {}
            if src:
                kind = "PR" if src.get("pull_request") else "issue"
                repo = (src.get("repository") or {}).get("full_name", "?")
                xrefs.append(
                    f"  {kind} {repo}#{src.get('number')}: "
                    f"{src.get('title')} [{src.get('state')}]"
                )
        elif event in ("closed", "referenced") and ev.get("commit_id"):
            xrefs.append(f"  commit {ev['commit_id'][:10]} ({event})")
    if xrefs:
        print("-" * 70)
        print("CROSS-REFERENCES (from timeline):")
        print("\n".join(dict.fromkeys(xrefs)))


for arg in sys.argv[1:]:
    dump(arg)
