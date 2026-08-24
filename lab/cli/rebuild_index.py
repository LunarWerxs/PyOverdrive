"""Rebuild lab/corpus/index.sqlite from the YAML opportunity records.

The YAML files are the source of truth (lab/corpus/SCHEMA.md); the SQLite
index is derived, disposable, and exists for querying/ranking as the corpus
grows past what eyeballs handle.

Usage: python lab/cli/rebuild_index.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import yaml

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
DB = CORPUS / "index.sqlite"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    records = []
    for path in sorted(CORPUS.glob("OPP-*.yaml")):
        rec = yaml.safe_load(path.read_text(encoding="utf-8"))
        records.append((path.name, rec))

    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.execute(
        """
        CREATE TABLE opportunities (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            operations TEXT,
            claimed_speedup REAL,
            verified_speedup REAL,
            correctness_risk TEXT,
            implementation_risk TEXT,
            reproducer TEXT,
            n_sources INTEGER,
            record_file TEXT,
            raw_json TEXT
        )
        """
    )
    for fname, rec in records:
        claim = rec.get("claim") or {}
        current = rec.get("current_numpy_result") or {}
        con.execute(
            "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec["id"],
                rec["title"],
                rec.get("status", "unverified"),
                ",".join(rec.get("affected_operations", [])),
                claim.get("speedup"),
                current.get("verified_speedup"),
                rec.get("correctness_risk"),
                rec.get("implementation_risk"),
                rec.get("reproducer"),
                len(rec.get("sources", [])),
                fname,
                json.dumps(rec, default=str),
            ),
        )
    con.commit()

    print(f"indexed {len(records)} records -> {DB.name}")
    for row in con.execute(
        "SELECT id, status, claimed_speedup, verified_speedup, operations "
        "FROM opportunities ORDER BY id"
    ):
        opp, status, claimed, verified, ops = row
        claimed = f"{claimed:g}x" if claimed else "-"
        verified = f"{verified:g}x" if verified else "-"
        print(f"  {opp}  {status:<14} claimed {claimed:<8} verified {verified:<8} {ops}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
