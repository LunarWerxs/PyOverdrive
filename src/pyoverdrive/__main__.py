"""``python -m pyoverdrive``: report, self-check, demo.

    python -m pyoverdrive                 # status report
    python -m pyoverdrive --selfcheck     # every fast path vs stock, exit 1 on any FAIL
    python -m pyoverdrive --demo          # headline ops, stock vs PyOverdrive, live timings
    python -m pyoverdrive --calibrate     # probe calibration-gated paths on THIS machine
    python -m pyoverdrive --json          # status() as JSON
"""

from __future__ import annotations

import argparse
import json
import sys

from . import report, selfcheck, status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pyoverdrive")
    ap.add_argument("--selfcheck", action="store_true", help="run every fast path against stock NumPy")
    ap.add_argument("--demo", action="store_true", help="headline operations, stock vs PyOverdrive, timed live")
    ap.add_argument("--quick", action="store_true", help="with --demo: smaller sizes, ~5 seconds total")
    ap.add_argument("--calibrate", action="store_true",
                    help="probe calibration-gated paths on this machine and persist the verdicts")
    ap.add_argument("--json", action="store_true", help="print status() as JSON")
    args = ap.parse_args(argv)
    if args.calibrate:
        from .calibration import calibrate

        calibrate()
        return 0
    if args.json:
        print(json.dumps(status(), indent=2, default=str))
        return 0
    if args.demo:
        from .demo import demo

        return demo(quick=args.quick)
    report()
    if args.selfcheck:
        print()
        results = selfcheck()
        return 1 if any(v.startswith("FAIL") for v in results.values()) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
