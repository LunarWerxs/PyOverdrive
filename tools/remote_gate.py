"""Ship HEAD to a second machine and run the full gate there.

The two-machine law says a cell ships only if it wins on both boxes, so the
second machine gets exercised every batch. It had been done by hand -
zip, copy, unzip, remember which commands - and hand-carrying is how the
results-file overwrite happened once already. This is that sequence, once,
with the verdict parsed rather than eyeballed.

The second box is a plain COPY of the tree, not a clone, so nothing here
touches git on the far side. `git archive HEAD` means what is verified is
what is committed, never a dirty working tree.

No host name lives in this file: pass the ssh alias. Define one in your ssh
client config with its own IdentityFile, because a box with a dedicated key
will refuse the default one and the failure reads as "the box is down".

Usage:
    .venv/Scripts/python tools/remote_gate.py <ssh-alias> [--remote-dir DIR]
    .venv/Scripts/python tools/remote_gate.py <ssh-alias> --selfcheck-only

Exit 0 = gate green on the far side.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOISE = re.compile(r"post-quantum|store now|may need to be upgraded|openssh\.com/pq")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _ssh(alias: str, powershell: str) -> subprocess.CompletedProcess:
    """One ssh hop running one PowerShell command on a Windows box."""
    return _run(["ssh", "-o", "BatchMode=yes", alias,
                 f'powershell -NoProfile -Command "{powershell}"'])


def _clean(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if l.strip() and not NOISE.search(l))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("alias", help="ssh alias of the second machine")
    ap.add_argument("--remote-dir", default=r"C:\PyOverdrive")
    ap.add_argument("--selfcheck-only", action="store_true")
    args = ap.parse_args(argv[1:])

    tmp = Path(tempfile.mkdtemp(prefix="pyov_remote_"))
    zip_path = tmp / "incoming.zip"

    head = _run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).stdout.strip()
    print(f"shipping {head} to {args.alias}:{args.remote_dir}")
    r = _run(["git", "archive", "-o", str(zip_path), "HEAD"], cwd=REPO)
    if r.returncode:
        print("git archive failed:", r.stderr.strip())
        return 1

    r = _run(["scp", "-q", str(zip_path), f"{args.alias}:{args.remote_dir}/incoming.zip"])
    if r.returncode:
        print("scp failed:", _clean(r.stderr))
        return 1

    # -Force so tracked files are replaced; untracked local state (.venv,
    # benchmark results) is left alone, which is the point of not using git
    r = _ssh(args.alias, (
        f"Expand-Archive -Path {args.remote_dir}\\incoming.zip "
        f"-DestinationPath {args.remote_dir} -Force; "
        f"Write-Output EXPANDED"
    ))
    if "EXPANDED" not in r.stdout:
        print("expand failed:", _clean(r.stdout + r.stderr))
        return 1

    # NO double quotes inside this python -c: it is already nested two deep
    # (ssh -> powershell -Command "..." -> python -c '...'), and a third
    # quoting level silently produces an empty result rather than an error.
    # Which interpreter and numpy produced the green IS the evidence, so a
    # missing line here is a failure, not cosmetic.
    r = _ssh(args.alias, (
        f"cd {args.remote_dir}; .\\.venv\\Scripts\\python.exe -c "
        f"'import sys,numpy,pyoverdrive;"
        f"print(sys.version.split()[0],numpy.__version__,pyoverdrive.__version__)'"
    ))
    env_line = _clean(r.stdout)
    if not env_line:
        print("could not read the remote environment:", _clean(r.stderr)[-300:])
        return 1
    print(f"remote env (python numpy pyoverdrive): {env_line}")

    r = _ssh(args.alias, (
        f"cd {args.remote_dir}; "
        f".\\.venv\\Scripts\\python.exe -m pyoverdrive --selfcheck"
    ))
    self_out = _clean(r.stdout + r.stderr)
    verdict = [l for l in self_out.splitlines() if "selfcheck:" in l]
    print(verdict[-1] if verdict else self_out[-400:])
    if not verdict or "0 failing" not in verdict[-1]:
        return 1
    if args.selfcheck_only:
        return 0

    r = _ssh(args.alias, (
        f"cd {args.remote_dir}; "
        f".\\.venv\\Scripts\\python.exe -u -m pytest tests compatibility -q"
    ))
    out = _clean(r.stdout + r.stderr)
    tail = [l for l in out.splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
    print(tail[-1] if tail else out[-600:])
    return 0 if (tail and "failed" not in tail[-1] and "error" not in tail[-1]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
