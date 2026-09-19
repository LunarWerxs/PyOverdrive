"""Re-judge every calibrated row against each numpy this package supports.

Every threshold in this project was measured once, against one numpy, and
then shipped as a promise about all of them. Batch 16 found what that costs:
searchsorted's int64 row wins 1.9-4.0x on numpy 2.4.5 and LOSES 0.26-0.74x
on 2.5.2 - same box, same code, same day. numpy 2.5 made stock searchsorted
fast enough that sorting the queries could no longer repay the argsort. The
row was not measured wrong. It rotted underneath us, and nothing in the
repo could have noticed, because a sweep only ever runs against whatever
numpy the box happens to have.

That is a different failure from every other one this project has found. A
bad measurement is caught by measuring again; rot is caught only by
measuring somewhere else. The package claims a floor in pyproject.toml, so
"somewhere else" means every version at or above it, and this is that sweep.

Its first full run moved that floor. Against 2.0.2, 2.1.3, 2.2.6, 2.3.5,
2.4.6 and 2.5.2 it found two unrelated families falling off the same cliff
between 2.2 and 2.3 - matmul_split_complex at 0.01-0.05x and four of
intersect_sorted's integer rows at 0.52-0.63x - so numpy>=2.0 had been a
false claim for as long as those paths had shipped. The floor is 2.3 now.
Correctness CI could not have caught it: every one of those versions passes
the whole test suite, and being 100x slower than stock is not a wrong
answer, just a worthless one.

It works because pyoverdrive is pure Python: a bare venv with nothing but
numpy in it can import the package straight from src/ and run the whole
row sweep. No build, no install, no wheel.

    .venv/Scripts/python tools/verify_across_numpy.py
    .venv/Scripts/python tools/verify_across_numpy.py --versions 2.4.5,2.5.2
    .venv/Scripts/python tools/verify_across_numpy.py --sizes    # slow, deep
    .venv/Scripts/python tools/verify_across_numpy.py --json-dir evidence --require-quiet

RUN IT ON THE QUIET BOX. Every caveat in verify_no_pessimization.py applies
here once per version - these are timings, and a contended machine produces
a table of noise with a version label on it.

A version whose wheel does not exist for this interpreter is reported as
SKIPPED with the reason, never silently dropped: a matrix that quietly
narrows itself reads exactly like a matrix that passed. That is not
hypothetical and it shaped this project's evidence: numpy 2.0.2 has no
cp313 wheel, the quiet Intel box has only 3.13 and 3.14, so the floor
version could ONLY be timed on the busy box, under a python 3.12 the quiet
one does not have. Run this under the oldest interpreter that can reach the
floor - `py -3.12 tools/verify_across_numpy.py` - or the floor row silently
is not covered. Since the floor moved to 2.3 (cp313 wheels exist) the quiet
box can now judge every supported version itself.

Exit 0 = every requested version completed, measured at least one cell,
and every measured cell is at or above --min. Exit 1 also covers unavailable
versions and failed or incomplete sweeps, even when their partial tables
contain useful measurements. --only narrows this claim to the selected cells.
--json-dir writes a separate evidence file for each requested version;
--require-quiet rejects unavailable or contended load measurements.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[1]

# THE WORKFLOW IS THE LIST. Which numpys this package claims to support is
# already stated in .github/workflows/ci.yml's matrix, and a second copy of
# a claim is a copy that drifts - this repo has now been bitten twice by a
# number that was true when it was written. Read it instead.
_FALLBACK_VERSIONS = ("2.3.0", "2.4.5", "latest")


def supported_versions() -> tuple[tuple[str, ...], str]:
    """(versions, where they came from). Falls back loudly, never silently."""
    wf = REPO / ".github" / "workflows" / "ci.yml"
    try:
        text = wf.read_text(encoding="utf-8")
    except OSError as exc:
        return _FALLBACK_VERSIONS, f"FALLBACK - could not read ci.yml ({exc})"
    found = re.findall(r'numpy:\s*"([^"]+)"', text)
    seen: list[str] = []
    for raw in found:
        v = raw.lstrip("=").strip()
        if v and v not in seen:
            seen.append(v)
    if not seen:
        return (_FALLBACK_VERSIONS,
                "FALLBACK - ci.yml has no numpy matrix entries this can read")
    return tuple(seen), f"{wf.name} matrix"

CELL = re.compile(r"^\s+(\d+\.\d+)x\s+(\S+)\s+(\S+)\s*$")
SUMMARY = re.compile(
    r"^judged (\d+) dispatching paths in their own processes, skipped (\d+) "
    r"\(no canonical input, or it does not dispatch\)$", re.MULTILINE)
LOSS_SUMMARY = re.compile(r"^PESSIMIZATION: ([1-9]\d*) path\(s\) below .+x$", re.MULTILINE)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _venv_for(version: str, cache: Path) -> tuple[Path | None, str]:
    """(interpreter, actual numpy version) for `version`, building the venv
    if it is not already there. Returns (None, reason) when unavailable."""
    home = cache / f"np{version}"
    py = home / "Scripts" / "python.exe"
    if not py.exists():
        py_posix = home / "bin" / "python"
        if py_posix.exists():
            py = py_posix
    if not py.exists():
        r = _run([sys.executable, "-m", "venv", str(home)])
        if r.returncode:
            return None, f"venv failed: {r.stderr.strip()[:120]}"
        py = home / "Scripts" / "python.exe"
        if not py.exists():
            py = home / "bin" / "python"

    spec = "numpy" if version == "latest" else f"numpy=={version}"
    upgrade = ["--upgrade"] if version == "latest" else []
    r = _run([str(py), "-m", "pip", "-q", "install", *upgrade, spec])
    if r.returncode:
        tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]
        return None, f"pip could not install {spec}: {tail[0][:140]}"
    r = _run([str(py), "-c", "import numpy;print(numpy.__version__)"])
    if r.returncode:
        return None, f"numpy will not import: {r.stderr.strip()[:120]}"
    return py, r.stdout.strip()


def _parse_args(argv: list[str], discovered: tuple[str, ...], source: str):
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", default=",".join(discovered),
                    help=f"default is read from {source}: "
                         f"{', '.join(discovered)}")
    ap.add_argument("--cache", default=str(Path.home() / ".pyoverdrive-npvenvs"),
                    help="where the per-version venvs live (reused between runs)")
    ap.add_argument("--min", type=float, default=1.0)
    ap.add_argument("--json-dir", type=Path,
                    help="write one JSON evidence file per requested NumPy version")
    ap.add_argument("--require-quiet", action="store_true",
                    help="require known, quiet foreign CPU load for every version sweep")
    ap.add_argument("--sizes", action="store_true",
                    help="also run the size sweep in every version (slow)")
    ap.add_argument("--shapes", action="store_true")
    ap.add_argument("--values", action="store_true",
                    help="also run the CARDINALITY axis in every version. "
                         "This is the one that answers whether a "
                         "low-cardinality loss is rot or simply a corner "
                         "nobody had measured - a distinction no single "
                         "version can make.")
    ap.add_argument("--only", action="append", default=[],
                    help="restrict to cells whose name contains this "
                         "(repeatable), passed straight through. A full "
                         "cross-version run is one whole sweep PER VERSION; "
                         "this makes a targeted question affordable. It "
                         "narrows coverage, so a green under --only is never "
                         "a green for the suite.")
    return ap.parse_args(argv[1:])


def _sweep_failure(proc, cells: dict[str, float]) -> str | None:
    """Require completion and coverage evidence, not just a plausible row."""
    out = proc.stdout or ""
    summary = SUMMARY.search(out)
    loss_summary = LOSS_SUMMARY.search(out)
    success = re.search(r"^no dispatching path is below .+x$", out, re.MULTILINE)
    unverified = re.search(r"^NOT verified: (.+)$", out, re.MULTILINE)
    reason = None
    if proc.returncode not in (0, 1):
        reason = f"child exited {proc.returncode}"
    elif unverified is not None:
        reason = unverified.group(1)
    elif summary is None:
        reason = "child did not emit a completed sweep summary"
    elif int(summary.group(1)) == 0:
        reason = "child measured zero cells"
    elif int(summary.group(1)) != len(cells):
        reason = f"incomplete table: summary judged {summary.group(1)}, received {len(cells)} cells"
    elif proc.returncode == 1 and loss_summary is None:
        reason = "child exited 1 without a completed loss verdict"
    elif proc.returncode == 0 and (success is None or loss_summary is not None):
        reason = "child exited 0 without a completed success verdict"
    if reason is not None and proc.stderr:
        reason += ": " + proc.stderr.strip()
    return reason


def _run_one_version(version: str, cache: Path,
                     args) -> tuple[str | None, dict[str, float], str | None]:
    """(info, cells, failure_reason) for one version, retaining partial rows.

    A completed losing sweep is valid evidence; a crashed/incomplete one is
    not verified. Neither is allowed to make the overall run pass.
    """
    py, info = _venv_for(version, cache)
    if py is None:
        print(f"SKIPPED numpy {version}: {info}")
        return None, {}, info
    label = f"{version} ({info})" if version == "latest" else info
    print(f"\n=== numpy {label} " + "=" * 40)
    cmd = [str(py), "-u", str(REPO / "tools" / "verify_no_pessimization.py"),
           "--rows", "-v", "--min", str(args.min)]
    if args.sizes:
        cmd.append("--sizes")
    if args.shapes:
        cmd.append("--shapes")
    if args.values:
        cmd.append("--values")
    for pat in args.only:
        cmd += ["--only", pat]
    if getattr(args, "json_dir", None) is not None:
        evidence_dir = Path(args.json_dir).resolve()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        # Name by the requested selector: latest and a pin may resolve to
        # the same installed version but still need separate run evidence.
        evidence = evidence_dir / f"numpy-{quote(version, safe='')}.json"
        cmd += ["--json", str(evidence)]
        print(f"  evidence: {evidence}")
    if getattr(args, "require_quiet", False):
        cmd.append("--require-quiet")
    proc = _run(cmd, cwd=str(REPO))
    out = proc.stdout or ""
    for line in out.splitlines():
        if line.startswith(("python ", "cpu classes", "measuring only")):
            print("  " + line)
    cells = {}
    for line in out.splitlines():
        m = CELL.match(line)
        if m:
            cells[m.group(2)] = float(m.group(1))
    below = {k: v for k, v in cells.items() if v < args.min}
    print(f"  {len(cells)} cells judged, {len(below)} below {args.min:g}x")
    summary = SUMMARY.search(out)
    if summary is not None:
        print(f"  {summary.group(2)} cells skipped (no canonical input, or no dispatch)")
    failure = _sweep_failure(proc, cells)
    if failure is None and proc.returncode == 1 and not below:
        failure = "child reported confirmed losses, but the parsed table contains none"
    if failure is not None:
        print(f"  NOT verified: {failure}")
    return info, cells, failure


def _collect_all(versions: list[str], cache: Path,
                 args) -> tuple[dict[str, dict[str, float]], list[tuple[str, str]]]:
    ran: dict[str, dict[str, float]] = {}
    skipped: list[tuple[str, str]] = []
    completed = 0
    for version in versions:
        info, cells, skip_reason = _run_one_version(version, cache, args)
        if info is not None and cells:
            label = f"latest ({info})" if version == "latest" else info
            ran[label] = cells
        if skip_reason is not None:
            skipped.append((version, skip_reason))
        elif info is not None and cells:
            completed += 1
    print(f"\nVersion coverage: {completed}/{len(versions)} completed with measurements; "
          f"{len(skipped)} unverified")
    return ran, skipped


def _compare(ran: dict[str, dict[str, float]],
            args) -> tuple[list[str], list[str], list[str], list[str]]:
    """(labels, every, losses, interesting).

    A row that loses on ANY supported version is a loss. The comparison
    across versions is the point: a cell that halves between two numpys is
    rotting even while it still clears the bar.
    """
    labels = list(ran)
    every = sorted({c for cells in ran.values() for c in cells})
    losses = [c for c in every
              if any(ran[v].get(c, float("inf")) < args.min for v in labels)]

    # a COPY: `interesting` grows to include the movers below, and aliasing
    # `losses` here would fold them into the loss count and the exit code
    interesting = list(losses)
    for c in every:
        vals = [ran[v].get(c) for v in labels]
        known = [x for x in vals if x is not None]
        # flag anything that moved by more than 25% between versions, even
        # when every reading still passes: that is rot in progress
        if c not in interesting and len(known) > 1 and min(known) < 0.75 * max(known):
            interesting.append(c)
    return labels, every, losses, interesting


def _report(labels: list[str], losses: list[str], interesting: list[str],
           ran: dict[str, dict[str, float]], skipped: list[tuple[str, str]],
           args) -> int:
    print(f"\n{'cell':38s} " + " ".join(f"{v:>9s}" for v in labels))
    for c in sorted(interesting):
        cells = " ".join(
            (f"{ran[v][c]:>8.2f}x" if c in ran[v] else f"{'-':>9s}") for v in labels)
        flag = "!!" if c in losses else "  "
        print(f"{flag} {c:36s} {cells}")
    if not interesting:
        print("  (no cell lost anywhere, and none moved more than 25% "
              "between versions)")

    if skipped:
        print("\nNOT verified, so not covered by this run:")
        for version, why in skipped:
            print(f"  - numpy {version}: {why}")
    if losses:
        print(f"\nPESSIMIZATION on at least one supported numpy: "
              f"{len(losses)} cell(s)")
        return 1
    if skipped or not ran or any(not cells for cells in ran.values()):
        print("\nNOT verified: every requested version must complete and measure cells")
        return 1
    if args.only:
        print(f"\nevery cell MATCHING --only {args.only} is at or above "
              f"{args.min:g}x on all {len(ran)} version(s) that ran - this "
              f"is a probe, and says nothing about the cells it filtered out")
        return 0
    print(f"\nevery cell is at or above {args.min:g}x on all "
          f"{len(ran)} version(s) that ran")
    return 0


def main(argv: list[str]) -> int:
    discovered, source = supported_versions()
    args = _parse_args(argv, discovered, source)

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    print(f"numpy versions from {source}: {', '.join(versions)}")
    if source.startswith("FALLBACK"):
        print("  ^ the workflow could not be read, so this list may be stale")

    ran, skipped = _collect_all(versions, cache, args)
    if not ran:
        print("\nno version ran - nothing was verified")
        return 1

    labels, _every, losses, interesting = _compare(ran, args)
    return _report(labels, losses, interesting, ran, skipped, args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
