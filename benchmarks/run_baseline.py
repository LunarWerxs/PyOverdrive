"""One-command baseline: every calibration battery, the public-API proof, the
test suite, and a diff of the calibrated tables against the shipped ones.

    .venv/Scripts/python benchmarks/run_baseline.py            # the standard run
    .venv/Scripts/python benchmarks/run_baseline.py --historical  # + all reproducers
    .venv/Scripts/python benchmarks/run_baseline.py --require-quiet  # refuse on a loaded box
    .venv/Scripts/python benchmarks/run_baseline.py --smoke     # wiring check, no evidence

Order: foreign-load check -> calibration batteries (PYRALLEL-CAL,
PYRALLEL-BIN-CAL, INTERSECT-CAL, RELAYOUT-CAL, FFTCONV-CAL) -> MVP-BASELINE
public-API proof -> [historical reproducers] -> pytest -> calibration-table
diff. Every suite writes its own fingerprinted
JSON under benchmarks/results/ exactly as when run by hand; this script only
sequences them and summarizes. A full log lands in
benchmarks/results/_scratch/ (gitignored).

Read the summary's CONTENDED column before believing any multi-thread number:
this box is shared with other agent sessions and a loaded run understates
every parallel candidate (lab/dyno/load.py).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lab.dyno import CONTENDED_LOAD  # noqa: E402
from lab.dyno.fingerprint import machine_fingerprint  # noqa: E402
from lab.dyno.load import cpu_busy_fraction  # noqa: E402

PY = sys.executable
SCRATCH = REPO_ROOT / "benchmarks" / "results" / "_scratch"

CALIBRATION = [
    ("PYRALLEL-CAL", "benchmarks/micro/bench_pyrallel_calibration.py"),
    ("PYRALLEL-BIN-CAL", "benchmarks/micro/bench_pyrallel_binary_calibration.py"),
    ("INTERSECT-CAL", "benchmarks/micro/bench_intersect_calibration.py"),
    ("RELAYOUT-CAL", "benchmarks/micro/bench_relayout_calibration.py"),
    ("FFTCONV-CAL", "benchmarks/micro/bench_fftconvolve_calibration.py"),
    ("FASTNANQ-CAL", "benchmarks/micro/bench_nanquantile_calibration.py"),
    ("EINSUM-CAL", "benchmarks/micro/bench_einsum_calibration.py"),
    ("SEARCHSORTED-CAL", "benchmarks/micro/bench_searchsorted_calibration.py"),
    ("SMALLINT-CAL", "benchmarks/micro/bench_smallint_calibration.py"),
    ("QUANTILE-CAL", "benchmarks/micro/bench_quantile_calibration.py"),
    ("MEANSUM-CAL", "benchmarks/micro/bench_meansum_calibration.py"),
    ("CHARVIEW-CAL", "benchmarks/micro/bench_charview_calibration.py"),
    ("BATCH4-CAL", "benchmarks/micro/bench_batch4_calibration.py"),
    ("BATCH5-CAL", "benchmarks/micro/bench_batch5_calibration.py"),
    ("PERCENTILE-CAL", "benchmarks/micro/bench_percentile_calibration.py"),
    ("BATCH6-CAL", "benchmarks/micro/bench_batch6_calibration.py"),
    ("NANPERCENTILE-CAL", "benchmarks/micro/bench_nanpercentile_calibration.py"),
    ("BATCH7-CAL", "benchmarks/micro/bench_batch7_calibration.py"),
    ("BATCH8-CAL", "benchmarks/micro/bench_batch8_calibration.py"),
    ("BATCH9-CAL", "benchmarks/micro/bench_batch9_calibration.py"),
    ("BATCH10-CAL", "benchmarks/micro/bench_batch10_calibration.py"),
    ("BATCH11-CAL", "benchmarks/micro/bench_batch11_calibration.py"),
    ("BATCH12-CAL", "benchmarks/micro/bench_batch12_calibration.py"),
]
PROOF = [("MVP-BASELINE", "benchmarks/micro/bench_enabled_vs_stock.py")]


def historical_reproducers() -> list[tuple[str, str]]:
    out = []
    for p in sorted((REPO_ROOT / "benchmarks" / "historical").glob("opp_*.py")):
        opp = "OPP-" + p.stem.split("_")[1]
        out.append((opp, str(p.relative_to(REPO_ROOT))))
    return out


def run(label: str, argv: list[str], log) -> tuple[bool, float]:
    print(f"\n=== {label}: {' '.join(argv)}", flush=True)
    log.write(f"\n=== {label}: {' '.join(argv)}\n")
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        argv, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
    proc.wait()
    dt = time.perf_counter() - t0
    print(f"=== {label}: exit {proc.returncode} in {dt:.0f}s", flush=True)
    return proc.returncode == 0, dt


def load_conditions(opp_id: str, fingerprint: str) -> str:
    import json

    path = REPO_ROOT / "benchmarks" / "results" / opp_id / f"{fingerprint}.json"
    if not path.exists():
        return "no result file"
    cond = json.loads(path.read_text(encoding="utf-8")).get("conditions", {})
    b, a = cond.get("cpu_busy_before"), cond.get("cpu_busy_after")
    fmt = lambda v: "?" if v is None else f"{v:.0%}"  # noqa: E731
    return f"{'CONTENDED' if cond.get('contended') else 'quiet':9} {fmt(b)} -> {fmt(a)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--historical", action="store_true", help="also run every historical reproducer")
    ap.add_argument("--require-quiet", action="store_true", help="exit 2 if foreign load exceeds the contended threshold")
    ap.add_argument("--smoke", action="store_true", help="pass --smoke to every suite (no evidence written)")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    fp = machine_fingerprint()
    busy = cpu_busy_fraction(1.0)
    print(f"[baseline] {fp['cpu']} | numpy {fp['numpy']} | fp {fp['fingerprint']}")
    print(f"[baseline] foreign CPU load now: {'?' if busy is None else f'{busy:.0%}'} (contended above {CONTENDED_LOAD:.0%})")
    if args.require_quiet and busy is not None and busy > CONTENDED_LOAD:
        print("[baseline] box is not quiet; refusing (--require-quiet). Try again later.")
        return 2

    SCRATCH.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = SCRATCH / f"baseline-{stamp}{'-smoke' if args.smoke else ''}.log"
    extra = ["--smoke"] if args.smoke else []
    suites = CALIBRATION + PROOF + (historical_reproducers() if args.historical else [])

    results: list[tuple[str, bool, float]] = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"baseline {stamp} fp {fp['fingerprint']} busy {busy}\n")
        for opp_id, script in suites:
            ok, dt = run(opp_id, [PY, script, *extra], log)
            results.append((opp_id, ok, dt))
        if not args.skip_tests:
            ok, dt = run("pytest", [PY, "-m", "pytest", "tests", "compatibility", "-q"], log)
            results.append(("pytest", ok, dt))
        if not args.smoke:
            run("calibrate_pyrallel", [PY, "lab/cli/calibrate_pyrallel.py"], log)
            run("calibrate_pyrallel_binary", [PY, "lab/cli/calibrate_pyrallel.py", "--suite", "PYRALLEL-BIN-CAL"], log)

    print("\n================ BASELINE SUMMARY ================")
    print(f"log: {log_path.relative_to(REPO_ROOT)}")
    for opp_id, ok, dt in results:
        cond = "" if (args.smoke or opp_id == "pytest") else load_conditions(opp_id, fp["fingerprint"])
        print(f"  {'PASS' if ok else 'FAIL':4}  {opp_id:14} {dt:6.0f}s  {cond}")
    if not args.smoke:
        print(
            "\nCompare the 'SUPPORTED proposal' blocks printed above with\n"
            "src/pyoverdrive/fastpaths/parallel_ufunc.py and parallel_binary.py;\n"
            "a QUIET run that proposes lower thresholds is the number to ship,\n"
            "a CONTENDED one is not."
        )
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
