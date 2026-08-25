"""Dyno: PyOverdrive's benchmark and verification harness.

Design goals (spec sections 7.4 and 8):

- setup is separated from timed work (callables passed in are the timed work);
- warmup runs precede measurement;
- per-sample work is auto-calibrated to a minimum duration so fast operations
  are not measured at timer-noise resolution;
- correctness is checked OUTSIDE the timed region, and a candidate that fails
  its check gets no speedup number at all. The check is called as
  ``check(candidate_result, baseline_result)`` - candidate FIRST - so name
  your parameters accordingly (two reproducers have already gotten this
  backwards in review);
- results carry the machine fingerprint and are written as JSON evidence under
  benchmarks/results/<OPP-ID>/<fingerprint>.json;
- a variant that raises is recorded as an error, not a crash of the suite.

Reproducer usage pattern (see benchmarks/historical/ for real examples):

    suite = BenchSuite("OPP-000001", "np.unique: hash vs sort path")
    data = make_seeded_data(...)          # setup, never timed
    suite.measure(
        case="int64_n1e6_card100",
        params={"dtype": "int64", "n": 1_000_000, "cardinality": 100},
        baseline=("numpy.unique", lambda: np.unique(data)),
        candidates={"sort_path": lambda: sort_unique(data)},
        check=np.array_equal,
    )
    suite.save()

Beware the classic loop-variable closure bug: build lambdas from loop state via
default arguments (``lambda d=data: np.unique(d)``) or a helper function.
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .fingerprint import machine_fingerprint
from .load import cpu_busy_fraction

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

# Foreign CPU load above this fraction at suite start or end marks the run as
# CONTENDED in the evidence and on the console. Single-thread baselines survive
# a loaded box (one free core suffices); multi-thread candidates do not.
CONTENDED_LOAD = 0.20

MIN_SAMPLE_NS = 5_000_000  # each timed sample aims at >= 5 ms of work
DEFAULT_SAMPLES = 15
DEFAULT_WARMUP = 3
MAX_LOOPS = 1_000_000


# timing statistics, rounded to SIG_FIGS on the way to disk
_TIMING_FIELDS = ("median_ns", "min_ns", "mean_ns", "mad_ns")
SIG_FIGS = 6


def _round_sig(v):
    """Six significant figures, scale-safe.

    Significant figures rather than decimal places on purpose: these values
    span nanoseconds to milliseconds, and a fixed number of decimals would
    be lossy at the bottom of that range and wasteful at the top.
    """
    if not isinstance(v, float) or v != v or v in (float("inf"), float("-inf")):
        return v
    return float(f"%.{SIG_FIGS}g" % v)


def compact_case(case: dict) -> dict:
    """The on-disk form of one case: rounded timings, no derived fields.

    Three things come off, none of them measurements:

    - `speedups`, a pure function of the medians beside it.
    - `role`, which said "baseline" for exactly the variant whose name is
      already in the case's own `baseline` field.
    - `correct`, but ONLY where it is True. It was true 7959 times out of
      7964 across the committed corpus, so it is stored as an exception
      rather than a constant. The five FALSE ones are real evidence - a
      candidate that failed its correctness check - and are written out
      explicitly. is_correct() reads it back with True as the default.
    """
    out = {k: v for k, v in case.items() if k != "speedups"}
    variants = out.get("variants")
    if isinstance(variants, dict):
        baseline = out.get("baseline")
        compacted = {}
        for name, stats in variants.items():
            if not isinstance(stats, dict):
                compacted[name] = stats
                continue
            slim = {}
            for k, v in stats.items():
                if k == "role":
                    continue  # derivable: name == case["baseline"]
                if k == "correct" and v is True:
                    continue  # the default; False is always written
                slim[k] = _round_sig(v) if k in _TIMING_FIELDS else v
            compacted[name] = slim
        out["variants"] = compacted
    return out


def role_of(case: dict, variant: str) -> str | None:
    """"baseline" for the case's baseline variant, else None - as stored."""
    return "baseline" if variant == case.get("baseline") else None


def is_correct(stats: dict) -> bool:
    """Correctness of one variant, with True as the on-disk default.

    Reading this with a bare .get("correct") would treat every passing
    variant as a failure, because passing variants no longer carry the key.
    """
    if "error" in stats:
        return False
    return bool(stats.get("correct", True))


def speedup_of(case: dict, variant: str):
    """The speedup a stored case implies, recomputed rather than stored.

    Returns None when the variant errored or failed its correctness check,
    which is exactly what the old stored `speedups` field recorded.
    """
    variants = case.get("variants") or {}
    base = variants.get(case.get("baseline")) or {}
    cand = variants.get(variant) or {}
    if not is_correct(cand) or "median_ns" not in cand:
        return None
    if not base.get("median_ns"):
        return None
    return base["median_ns"] / cand["median_ns"]


def _fmt_ns(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns / 1e9:.2f} s"
    if ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.2f} us"
    return f"{ns:.0f} ns"


def _calibrate_loops(fn) -> int:
    loops = 1
    while True:
        t0 = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        dt = time.perf_counter_ns() - t0
        if dt >= MIN_SAMPLE_NS or loops >= MAX_LOOPS:
            return loops
        grow = max(2, int(MIN_SAMPLE_NS / max(dt, 1)) + 1)
        loops = min(loops * grow, MAX_LOOPS)


def _time_variant(fn, samples: int, warmup: int):
    warmup = max(warmup, 1)
    result = None
    for _ in range(warmup):
        result = fn()
    loops = _calibrate_loops(fn)
    times = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(samples):
            t0 = time.perf_counter_ns()
            for _ in range(loops):
                fn()
            times.append((time.perf_counter_ns() - t0) / loops)
    finally:
        if gc_was_enabled:
            gc.enable()
    med = statistics.median(times)
    stats = {
        "median_ns": med,
        "min_ns": min(times),
        "mean_ns": statistics.fmean(times),
        "mad_ns": statistics.median(abs(t - med) for t in times),
        "samples": samples,
        "loops_per_sample": loops,
    }
    return result, stats


class BenchSuite:
    def __init__(self, opp_id: str, title: str):
        self.opp_id = opp_id
        self.title = title
        self.cases: list[dict] = []
        self.fingerprint = machine_fingerprint()
        self.conditions: dict = {"cpu_busy_before": cpu_busy_fraction()}
        print(f"[dyno] {opp_id}: {title}")
        self._warn_if_contended("before")
        print(
            f"[dyno] {self.fingerprint['cpu']} | numpy "
            f"{self.fingerprint['numpy']} | fp {self.fingerprint['fingerprint']}"
        )

    def measure(
        self,
        case: str,
        params: dict,
        baseline: tuple,
        candidates: dict,
        check=None,
        samples: int = DEFAULT_SAMPLES,
        warmup: int = DEFAULT_WARMUP,
    ) -> dict:
        base_name, base_fn = baseline
        try:
            base_result, base_stats = _time_variant(base_fn, samples, warmup)
        except Exception as exc:
            rec = {"case": case, "params": params, "error": f"baseline: {exc!r}"}
            self.cases.append(rec)
            print(f"  {case}: BASELINE ERROR {exc!r}")
            return rec

        variants = {base_name: dict(base_stats, role="baseline", correct=True)}
        speedups: dict[str, float | None] = {}
        for name, fn in candidates.items():
            try:
                cand_result, stats = _time_variant(fn, samples, warmup)
            except Exception as exc:
                variants[name] = {"error": repr(exc), "correct": False}
                speedups[name] = None
                continue
            ok = True
            if check is not None:
                try:
                    ok = bool(check(cand_result, base_result))
                except Exception as exc:
                    ok = False
                    stats["check_error"] = repr(exc)
            stats["correct"] = ok
            variants[name] = stats
            speedups[name] = (
                base_stats["median_ns"] / stats["median_ns"] if ok else None
            )

        rec = {
            "case": case,
            "params": params,
            "baseline": base_name,
            "variants": variants,
            "speedups": speedups,
        }
        self.cases.append(rec)
        self._print_case(rec, base_stats)
        return rec

    def _print_case(self, rec: dict, base_stats: dict) -> None:
        parts = [f"  {rec['case']}: {rec['baseline']} {_fmt_ns(base_stats['median_ns'])}"]
        for name, sp in rec["speedups"].items():
            v = rec["variants"][name]
            if "error" in v:
                parts.append(f"{name} ERROR")
            elif not v.get("correct", False):
                parts.append(f"{name} CORRECTNESS-FAIL")
            else:
                parts.append(f"{name} {_fmt_ns(v['median_ns'])} ({sp:.2f}x)")
        print(" | ".join(parts))

    def _warn_if_contended(self, when: str) -> None:
        busy = self.conditions.get(f"cpu_busy_{when}")
        if busy is None:
            return
        if busy > CONTENDED_LOAD:
            print(
                f"[dyno] WARNING: machine {busy:.0%} busy {when} this run "
                f"(foreign load > {CONTENDED_LOAD:.0%}); multi-thread candidates "
                f"are understated. Recorded as contended evidence."
            )
        else:
            print(f"[dyno] machine {busy:.0%} busy {when} this run")

    # --- on-disk form -----------------------------------------------------
    #
    # Evidence files were 60% of the repository by bytes: 100 files, 3.4 MB,
    # pretty-printed, storing 17-significant-digit floats for NANOSECOND
    # timings whose own MAD is tens of nanoseconds, plus a speedups block
    # that is a pure function of the medians beside it.
    #
    # Nothing is lost here. Whitespace carries no information. Six
    # significant figures is a thousand times finer than the measurement
    # noise these numbers describe, so the extra digits were recording the
    # float representation rather than the measurement. And a derived field
    # is not evidence - speedup_of() recomputes it exactly.

    def save(self) -> Path:
        out_dir = RESULTS_DIR / self.opp_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{self.fingerprint['fingerprint']}.json"
        self.conditions["cpu_busy_after"] = cpu_busy_fraction()
        self._warn_if_contended("after")
        loads = [v for v in self.conditions.values() if v is not None]
        self.conditions["contended"] = bool(loads) and max(loads) > CONTENDED_LOAD
        payload = {
            "opp_id": self.opp_id,
            "title": self.title,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": self.fingerprint,
            "conditions": self.conditions,
            "cases": [compact_case(c) for c in self.cases],
        }
        out.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"[dyno] wrote {out.relative_to(REPO_ROOT)}")
        return out
