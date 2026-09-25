"""A/B verdicts that refuse to call noise a speedup.

WHY this exists: every gate in this project used to compare a median ratio
against a bar (MIN_WIN, --min) and nothing else. A median of a noisy run can
land on either side of 1.3x by luck, and across a suite of cells the odd
lucky one is guaranteed. That is exactly how a false speedup gets claimed
and a fast path gets enabled on a machine where it does not pay.

So a comparison here has four honest outcomes, not two:

    faster        the whole confidence interval clears the bar, the
                  Holm-adjusted one-sided test agrees, and every baseline
                  run was slower than every candidate run
    slower        the same three conditions, the other way round against 1.0x
    no-win        conclusively below the bar, but not a proven regression
    inconclusive  anything else: fewer than 3 runs, overlapping per-run
                  ranges, a CI that straddles the bar, or a test that does
                  not survive the Holm step-down across the suite
    invalid       the samples themselves are unusable (non-finite, <= 0)

Callers treat only "faster" as permission to enable or claim, and only
"slower" as a regression; "inconclusive" means "measure more", never "pass".

Method (written fresh for PyOverdrive; ideas only from clash-verge-rev's
scripts/perf verdict rules and nodejs/node benchmark/compare.js): times
are compared on the log scale, so the speedup is a ratio of geometric means
and its interval is multiplicative. Each cell gets Welch's t-test with the
Welch-Satterthwaite degrees of freedom, and the one-sided p-values of a
whole suite are adjusted with the Holm-Bonferroni step-down so that a
many-cell sweep does not manufacture a win. Stdlib only: this module ships
in the wheel and is imported by calibration.
"""

from __future__ import annotations

import math
import statistics

ALPHA = 0.05
MIN_RUNS = 3

VERDICTS = ("faster", "slower", "no-win", "inconclusive", "invalid")


# --------------------------------------------------------------------------
# Student's t distribution, from the regularized incomplete beta function
# --------------------------------------------------------------------------

def _beta_cf(a: float, b: float, x: float) -> float:
    """Continued fraction for I_x(a, b), evaluated with Lentz's method."""
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
                    -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            step = c * d
            h *= step
        if abs(step - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    # The fraction converges fast only on one side of the mean; use symmetry.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with `df` (possibly fractional) freedom."""
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def t_ppf(q: float, df: float) -> float:
    """The t quantile for 0.5 < q < 1, by bisection on t_sf."""
    lo, hi = 0.0, 1.0
    while t_sf(hi, df) > 1.0 - q:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_sf(mid, df) > 1.0 - q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in input order."""
    m = len(pvalues)
    adjusted = [1.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=pvalues.__getitem__)):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# --------------------------------------------------------------------------
# One cell, then a suite
# --------------------------------------------------------------------------

def _usable(values) -> bool:
    return (isinstance(values, (list, tuple)) and len(values) > 0
            and all(type(v) in (int, float) and math.isfinite(v) and v > 0
                    for v in values))


def _welch(baseline: list[float], candidate: list[float], min_speedup: float,
           alpha: float) -> dict:
    """Log-scale Welch statistics for speedup = baseline / candidate."""
    la = [math.log(v) for v in baseline]
    lb = [math.log(v) for v in candidate]
    d = statistics.fmean(la) - statistics.fmean(lb)
    wa = statistics.variance(la) / len(la)
    wb = statistics.variance(lb) / len(lb)
    se = math.sqrt(wa + wb)
    bar = math.log(min_speedup)
    if se == 0.0:
        # Identical readings on each side: no spread to test against.
        return {"speedup": math.exp(d), "ci": (math.exp(d), math.exp(d)),
                "p_faster": 0.0 if d > bar else 1.0,
                "p_slower": 0.0 if d < 0.0 else 1.0, "df": math.inf}
    df = (wa + wb) ** 2 / (wa * wa / (len(la) - 1) + wb * wb / (len(lb) - 1))
    half = t_ppf(1.0 - alpha / 2.0, df) * se
    return {
        "speedup": math.exp(d),
        "ci": (math.exp(d - half), math.exp(d + half)),
        "p_faster": t_sf((d - bar) / se, df),
        "p_slower": t_sf(-d / se, df),
        "df": df,
    }


def compare_suite(cells: dict, *, min_speedup: float = 1.0,
                  alpha: float = ALPHA, min_runs: int = MIN_RUNS) -> dict:
    """Verdict per cell for {name: (baseline_times, candidate_times)}.

    Times are lower-is-better and each entry is one independent run. The
    Holm family is every cell with enough runs to be tested, so adding cells
    to a sweep makes each individual claim harder, as it should.
    """
    if not (min_speedup > 0 and math.isfinite(min_speedup)):
        raise ValueError("min_speedup must be a positive finite ratio")
    out: dict[str, dict] = {}
    tested: list[str] = []
    for name, (baseline, candidate) in cells.items():
        if not (_usable(baseline) and _usable(candidate)):
            out[name] = {"verdict": "invalid",
                         "reason": "non-finite, non-positive or missing times"}
            continue
        runs = (len(baseline), len(candidate))
        if min(runs) < max(min_runs, 2):
            out[name] = {"verdict": "inconclusive", "runs": runs,
                         "reason": f"fewer than {min_runs} independent runs"}
            continue
        stats = _welch(list(baseline), list(candidate), min_speedup, alpha)
        stats.update(runs=runs,
                     separated_faster=min(baseline) > max(candidate),
                     separated_slower=max(baseline) < min(candidate))
        out[name] = stats
        tested.append(name)

    adj_fast = holm([out[n]["p_faster"] for n in tested])
    adj_slow = holm([out[n]["p_slower"] for n in tested])
    for name, pf, ps in zip(tested, adj_fast, adj_slow):
        cell = out[name]
        lo, hi = cell["ci"]
        cell["p_faster"], cell["p_slower"] = pf, ps
        if pf < alpha and lo >= min_speedup and cell["separated_faster"]:
            cell["verdict"], cell["reason"] = "faster", "CI clears the bar"
        elif ps < alpha and hi < 1.0 and cell["separated_slower"]:
            cell["verdict"], cell["reason"] = "slower", "CI lies below 1.0x"
        elif hi < min_speedup:
            cell["verdict"], cell["reason"] = "no-win", "CI lies below the bar"
        elif not (cell["separated_faster"] or cell["separated_slower"]):
            cell["verdict"], cell["reason"] = "inconclusive", "per-run ranges overlap"
        elif lo < min_speedup <= hi:
            cell["verdict"] = "inconclusive"
            cell["reason"] = f"CI {lo:.3f}-{hi:.3f}x straddles {min_speedup:g}x"
        else:
            cell["verdict"] = "inconclusive"
            cell["reason"] = "not significant after Holm correction"
    return out


def compare(baseline: list[float], candidate: list[float], **kwargs) -> dict:
    """compare_suite for a single cell (a Holm family of one)."""
    return compare_suite({"cell": (baseline, candidate)}, **kwargs)["cell"]
