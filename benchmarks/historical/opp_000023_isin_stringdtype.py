"""OPP-000023: np.isin on StringDType falls back to the slow object-array
path; U-cast sorting route and hash-set membership route as candidates.

numpy/numpy#32161 (ngoldbaum, numpy MEMBER, 2026-07-31): np.isin on a
StringDType ('T') array takes the hasobject fallback in the _isin helper, a
per-test-element comparison loop over the full element array (O(n*m) kernel
invocations), instead of the sorting path fixed-width unicode reaches. The
reporter's single measured point: a = np.array(['abcd'*5, 'defg'*5,
'abhoj'*6] * 10**4, dtype='T') (30000 elements, 3 distinct strings),
np.isin(a, a) at 2.9416s vs np.isin(a.astype('U30'), a.astype('U30')) at
0.0205s -- a DERIVED 143.53x, with both astype('U30') casts inside the timed
lambda, so the sort-path advantage net of conversion is understated if
anything.

This reproducer measures:

  - baseline: stock np.isin(element, test_elements) on the 'T' arrays,
    exactly as a user writes it, default keywords.
  - candidate "ucast_isin_u30" (claim case only): the reporter's literal
    route, np.isin(a.astype('U30'), a.astype('U30')), casts inside the
    timed call.
  - candidate "ucast_isin" (all other cases): the same sorting-path route
    with the U width computed from the data inside the timed call
    (np.strings.str_len(...).max()), as OPP-000023.md requires ("the
    U-cast width is computed from the data rather than assumed"), so the
    measured speedup pays for both the length scan and the casts.
  - candidate "hashset_membership": the reporter's second route, a
    hash-based membership pass -- here a plain Python set of the test
    elements probed per element via np.fromiter. Pure stdlib + numpy
    construction, standing in for the "hash-based unique infrastructure"
    route of the issue body and draft PR numpy/numpy#32217.

Neither candidate imports pyoverdrive. The ucast candidates DO call
np.isin, but only on fixed-width unicode ('U#') arrays: the fast-path
predicate this record proposes intercepts np.isin only when an argument is
StringDType, which a U array never is, so a patched np.isin cannot recurse
through these candidates (same reasoning as OPP-000013's skip_to_quantile
calling np.quantile). The hashset candidate never calls np.isin at all.

Correctness: isin is a string-membership test and every route computes the
same membership over the same strings, so the check is exact full
boolean-array equality (np.array_equal), no tolerance. Dyno calls
check(candidate_result, baseline_result), candidate first; array_equal is
symmetric so the order cannot bite here.

Regime and shrinks: the claim point (n=30000, cardinality 3, self-isin) is
reproduced EXACTLY, but at samples=5 with warmup=1 because its baseline
runs ~3s per call on the reporter's laptop (7 baseline invocations, about
20s, is the bulk of this battery). The record's suggested wider sweep
(element counts to 1e6, per OPP-000023.md item 2) is NOT run at full size:
the O(n*m) fallback makes a 1e6 self-isin take on the order of hours, so
the cardinality sweep (3, 100, n distinct) runs at n=3000 where the
baseline is ~30ms, the disjoint small-test-set case (the common real-world
isin shape) keeps n=30000 elements against m=100 test strings (only m
full-array comparisons, so it stays cheap), and the long-string outlier
case (one 2000-char string, exposing the U-cast width hazard) runs at
n=3000. This keeps the full non-smoke battery around 25-30s, within the
~90s house budget.

Not measured here: na_object configurations (string, nan-like), invert=,
assume_unique=, and kind= keyword paths. Those are routing-edge
correctness probes for a real dispatch (nan-like na_object must fall back
to stock), not part of the performance claim; the claim regime has no
missing data and default keywords.

Version boundary: draft PR numpy/numpy#32217 ("ENH: speed up StringDType
isin") fixes this upstream. The Dyno fingerprint records the numpy version
this ran on; if that numpy contains the merged fix, the honest outcome is
a collapsed or absent gap (not_reproduced), not a broken reproducer.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 32161
SMOKE = "--smoke" in sys.argv


def make_unique_strings(rng, count, min_len=6, max_len=18):
    """count distinct lowercase strings; an index suffix forces uniqueness
    without a rejection loop. Setup only, never timed."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    out = []
    for i in range(count):
        ln = int(rng.integers(min_len, max_len + 1))
        body = "".join(letters[j] for j in rng.integers(0, 26, size=ln))
        out.append(f"{body}{i:05d}")
    return out


def from_pool(rng, pool, n):
    """n-element 'T' array drawn uniformly from pool. Setup only."""
    idx = rng.integers(0, len(pool), size=n)
    return np.array([pool[int(i)] for i in idx], dtype="T")


def ucast_isin(element, test_elements):
    """Sorting-path route: cast both arguments to fixed-width unicode wide
    enough for the longest string present, then stock np.isin (which takes
    the sort path for 'U'). Width scan and casts are inside the timed call.
    Only ever handed U arrays to np.isin, so a StringDType-predicated
    fast-path patch cannot recurse through it (see module docstring)."""
    w = max(
        int(np.strings.str_len(element).max()),
        int(np.strings.str_len(test_elements).max()),
        1,
    )
    return np.isin(element.astype(f"U{w}"), test_elements.astype(f"U{w}"))


def hashset_isin(element, test_elements):
    """Hash-set membership route: Python set of the test elements, probed
    once per element. Never calls np.isin."""
    lookup = set(test_elements.tolist())
    return np.fromiter(
        (s in lookup for s in element.tolist()), dtype=bool, count=element.size
    )


if SMOKE:
    CLAIM_REPS = 100      # 300-element claim-shaped array
    SWEEP_N = 210
    SWEEP_CARDS = [3, 20, 210]
    DISJOINT_N, DISJOINT_M = 300, 20
    LONG_REPS = 100
    OUTLIER_LEN = 200
else:
    CLAIM_REPS = 10**4    # the reporter's exact 30000-element array
    SWEEP_N = 3000
    SWEEP_CARDS = [3, 100, 3000]
    DISJOINT_N, DISJOINT_M = 30000, 100
    LONG_REPS = 1000
    OUTLIER_LEN = 2000

suite = BenchSuite(
    "OPP-000023",
    "np.isin on StringDType: hasobject fallback vs U-cast sort / hash-set routes",
)

rng = np.random.default_rng(SEED)

# --- Case 1: the reporter's exact claim point (shrunk only in smoke) ------
a_claim = np.array(["abcd" * 5, "defg" * 5, "abhoj" * 6] * CLAIM_REPS, dtype="T")
suite.measure(
    case=f"claim_self_isin_n{a_claim.size}_card3",
    params={
        "dtype": "StringDType",
        "n": int(a_claim.size),
        "cardinality": 3,
        "relation": "self",
        "str_lens": [20, 20, 30],
    },
    baseline=("numpy.isin", lambda a=a_claim: np.isin(a, a)),
    candidates={
        # The literal route from the issue body, fixed U30 width.
        "ucast_isin_u30": lambda a=a_claim: np.isin(
            a.astype("U30"), a.astype("U30")
        ),
        "hashset_membership": lambda a=a_claim: hashset_isin(a, a),
    },
    check=np.array_equal,
    samples=3 if SMOKE else 5,
    warmup=1,  # baseline is ~3s/call at full size; see module docstring
)

# --- Case 2: cardinality sweep at reduced n (self-isin) -------------------
for card in SWEEP_CARDS:
    pool = make_unique_strings(rng, card)
    if card == SWEEP_N:
        perm = rng.permutation(SWEEP_N)
        a = np.array([pool[int(i)] for i in perm], dtype="T")
    else:
        a = from_pool(rng, pool, SWEEP_N)
    suite.measure(
        case=f"sweep_self_isin_n{SWEEP_N}_card{card}",
        params={
            "dtype": "StringDType",
            "n": SWEEP_N,
            "cardinality": card,
            "relation": "self",
            "str_lens": "6-18 plus 5-digit suffix",
        },
        baseline=("numpy.isin", lambda a=a: np.isin(a, a)),
        candidates={
            "ucast_isin": lambda a=a: ucast_isin(a, a),
            "hashset_membership": lambda a=a: hashset_isin(a, a),
        },
        check=np.array_equal,
        samples=3 if SMOKE else 7,
    )

# --- Case 3: disjoint small test set vs large element array ---------------
pool = make_unique_strings(rng, DISJOINT_M + DISJOINT_M // 2)
elements = from_pool(rng, pool[: DISJOINT_M + DISJOINT_M // 2], DISJOINT_N)
half = DISJOINT_M // 2
test_strs = pool[:half] + [s + "_absent" for s in pool[half : DISJOINT_M]]
test_arr = np.array(test_strs, dtype="T")
suite.measure(
    case=f"disjoint_n{DISJOINT_N}_m{DISJOINT_M}",
    params={
        "dtype": "StringDType",
        "n": DISJOINT_N,
        "m_test": DISJOINT_M,
        "cardinality": len(pool),
        "relation": "disjoint, half of test set present",
        "str_lens": "6-18 plus 5-digit suffix",
    },
    baseline=(
        "numpy.isin",
        lambda e=elements, t=test_arr: np.isin(e, t),
    ),
    candidates={
        "ucast_isin": lambda e=elements, t=test_arr: ucast_isin(e, t),
        "hashset_membership": lambda e=elements, t=test_arr: hashset_isin(e, t),
    },
    check=np.array_equal,
    samples=3 if SMOKE else 7,
)

# --- Case 4: long-string outlier exposing the U-cast width hazard ---------
a_long = np.array(
    ["z" * OUTLIER_LEN, "defg" * 5, "abhoj" * 6] * LONG_REPS, dtype="T"
)
suite.measure(
    case=f"longstring_self_isin_n{a_long.size}_card3_maxlen{OUTLIER_LEN}",
    params={
        "dtype": "StringDType",
        "n": int(a_long.size),
        "cardinality": 3,
        "relation": "self",
        "str_lens": [OUTLIER_LEN, 20, 30],
    },
    baseline=("numpy.isin", lambda a=a_long: np.isin(a, a)),
    candidates={
        "ucast_isin": lambda a=a_long: ucast_isin(a, a),
        "hashset_membership": lambda a=a_long: hashset_isin(a, a),
    },
    check=np.array_equal,
    samples=3 if SMOKE else 5,
)

if not SMOKE:
    suite.save()
