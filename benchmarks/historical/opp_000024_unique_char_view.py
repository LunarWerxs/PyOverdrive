"""OPP-000024: integer-view fast path for np.unique (and np.sort) on
single-character U1/S1 string arrays.

numpy/numpy#24821 reports that np.unique on a 10,000-element U1 array (10
distinct characters, 'ASDFGHJKLZ' repeated) runs about 3x slower than the
same call routed through an integer reinterpretation of the same buffer:
np.unique(x.view(np.int32)).view(x.dtype). The reporter's own %timeit pair
is 204 us vs 66.7 us at 10,000 elements (a derived 3.06x; the thread never
states the ratio), and 2.17 us vs 2.53 us at 10 elements, where the view
path LOSES (0.86x). So the claim carries its own anti-regime and a
crossover somewhere between 10 and 10,000 elements that the thread does
not locate. The mechanism, per an ngoldbaum mailing-list excerpt quoted in
the issue body, is that numpy's string quicksort is 2-3x slower than its
int quicksort on the same data, and np.unique sorts internally.

What this reproducer measures:

  - baseline: stock np.unique(x) (and, for the sort-only pair, np.sort(x))
    exactly as a user writes it, on U1 and S1 arrays.
  - candidates: the pure-numpy view routes named in the record. U1 views
    as int32 (numpy orders U1 by its single UCS4 code unit, and every
    valid codepoint <= 0x10FFFF is non-negative in int32, so the view is
    a strictly monotone bijection). S1 views as uint8, NOT int8 (S1
    compares as unsigned bytes; a signed int8 view would misorder bytes
    >= 0x80). Because the mapping is order- and equality-preserving,
    values, index, inverse, and counts are identical by construction, so
    every correctness check here is EXACT equality on every returned
    array, dtype included. Nothing weaker.

Case axes, following the record's reproducer spec:

  1. U1 size sweep 10 .. 1,000,000 on the issue's own 10-character
     alphabet, to locate the crossover the thread brackets only between
     10 and 10,000 and to see how the 3.06x behaves at sizes the reporter
     never ran.
  2. S1/uint8 sweep at 10, 10,000, and 1,000,000 (the reporter claims the
     S1 pattern matches U1 but posts numbers only for U1).
  3. return_counts=True and a return_index+return_inverse case at 10,000
     elements, to prove passthrough correctness (element positions are
     unaffected by the reinterpretation).
  4. Alphabet axis at 10,000 elements: 94-character printable ASCII, a
     10-character Greek set (codepoints > 0x7F for U1), and an S1
     alphabet spanning 0x00-0xFF including bytes >= 0x80, which is the
     int8-vs-uint8 trap detector (the candidate's uint8 view must match
     stock output exactly even on high bytes; the 0x00 byte also
     exercises the empty-string edge, which sorts first under both
     representations).
  5. A separate np.sort-only U1 pair at 10,000 and 1,000,000 elements to
     isolate ngoldbaum's "two or three times slower" sort-kernel figure
     from the unique-level end-to-end ratio.

Not measured here (outside the record's regime): the axis= path,
byte-swapped (non-native-endian) dtypes, and multi-character strings
(U>1/S>1) -- a real fast path must leave all of those on the stock route,
and the view identity does not even hold for them.

House rules: this script never imports pyoverdrive. The candidates do call
np.unique / np.sort, but only on integer views (int32/uint8), never on a
U1/S1 array -- so a patched fast path whose predicate matches only
single-character string dtypes cannot recurse through them, and within
this script they are plain stock calls regardless.

Sizes are run in FULL, no shrink: np.unique and np.sort at 1,000,000
elements are tens-of-milliseconds calls, so the whole non-smoke battery
sits comfortably inside the ~90s budget.

The issue was filed 2023-09-27 against an unstated numpy version; numpy's
string sort may have changed since, so a shrunken or vanished speedup on
current numpy is a legitimate outcome of this reproducer, as is the
expected small-n loss (the view path paying two extra view calls it
cannot amortize).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lab.dyno import BenchSuite

SEED = 24821
SMOKE = "--smoke" in sys.argv

# Alphabets. Greek runs 0x3B1..0x3BA (alpha..kappa), all above 0x7F, to
# exercise non-Latin codepoints for U1. The S1 high-byte alphabet spans
# 0x00-0xFF including six bytes >= 0x80: exact equality against stock
# np.unique on it fails for a signed int8 view and passes for uint8.
ALPHA10_U = np.array(list("ASDFGHJKLZ"), dtype="U1")
ALPHA10_S = np.frombuffer(b"ASDFGHJKLZ", dtype="S1")
ASCII94_U = np.array([chr(c) for c in range(33, 127)], dtype="U1")
GREEK10_U = np.array([chr(0x3B1 + i) for i in range(10)], dtype="U1")
HIBYTE10_S = np.frombuffer(
    bytes([0x00, 0x41, 0x5A, 0x7F, 0x80, 0x9C, 0xB5, 0xC8, 0xE0, 0xFF]),
    dtype="S1",
)


def draw(rng, alphabet, n):
    """n elements drawn uniformly from the alphabet (setup, never timed)."""
    return alphabet[rng.integers(0, len(alphabet), size=n)]


def unique_view_u1(x):
    """The issue's own candidate: U1 buffer reinterpreted as int32."""
    return np.unique(x.view(np.int32)).view(x.dtype)


def unique_view_s1(x):
    """S1 equivalent: uint8, never int8 (S1 orders as unsigned bytes)."""
    return np.unique(x.view(np.uint8)).view(x.dtype)


def unique_view_u1_counts(x):
    vals, counts = np.unique(x.view(np.int32), return_counts=True)
    return vals.view(x.dtype), counts


def unique_view_u1_index_inverse(x):
    vals, idx, inv = np.unique(
        x.view(np.int32), return_index=True, return_inverse=True
    )
    return vals.view(x.dtype), idx, inv


def sort_view_u1(x):
    return np.sort(x.view(np.int32)).view(x.dtype)


def exact_equal(cand, base):
    """Exact equality on every returned array, dtype included. The view is
    a monotone bijection, so anything short of bit-identical output is a
    correctness failure, not noise; no tolerance is appropriate here."""
    if isinstance(base, tuple):
        return (
            isinstance(cand, tuple)
            and len(cand) == len(base)
            and cand[0].dtype == base[0].dtype
            and all(np.array_equal(c, b) for c, b in zip(cand, base))
        )
    return cand.dtype == base.dtype and np.array_equal(cand, base)


if SMOKE:
    U1_SWEEP_SIZES = [10, 1_000]
    S1_SWEEP_SIZES = []          # the S1 path is still covered by the
    ALPHABET_CASES = ["hibyte"]  # high-byte trap-detector case below
    FLAG_N = 1_000
    SORT_SIZES = [1_000]
else:
    U1_SWEEP_SIZES = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    S1_SWEEP_SIZES = [10, 10_000, 1_000_000]
    ALPHABET_CASES = ["ascii94", "greek10", "hibyte"]
    FLAG_N = 10_000
    SORT_SIZES = [10_000, 1_000_000]


def samples_for(n):
    if SMOKE:
        return 3
    if n >= 1_000_000:
        return 5
    if n >= 100_000:
        return 7
    return 11


suite = BenchSuite(
    "OPP-000024",
    "np.unique on U1/S1 single-char arrays: integer-view route vs stock",
)

rng = np.random.default_rng(SEED)

# --- 1. U1 crossover sweep, the issue's own alphabet -----------------------
for n in U1_SWEEP_SIZES:
    x = draw(rng, ALPHA10_U, n)
    suite.measure(
        case=f"unique_U1_alpha10_n{n}",
        params={"dtype": "U1", "n": n, "alphabet": 10, "returns": "values"},
        baseline=("numpy.unique", lambda x=x: np.unique(x)),
        candidates={"unique_int32_view": lambda x=x: unique_view_u1(x)},
        check=exact_equal,
        samples=samples_for(n),
    )

# --- 2. S1/uint8 sweep -----------------------------------------------------
for n in S1_SWEEP_SIZES:
    x = draw(rng, ALPHA10_S, n)
    suite.measure(
        case=f"unique_S1_alpha10_n{n}",
        params={"dtype": "S1", "n": n, "alphabet": 10, "returns": "values"},
        baseline=("numpy.unique", lambda x=x: np.unique(x)),
        candidates={"unique_uint8_view": lambda x=x: unique_view_s1(x)},
        check=exact_equal,
        samples=samples_for(n),
    )

# --- 3. Optional-returns passthrough at the issue's evidenced size ---------
x_flag = draw(rng, ALPHA10_U, FLAG_N)
suite.measure(
    case=f"unique_U1_alpha10_n{FLAG_N}_counts",
    params={"dtype": "U1", "n": FLAG_N, "alphabet": 10, "returns": "values+counts"},
    baseline=(
        "numpy.unique",
        lambda x=x_flag: np.unique(x, return_counts=True),
    ),
    candidates={"unique_int32_view": lambda x=x_flag: unique_view_u1_counts(x)},
    check=exact_equal,
    samples=samples_for(FLAG_N),
)
suite.measure(
    case=f"unique_U1_alpha10_n{FLAG_N}_index_inverse",
    params={
        "dtype": "U1",
        "n": FLAG_N,
        "alphabet": 10,
        "returns": "values+index+inverse",
    },
    baseline=(
        "numpy.unique",
        lambda x=x_flag: np.unique(x, return_index=True, return_inverse=True),
    ),
    candidates={
        "unique_int32_view": lambda x=x_flag: unique_view_u1_index_inverse(x)
    },
    check=exact_equal,
    samples=samples_for(FLAG_N),
)

# --- 4. Alphabet axis ------------------------------------------------------
ALPHABETS = {
    "ascii94": ("U1", ASCII94_U, unique_view_u1, "unique_int32_view", 94),
    "greek10": ("U1", GREEK10_U, unique_view_u1, "unique_int32_view", 10),
    "hibyte": ("S1", HIBYTE10_S, unique_view_s1, "unique_uint8_view", 10),
}
for key in ALPHABET_CASES:
    dtype, alphabet, cand_fn, cand_name, alpha_size = ALPHABETS[key]
    x = draw(rng, alphabet, FLAG_N)
    suite.measure(
        case=f"unique_{dtype}_{key}_n{FLAG_N}",
        params={
            "dtype": dtype,
            "n": FLAG_N,
            "alphabet": alpha_size,
            "alphabet_kind": key,
            "returns": "values",
        },
        baseline=("numpy.unique", lambda x=x: np.unique(x)),
        candidates={cand_name: lambda x=x, f=cand_fn: f(x)},
        check=exact_equal,
        samples=samples_for(FLAG_N),
    )

# --- 5. Sort-only pair: isolate the sort-kernel gap ------------------------
for n in SORT_SIZES:
    x = draw(rng, ALPHA10_U, n)
    suite.measure(
        case=f"sort_U1_alpha10_n{n}",
        params={"dtype": "U1", "n": n, "alphabet": 10, "op": "sort"},
        baseline=("numpy.sort", lambda x=x: np.sort(x)),
        candidates={"sort_int32_view": lambda x=x: sort_view_u1(x)},
        check=exact_equal,
        samples=samples_for(n),
    )

if not SMOKE:
    suite.save()
