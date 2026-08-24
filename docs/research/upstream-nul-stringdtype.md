# Upstream report draft: StringDType NUL-character bugs in numpy

Status: FILED 2026-08-24 as numpy/numpy#32414 (defects 1 and 2, one
issue); the searchsorted item landed as a confirmation comment on
numpy/numpy#29727 (its exact class was already tracked there). Everything below reproduces on numpy 2.4.5
and 2.5.2 (Windows AMD64, MSVC builds; the isin miss also reproduces on
Linux x86-64, python:3.13-slim, numpy 2.5.2). Related upstream context:
numpy/numpy#32161 (StringDType isin performance) is where PyOverdrive
found these while building a guarded fast path (OPP-000023).

Two distinct defects, one root smell (C NUL-termination leaking into a
length-carrying string dtype):

## 1. np.isin misses strings made only of NUL characters

```python
import numpy as np

dt = np.dtypes.StringDType()
a = np.array(["\x00", "\x00\x00", "plain\x00mid", "x", ""], dtype=dt)
np.isin(a, a)
# numpy 2.4.5 and 2.5.2, Windows and Linux:
#   array([False, False,  True,  True,  True])
# expected: all True (every element is trivially in the array itself)
```

Strings that CONTAIN a NUL among other characters match fine; strings
that consist ONLY of NULs ("\x00", "\x00\x00", ...) are never found,
even by themselves. Elementwise ``==`` against an np StringDType scalar
gives the correct answer on 2.5.2, so equality and isin disagree with
each other there.

What changed between 2.4 and 2.5: ``np.strings.str_len`` used to report
0 for pure-NUL strings (2.4.5) and now reports the true length (2.5.2),
but the isin miss is unchanged. So the class of machinery is being
fixed piecemeal while isin still carries the old behavior.

## 2. A Python "\x00" scalar argument C-truncates inside string ufuncs

```python
probe = np.array(["\x00", "\x00\x00", "a\x00b", "", "x"], dtype=dt)
# correct count(probe, "\x00") would be [1, 2, 1, 0, 0]

np.strings.count(probe, "\x00")                          # Python scalar
# 2.5.2: array([2, 3, 4, 1, 2])   <- exactly count(probe, "") = str_len + 1
# 2.4.5: array([1, 1, 4, 1, 2])   <- also count(probe, ""), under 2.4's
#                                    NUL-blind str_len ([0, 0, 3, 0, 1])

np.strings.count(probe, np.array("\x00", dtype=dt))      # np scalar
# 2.5.2: array([1, 2, 1, 0, 0])   <- correct
# 2.4.5: array([1, 1, 4, 1, 2])   <- truncated here too: on 2.4 BOTH
#                                    spellings degrade to the "" needle

np.strings.replace(probe, "\x00", "N")                   # Python scalar
# 2.5.2: ['N\x00N', 'N\x00N\x00N', 'NaN\x00NbN', 'N', 'NxN']
#        <- replace(probe, "", "N"): inserts everywhere, replaces nothing
```

The Python-str needle is truncated at the first NUL somewhere on the
conversion path, so ``count``/``replace``/``strip`` silently operate on
the empty string instead. The same values passed as a 0-d StringDType
array survive intact on 2.5.2 - the np-scalar path was evidently fixed
between 2.4 and 2.5 while the Python-scalar path kept the truncation,
so the two spellings of the same call now disagree. A user has no
reason to expect them to differ. (All outputs above re-measured
2026-08-24 on 2.4.5 Windows AMD64 and 2.5.2 Windows Intel; the isin
miss also reproduces on Linux x86-64.)

## Why it matters beyond aesthetics

NUL-bearing strings are exactly what StringDType exists to make legal
(fixed-width U/S dtypes cannot round-trip trailing NULs). Any dedup,
membership, or join keyed on such data silently drops the pure-NUL
class, with no error and no warning - the failure mode is wrong answers
in data cleaning, not crashes.

## 3. (Separate issue, confirm before filing) uint64 searchsorted with a
## Python int key returns a wrong index

Found while building OPP-000039 (2026-08-24, numpy 2.5.2, Windows
AMD64): np.searchsorted on a uint64 array with an in-range Python int
key returned a DIFFERENT index than the same key cast to np.uint64 -
consistent with the float64-promotion precision loss already tracked
upstream as numpy/numpy#29727 (spun off #29719). Our battery cell
"searchsorted_u64_n100000_pyint" (benchmarks/results/OPP-000039/)
recorded the divergence live. If #29727 already covers it, add the
repro as a comment there rather than filing anew.

## Suggested repro environment lines for the issue

- numpy 2.4.5 / Python 3.14, Windows 11 AMD64 (MSVC build)
- numpy 2.5.2 / Python 3.13, Windows 11 AMD64 and Debian Linux x86-64
