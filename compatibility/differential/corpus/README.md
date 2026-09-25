# Differential fuzz corpus

Every file here is a minimized input on which a patched PyOverdrive call once
diverged from stock NumPy. `fuzz.py` (one directory up) finds it, shrinks it and
writes it as `<fast path>/<content hash>.npz`; `test_fuzzed_differential.py`
replays every file on each plain test run, so a fixed bug stays fixed.

Each `.npz` holds, per input array `x`:

- `array:x` - the logical values (always C-contiguous on disk);
- `layout:x` - a 2 x ndim int64 array: the axis order from outermost stored
  axis inward, then the per-axis slicing step (negative = reversed view).
  Replay rebuilds the exact strided view from these, because the layout is
  often what made the case fail;

plus `meta`, a JSON string with the fast path, the operation, the scalar
parameters, the seed and sample index that produced it, and the mismatch seen.

Commit a new case together with the fix it proves. Delete one only when its
fast path is withdrawn (the replay test fails on a case whose path has no spec).
