# Upstream: np.linalg.svd with vectors never returns on an infinite diagonal entry (Linux)

STATUS: confirmed on Linux x86-64, minimal repro in hand, not yet filed.
Found 2026-08-25 while diagnosing why batch 12's public CI wedged.

## The bug

On Linux, `np.linalg.svd` with `compute_uv=True` **never returns** when the
input matrix is 3x3 or larger and carries an infinite value on its
DIAGONAL. The thread does not block - it spins, pegged in state `R`, with
no other thread involved. `np.linalg.pinv` inherits it, because pinv calls
`svd(a, full_matrices=False)` internally.

The same calls on Windows return in well under a millisecond.

```python
import numpy as np
m = np.eye(3)
m[0, 0] = np.inf

np.linalg.svd(m, compute_uv=False)   # [nan nan nan]   - fine everywhere
np.linalg.norm(m, ord=2)             # nan             - fine everywhere
np.linalg.svd(m)                     # NEVER RETURNS on Linux
np.linalg.svd(m, full_matrices=False)  # NEVER RETURNS on Linux
np.linalg.pinv(m)                    # NEVER RETURNS on Linux
```

## Measured boundary

Every row below was run as its own process under `timeout -s KILL 45` on
`ubuntu-latest`, numpy 2.5.2, CPython 3.13.15, x86-64, 4 cores. "wedged"
means killed by the timeout, i.e. it had not returned.

| case | Linux |
| --- | --- |
| `svd(compute_uv=False)`, 3x3, inf on diagonal | returns `[nan nan nan]` |
| `norm(ord=2)`, 3x3, inf on diagonal | returns `nan` |
| `svd(full_matrices=False)`, 3x3, inf on diagonal | **wedged** |
| `svd()` (default `compute_uv=True`), 3x3, inf on diagonal | **wedged** |
| `pinv`, 3x3, `+inf` on diagonal | **wedged** |
| `pinv`, 3x3, `-inf` on diagonal | **wedged** |
| `pinv`, 4x4, inf on diagonal | **wedged** |
| `pinv`, **2x2**, inf on diagonal | returns |
| `pinv`, 3x3, inf **off** the diagonal | returns |
| `pinv`, 3x3, **nan** instead of inf | raises `LinAlgError: SVD did not converge` |
| `eigvals`, 3x3, inf on diagonal | raises |
| batched `svd(compute_uv=False)`, (100,3,3) with one inf | returns |
| batched `norm(ord=2)`, (100,3,3) with one inf | returns |
| batched `pinv`, (100,3,3) with one inf | **wedged** |

So the trigger is narrow and consistent: **the vector-computing SVD
(`compute_uv=True`, LAPACK `jobz='S'`/`'A'`), dimension >= 3, an infinity
on the diagonal.** The values-only path (`jobz='N'`) is unaffected at every
size, which is why `svd(compute_uv=False)` and `norm(ord=2)` stay healthy -
`norm` reaches stock's own `_multi_svd_norm`, which passes
`compute_uv=False`.

NaN behaves correctly (prompt `LinAlgError`), so this is specifically about
infinities, not non-finite input in general.

## Version and platform spread

Not a single-version regression. The wedge reproduced on **every** Linux
leg tried - CPython 3.12, 3.13 and 3.14, against numpy 2.0.2, 2.4.5 and
2.5.2 (the PyPI manylinux wheels, so OpenBLAS) - and on none of the Windows
legs (numpy 2.4.5 locally, numpy latest on `windows-latest`). That points
at the LAPACK build rather than at numpy's Python layer: the
divide-and-conquer driver behind `jobz='S'` appears to iterate without a
convergence bail-out once an infinity reaches the bidiagonal form, where
the `jobz='N'` path returns `info > 0` and numpy raises.

## How it was found

Batch 12's CI showed four `ubuntu-latest` jobs sitting "in progress" for
over an hour while `windows-latest` passed the same commit in 41 seconds.
That state is indistinguishable from a runner queue and was first read as
one. It is not: every one of seven stuck jobs, across two independent runs
and three numpy versions, had stopped after exactly the same number of
tests.

Two instruments settled it. `timeout-minutes` on the job turns a 6-hour
ceiling into a 20-minute failure. The stack came from running pytest in the
background, waiting 60 seconds, and then dumping - `py-spy` was refused
ptrace on the runner, but `/proc/<pid>/task/*/stat` showed the main thread
in `R` with every other thread idle in `futex_do_wait` (so: a spin, not a
deadlock), and `kill -ABRT` into a `-X faulthandler` process printed the
exact Python frames:

```
File "numpy/linalg/_linalg.py", line 1840 in svd
File "numpy/linalg/_linalg.py", line 2256 in pinv
File ".../test_svd_small_batch_differential.py", line 106 in _call
```

Worth recording: pytest's own `faulthandler_timeout` did **not** fire on
this hang even though the ini was set and the same setting fires correctly
on a `time.sleep` test locally. Do not rely on it alone against a C-level
spin; the out-of-process `SIGABRT` dump is what produced the answer.

Also worth recording: with output going through a pipe rather than a tty,
the progress dots lag the true position. The buffered runs pointed at a
test roughly fifty places before the real one, and re-running with
`python -u` moved the answer from "a well-conditioned random batch" to
"the infinity refusal test". The first reading sent this investigation down
a wrong path for a while - unbuffer before trusting a progress counter.

## What PyOverdrive does about it

Nothing to the shipped behaviour, deliberately. PyOverdrive already refuses
any non-finite batch and hands it to stock, so a user sees exactly what
stock numpy does on their platform - hang for hang, NaN for NaN. That is
parity, which is the contract, and no regression is introduced.

The only change is in the tests: the differential and property suites can
no longer *execute* stock `pinv` on an infinite batch, because doing so
never returns on Linux. Refusal is asserted at `decide()` instead, which is
the part PyOverdrive actually owns, and `svd(compute_uv=False)` and
`norm(ord=2)` keep their full raise-parity checks since they are unaffected.

There is a real product question left open for the owner. PyOverdrive's
closed form returns NaN for these inputs in microseconds, and NaN is what
Windows numpy returns, and what Linux `svd(compute_uv=False)` returns for
the same matrix. Serving infinite batches instead of refusing them would
turn an infinite hang into the answer every other spelling already gives.
It is tempting and it is a genuine divergence from stock-on-Linux, so it is
not a call to make silently.
