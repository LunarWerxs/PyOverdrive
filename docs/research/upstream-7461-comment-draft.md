# Draft comment for numpy/numpy#7461 - NOT POSTED

Posting to a public tracker is the owner's call. This is ready to paste as
a comment on https://github.com/numpy/numpy/issues/7461 if he says go.

---

Still reproduces ten years on, and the trigger turns out to be narrower and
more diagnostic than the original report suggests. I hit this from the
outside (a CI suite that wedged only on Linux) and then bisected it.

**The key observation: it is `compute_uv=True` that hangs. The values-only
SVD is healthy on the very same matrix.**

```python
import numpy as np
m = np.eye(3)
m[0, 0] = np.inf

np.linalg.svd(m, compute_uv=False)     # -> [nan nan nan]   returns
np.linalg.norm(m, ord=2)               # -> nan             returns
np.linalg.svd(m, full_matrices=False)  # never returns
np.linalg.svd(m)                       # never returns
np.linalg.pinv(m)                      # never returns  (it calls the above)
```

So `pinv` is not really the subject, it is a caller. The hang is in `gesdd`
with `jobz='S'`/`'A'`, while `jobz='N'` on identical input returns NaN
promptly. That asymmetry looks like the bidiagonal divide-and-conquer path
lacking the non-convergence bail-out that the QR-iteration path has: with
`jobz='N'` a NaN input yields `info > 0` and numpy raises
`LinAlgError: SVD did not converge`, which is the behaviour this issue asks
for.

**It is platform-split**, which I did not see mentioned. Every case below
was run in its own process under `timeout -s KILL 45`.

| case | Linux x86-64 | Windows x86-64 |
| --- | --- | --- |
| `svd(compute_uv=False)`, 3x3, inf on diagonal | `[nan nan nan]` | same |
| `norm(ord=2)`, 3x3, inf on diagonal | `nan` | same |
| `svd(full_matrices=False)`, 3x3, inf on diagonal | **hangs** | returns |
| `svd()`, 3x3, inf on diagonal | **hangs** | returns |
| `pinv`, 3x3, `+inf` on diagonal | **hangs** | returns |
| `pinv`, 3x3, `-inf` on diagonal | **hangs** | returns |
| `pinv`, 4x4, inf on diagonal | **hangs** | returns |
| `pinv`, **2x2**, inf on diagonal | returns | returns |
| `pinv`, 3x3, inf **off** the diagonal | returns | returns |
| `pinv`, 3x3, **nan** instead of inf | raises `LinAlgError` | same |
| `eigvals`, 3x3, inf on diagonal | raises | same |
| batched `pinv`, (100,3,3) with one inf | **hangs** | returns |

The thread **spins** rather than blocking: `/proc/<pid>/task/*/stat` shows
the main thread in state `R` with every other thread idle in
`futex_do_wait`, and 100% of one core. Python stack at the time:

```
File "numpy/linalg/_linalg.py", line 1840 in svd
File "numpy/linalg/_linalg.py", line 2256 in pinv
```

**Versions.** Reproduced on `ubuntu-latest` (GitHub-hosted, x86-64, PyPI
manylinux wheels) with CPython 3.12, 3.13 and 3.14 against numpy 2.0.2,
2.4.5 and 2.5.2, so not a recent regression and not confined to one release
line. Not reproduced on `windows-latest` or on a local Windows box (numpy
2.4.5), where every call above returns in well under a millisecond.

Practical note for anyone else who lands here: because `jobz='N'` is
unaffected, `np.linalg.svd(a, compute_uv=False)` is a safe way to get
singular values out of possibly-infinite data, and it is what
`np.linalg.norm(..., ord=2)` already uses internally.
