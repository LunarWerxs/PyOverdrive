"""Machine fingerprinting for Dyno results.

Every benchmark result must be attributable to the exact hardware/software
stack that produced it (spec section 8.1). The fingerprint id is a stable hash
of the identity-bearing fields, so results from the same machine+stack land in
the same results file.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys


def machine_fingerprint() -> dict:
    import numpy as np

    try:
        from numpy._core._multiarray_umath import __cpu_features__

        simd_features = sorted(k for k, v in __cpu_features__.items() if v)
    except Exception:
        simd_features = []

    blas = None
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {}).get("name")
    except Exception:
        pass

    info = {
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "logical_cores": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "blas": blas,
    }
    digest = hashlib.sha256(
        json.dumps(info, sort_keys=True).encode()
    ).hexdigest()[:12]
    info["simd_features"] = simd_features
    info["fingerprint"] = digest
    return info


# No __main__ block here on purpose: tools/fingerprint.py is the documented
# CLI (AGENTS.md) and had the identical body. Two entry points to the same
# two lines is one more than anyone needs.
