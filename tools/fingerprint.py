"""CLI wrapper: print this machine's Dyno fingerprint as JSON."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.dyno.fingerprint import machine_fingerprint

print(json.dumps(machine_fingerprint(), indent=2))
