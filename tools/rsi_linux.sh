#!/bin/sh
# The RSI measure of one cell's instruction count on any Linux (.rsi/rsi.yaml): installs only what a bare
# container lacks, then hands over to tools/instcount.py. numpy is pinned so a reading taken today and
# one taken next month count the same library (any other numpy is replaced: it runs in the throwaway Linux-leg
# container, `image:` in the manifest); a host that already has numpy 2.5.3 and valgrind installs nothing.
set -e
command -v valgrind >/dev/null 2>&1 || {
  apt-get -qq update && apt-get -qq install -y --no-install-recommends valgrind
} >/dev/null 2>&1 || { echo "rsi_linux: installing valgrind failed (root? network?)" >&2; exit 1; }
python -c "import numpy, sys; sys.exit(numpy.__version__ != '2.5.3')" 2>/dev/null ||
  python -m pip -q install --root-user-action=ignore --disable-pip-version-check "numpy==2.5.3" >/dev/null 2>&1 ||
  { echo "rsi_linux: installing numpy 2.5.3 failed (network? no wheel for this Python?)" >&2; exit 1; }
exec python tools/instcount.py "$@"
