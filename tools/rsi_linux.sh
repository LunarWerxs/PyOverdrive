#!/bin/sh
# The RSI measure of one cell's instruction count on any Linux (.rsi/rsi.yaml): installs only what a bare
# container lacks, then hands over to tools/instcount.py. numpy is pinned so a reading taken today and
# one taken next month count the same library; a host that already has numpy and valgrind installs nothing.
set -e
command -v valgrind >/dev/null 2>&1 || {
  apt-get -qq update && apt-get -qq install -y --no-install-recommends valgrind
} >/dev/null 2>&1
python -c "import numpy" 2>/dev/null || python -m pip -q install --root-user-action=ignore "numpy==2.5.3" >/dev/null
exec python tools/instcount.py "$@"
