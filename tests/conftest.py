from __future__ import annotations

import os

# Multi-threaded OpenBLAS makes the end-to-end retrievals several times slower
# (the quickstart sets this too).  This has to happen before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
