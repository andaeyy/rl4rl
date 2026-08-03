"""Trusted isolated-mode bootstrap for common.training_worker."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.training_worker import main


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: training_worker_bootstrap.py JOB_JSON RESPONSE_JSON")
    main(sys.argv[1], sys.argv[2])
