"""OpenEvolve adapter for the shared discovery evaluator."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.openevolve_adapter import evaluate_for_openevolve


def evaluate(program_path: str):
    return evaluate_for_openevolve(program_path)
