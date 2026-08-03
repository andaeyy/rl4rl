"""Generated-candidate loading kept behind the worker-process boundary."""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path
from types import ModuleType


def load_candidate(path: str | Path) -> ModuleType:
    candidate_path = Path(path).resolve()
    module_name = f"discovery_candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load candidate at {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
