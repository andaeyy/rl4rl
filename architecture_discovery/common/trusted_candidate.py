"""Immutable identity for the only candidate allowed in the CUDA smoke lane."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRUSTED_INITIAL_CANDIDATE_PATH = (ROOT / "common" / "initial_candidate.py").resolve()
TRUSTED_INITIAL_CANDIDATE_SHA256 = (
    "fee5f0784fbc3bf2c51d450e24064a4a6a988b92658043ce19b40101db294007"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_trusted_initial_candidate(path: str | Path) -> str:
    candidate = Path(path).resolve()
    if candidate != TRUSTED_INITIAL_CANDIDATE_PATH:
        raise ValueError(
            "smoke_train_cuda_v1 is restricted to common/initial_candidate.py"
        )
    observed = sha256_bytes(candidate.read_bytes())
    if observed != TRUSTED_INITIAL_CANDIDATE_SHA256:
        raise ValueError(
            "trusted CUDA smoke candidate hash differs from its pinned identity"
        )
    return observed
