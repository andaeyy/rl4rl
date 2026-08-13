"""Read-only negative probes for the version-2 CUDA resume contract.

This script loads a trusted resume checkpoint with ``weights_only=True``, first
validates its unmodified identity, and then validates four in-memory tampered
copies.  Each copy changes exactly one identity hash.  The checkpoint itself is
never written, and a byte hash taken before and after the probes must match.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.task_adapter import DEFAULT_TASK  # noqa: E402
from common.trainer import (  # noqa: E402
    ResumeMismatchError,
    _dependency_lock_hash,
    _validate_resume,
    sha256_file,
    trusted_component_set_sha256,
)
from common.training_config import (  # noqa: E402
    TrainingSeedBundle,
    get_training_profile,
)

SCHEMA_NAME = "ResumeContractVerification"
SCHEMA_VERSION = "1.0"
PROBE_FIELDS = (
    "candidate_source_hash",
    "profile_hash",
    "seed_bundle_hash",
    "dependency_lock_hash",
)


class ResumeContractVerificationError(RuntimeError):
    """The negative resume probes could not establish the frozen contract."""


def _alternate_sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ResumeContractVerificationError(
            "checkpoint identity fields must contain SHA-256 text"
        )
    alternate = "0" * 64
    return "1" * 64 if value == alternate else alternate


def _atomic_json(destination: Path, payload: dict[str, Any]) -> None:
    if destination.exists() or destination.is_symlink():
        raise ResumeContractVerificationError(
            "resume-contract evidence destination already exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_resume_contract(
    *,
    checkpoint_path: str | Path,
    candidate_path: str | Path,
    profile_name: str,
    run_seed: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Verify four identity mismatches are rejected without changing the input."""

    checkpoint = Path(checkpoint_path)
    candidate = Path(candidate_path)
    output = Path(output_path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ResumeContractVerificationError(
            "resume checkpoint must be one regular, non-symlink file"
        )
    if candidate.is_symlink() or not candidate.is_file():
        raise ResumeContractVerificationError(
            "candidate must be one regular, non-symlink file"
        )
    if output.resolve() in {checkpoint.resolve(), candidate.resolve()}:
        raise ResumeContractVerificationError(
            "evidence output may not replace an input artifact"
        )

    profile = get_training_profile(profile_name)
    if profile.version != "2" or profile.device_requirement != "cuda":
        raise ResumeContractVerificationError(
            "negative resume probes are restricted to version-2 CUDA profiles"
        )
    seeds = TrainingSeedBundle.from_run_seed(run_seed)
    candidate_hash = sha256_file(candidate)
    dependency_lock_hash = _dependency_lock_hash()
    component_set_hash = trusted_component_set_sha256()
    checkpoint_sha256_before = sha256_file(checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ResumeContractVerificationError("resume checkpoint must be a mapping")
    if loaded.get("checkpoint_kind") != "trusted_resume_state_v2":
        raise ResumeContractVerificationError(
            "negative resume probes require trusted_resume_state_v2"
        )

    validation_arguments = {
        "candidate_hash": candidate_hash,
        "profile": profile,
        "task": DEFAULT_TASK,
        "seeds": seeds,
        "trusted_component_set_hash": component_set_hash,
        "dependency_lock_hash": dependency_lock_hash,
    }
    try:
        _validate_resume(loaded, **validation_arguments)
    except ResumeMismatchError as error:
        raise ResumeContractVerificationError(
            "unmodified checkpoint failed its current resume identity"
        ) from error

    probes: list[dict[str, Any]] = []
    for field in PROBE_FIELDS:
        tampered = dict(loaded)
        tampered_value = _alternate_sha256(tampered.get(field))
        tampered[field] = tampered_value
        try:
            _validate_resume(tampered, **validation_arguments)
        except ResumeMismatchError as error:
            if field not in str(error):
                raise ResumeContractVerificationError(
                    f"{field} probe was rejected for an unrelated reason"
                ) from error
        else:
            raise ResumeContractVerificationError(
                f"resume validation accepted a mismatched {field}"
            )
        probes.append(
            {
                "error_type": "ResumeMismatchError",
                "field": field,
                "name": f"{field}_mismatch",
                "rejected": True,
                "tampered_value": tampered_value,
            }
        )

    checkpoint_sha256_after = sha256_file(checkpoint)
    if checkpoint_sha256_after != checkpoint_sha256_before:
        raise ResumeContractVerificationError(
            "resume checkpoint bytes changed during read-only probes"
        )
    evidence = {
        "all_mismatches_rejected": True,
        "checkpoint_kind": loaded["checkpoint_kind"],
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256_before,
        "checkpoint_unchanged": True,
        "expected_identity": {
            "candidate_source_hash": candidate_hash,
            "dependency_lock_hash": dependency_lock_hash,
            "profile_hash": profile.profile_hash,
            "seed_bundle_hash": seeds.bundle_hash,
        },
        "probe_count": len(probes),
        "probes": probes,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "verification_mode": "read_only_in_memory_tamper_copies",
    }
    _atomic_json(output, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        evidence = verify_resume_contract(
            checkpoint_path=args.checkpoint,
            candidate_path=args.candidate,
            profile_name=args.profile,
            run_seed=args.seed,
            output_path=args.output,
        )
    except (OSError, ValueError, ResumeContractVerificationError) as error:
        raise SystemExit(
            f"resume-contract verification failed: {type(error).__name__}: {error}"
        ) from error
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
