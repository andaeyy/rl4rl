#!/usr/bin/env python3
"""Compare two trusted CUDA smoke runs for semantic reproducibility.

Archive byte hashes are reported but are not the acceptance criterion. The
comparison loads trusted local checkpoints with ``weights_only=True`` and
checks tensor values, optimizer/global steps, losses, validation metrics, and
the selected checkpoint step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from common.task_adapter import DEFAULT_TASK  # noqa: E402
from common.training_config import (  # noqa: E402
    FULL_TRAIN_V1,
    SMOKE_TRAIN_CUDA_V1,
    TrainingSeedBundle,
)
from common.trusted_candidate import TRUSTED_INITIAL_CANDIDATE_SHA256  # noqa: E402
from scripts.record_cuda_a40_validation import (  # noqa: E402
    validate_cuda_a40_training_output,
)


FROZEN_FULL_MPS_PROFILE_HASH = (
    "046034a7949f3563fc13dcb38df4b34e997cb5a1ffe6b90e755e2f44bfd9f06e"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _contained_file(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{field} escapes {root}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        events.append(payload)
    if not events:
        raise ValueError(f"{path} contains no training events")
    return events


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a trusted checkpoint mapping")
    return payload


def _optimizer_steps(checkpoint: Mapping[str, Any]) -> list[float]:
    optimizer = checkpoint.get("optimizer_state")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise ValueError("resume checkpoint lacks optimizer state")
    steps: list[float] = []
    for state in optimizer["state"].values():
        if not isinstance(state, dict) or "step" not in state:
            continue
        value = state["step"]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("optimizer step tensor must be scalar")
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("optimizer step must be numeric")
        steps.append(float(value))
    if not steps:
        raise ValueError("resume checkpoint contains no optimizer steps")
    return steps


def _state_tensors_equal(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    differences: list[str] = []
    if not first or not second:
        differences.append("model state is empty")
        return False, differences
    if set(first) != set(second):
        differences.append("model state keys differ")
        return False, differences
    for name in sorted(first):
        left = first[name]
        right = second[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            if left != right:
                differences.append(f"non-tensor state differs: {name}")
            continue
        if left.dtype != right.dtype or left.shape != right.shape:
            differences.append(f"tensor metadata differs: {name}")
        elif not torch.equal(left, right):
            differences.append(f"tensor values differ: {name}")
    return not differences, differences


def _load_run(output_dir: Path) -> dict[str, Any]:
    root = output_dir.resolve()
    summary = _read_json(root / "training_summary.json")
    manifest = _read_json(root / "training_manifest.json")
    event_path = _contained_file(root, summary.get("event_log_path"), "event_log_path")
    best_path = _contained_file(root, summary.get("checkpoint_path"), "checkpoint_path")
    resume_path = root / "latest_resume_checkpoint.pt"
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    events = _read_events(event_path)
    best = _load_checkpoint(best_path)
    resume = _load_checkpoint(resume_path)
    return {
        "root": root,
        "summary": summary,
        "manifest": manifest,
        "events": events,
        "best": best,
        "resume": resume,
        "best_sha256": _sha256_file(best_path),
        "resume_sha256": _sha256_file(resume_path),
    }


def compare_cuda_smoke_runs(first_dir: Path, second_dir: Path) -> dict[str, Any]:
    # Reuse the strict receipt validator so matching-but-wrong runs cannot pass.
    validate_cuda_a40_training_output(first_dir, expected_seed=1)
    validate_cuda_a40_training_output(second_dir, expected_seed=1)
    first = _load_run(first_dir)
    second = _load_run(second_dir)
    first_summary = first["summary"]
    second_summary = second["summary"]
    first_manifest = first["manifest"]
    second_manifest = second["manifest"]
    first_events = first["events"]
    second_events = second["events"]

    tensor_equal, tensor_differences = _state_tensors_equal(
        first["best"].get("model_state", {}),
        second["best"].get("model_state", {}),
    )
    resume_tensor_equal, resume_tensor_differences = _state_tensors_equal(
        first["resume"].get("model_state", {}),
        second["resume"].get("model_state", {}),
    )
    tensor_differences.extend(
        f"resume: {difference}" for difference in resume_tensor_differences
    )
    first_losses = [event.get("loss") for event in first_events]
    second_losses = [event.get("loss") for event in second_events]
    first_validation = [
        {
            "optimizer_step": event.get("optimizer_step"),
            "validation_loss": event.get("validation_loss"),
            "validation_exact_match_accuracy": event.get(
                "validation_exact_match_accuracy"
            ),
        }
        for event in first_events
        if event.get("validation_loss") is not None
    ]
    second_validation = [
        {
            "optimizer_step": event.get("optimizer_step"),
            "validation_loss": event.get("validation_loss"),
            "validation_exact_match_accuracy": event.get(
                "validation_exact_match_accuracy"
            ),
        }
        for event in second_events
        if event.get("validation_loss") is not None
    ]
    first_optimizer_steps = _optimizer_steps(first["resume"])
    second_optimizer_steps = _optimizer_steps(second["resume"])
    identity_fields = (
        "profile_hash",
        "candidate_source_hash",
        "initialization_seed",
        "data_seed",
        "development_seed",
        "dataloader_seed",
    )
    manifest_identity_fields = (
        "profile_hash",
        "candidate_source_hash",
        "seed_bundle_hash",
        "task_adapter_hash",
        "controller_source_hash",
        "dependency_lock_hash",
    )
    expected_seeds = TrainingSeedBundle.from_run_seed(1)
    checkpoint_identity_fields = {
        "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
        "candidate_source_hash": TRUSTED_INITIAL_CANDIDATE_SHA256,
        "task_adapter_hash": DEFAULT_TASK.config_hash,
        "seed_bundle_hash": expected_seeds.bundle_hash,
    }
    checkpoints = (
        first["best"],
        second["best"],
        first["resume"],
        second["resume"],
    )
    checks = {
        "distinct_output_directories": first["root"] != second["root"],
        "successful_cuda_smoke_runs": all(
            summary.get("success") is True
            and summary.get("device") == "cuda:0"
            and summary.get("profile_name") == SMOKE_TRAIN_CUDA_V1.name
            and summary.get("profile_hash") == SMOKE_TRAIN_CUDA_V1.profile_hash
            and summary.get("steps_completed") == 10
            for summary in (first_summary, second_summary)
        ),
        "input_identities_match": all(
            first_summary.get(field) == second_summary.get(field)
            for field in identity_fields
        )
        and all(
            first_manifest.get(field) == second_manifest.get(field)
            for field in manifest_identity_fields
        )
        and all(
            summary.get("candidate_source_hash")
            == manifest.get("candidate_source_hash")
            == TRUSTED_INITIAL_CANDIDATE_SHA256
            for summary, manifest in (
                (first_summary, first_manifest),
                (second_summary, second_manifest),
            )
        )
        and all(
            checkpoint.get(field) == expected
            for checkpoint in checkpoints
            for field, expected in checkpoint_identity_fields.items()
        )
        and all(
            summary.get(field) == expected
            for summary in (first_summary, second_summary)
            for field, expected in {
                "initialization_seed": expected_seeds.model_initialization_seed,
                "data_seed": expected_seeds.training_data_seed,
                "development_seed": expected_seeds.development_set_seed,
                "dataloader_seed": expected_seeds.dataloader_seed,
            }.items()
        ),
        "model_state_tensors_equal": tensor_equal and resume_tensor_equal,
        "global_optimizer_step_equal": (
            first["resume"].get("global_step")
            == second["resume"].get("global_step")
            == 10
        ),
        "per_parameter_optimizer_steps_equal": (
            first_optimizer_steps == second_optimizer_steps
            and all(step == 10.0 for step in first_optimizer_steps)
        ),
        "optimizer_step_sequence_equal": (
            [event.get("optimizer_step") for event in first_events]
            == [event.get("optimizer_step") for event in second_events]
            == list(range(1, 11))
        ),
        "loss_sequence_equal": first_losses == second_losses,
        "validation_metrics_equal": (
            bool(first_validation) and first_validation == second_validation
        ),
        "selected_checkpoint_step_equal": (
            first_summary.get("best_development_step")
            == second_summary.get("best_development_step")
            == first["best"].get("global_step")
            == second["best"].get("global_step")
        ),
        "selected_summary_metrics_equal": all(
            first_summary.get(field) == second_summary.get(field)
            for field in (
                "best_development_exact_match_accuracy",
                "best_development_loss",
                "final_training_loss",
            )
        ),
        "original_mps_profile_hash_preserved": (
            FULL_TRAIN_V1.profile_hash == FROZEN_FULL_MPS_PROFILE_HASH
        ),
    }
    return {
        "schema_version": "cuda_smoke_semantic_reproducibility_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "tensor_differences": tensor_differences,
        "first": {
            "output_dir": str(first["root"]),
            "best_checkpoint_sha256": first["best_sha256"],
            "resume_checkpoint_sha256": first["resume_sha256"],
        },
        "second": {
            "output_dir": str(second["root"]),
            "best_checkpoint_sha256": second["best_sha256"],
            "resume_checkpoint_sha256": second["resume_sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = compare_cuda_smoke_runs(arguments.first, arguments.second)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
