from __future__ import annotations

import json
from pathlib import Path

import torch

from common.task_adapter import DEFAULT_TASK
from common.training_config import (
    SMOKE_TRAIN_CUDA_V1,
    TrainingSeedBundle,
)
from common.trusted_candidate import TRUSTED_INITIAL_CANDIDATE_SHA256
from scripts.compare_cuda_smoke_runs import compare_cuda_smoke_runs


def _write_run(root: Path, *, loss_delta: float = 0.0) -> None:
    root.mkdir()
    events = []
    for step in range(1, 11):
        events.append(
            {
                "optimizer_step": step,
                "loss": float(11 - step) + loss_delta,
                "validation_loss": 0.5 if step == 10 else None,
                "validation_exact_match_accuracy": 0.25 if step == 10 else None,
            }
        )
    event_path = root / "training_events.jsonl"
    event_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    model_state = {"weight": torch.tensor([[1.0, 2.0]])}
    seeds = TrainingSeedBundle.from_run_seed(1)
    checkpoint_identity = {
        "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
        "candidate_source_hash": TRUSTED_INITIAL_CANDIDATE_SHA256,
        "task_adapter_hash": DEFAULT_TASK.config_hash,
        "seed_bundle_hash": seeds.bundle_hash,
    }
    best_path = root / "best_checkpoint.pt"
    torch.save(
        {"model_state": model_state, "global_step": 10, **checkpoint_identity},
        best_path,
    )
    torch.save(
        {
            "model_state": model_state,
            "global_step": 10,
            "optimizer_state": {
                "state": {0: {"step": torch.tensor(10.0)}},
                "param_groups": [],
            },
            **checkpoint_identity,
        },
        root / "latest_resume_checkpoint.pt",
    )
    summary = {
        "success": True,
        "device": "cuda:0",
        "profile_name": SMOKE_TRAIN_CUDA_V1.name,
        "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
        "candidate_source_hash": TRUSTED_INITIAL_CANDIDATE_SHA256,
        "initialization_seed": seeds.model_initialization_seed,
        "data_seed": seeds.training_data_seed,
        "development_seed": seeds.development_set_seed,
        "dataloader_seed": seeds.dataloader_seed,
        "steps_completed": 10,
        "best_development_step": 10,
        "best_development_exact_match_accuracy": 0.25,
        "best_development_loss": 0.5,
        "final_training_loss": 1.0 + loss_delta,
        "event_log_path": str(event_path),
        "checkpoint_path": str(best_path),
    }
    manifest = {
        "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
        "candidate_source_hash": TRUSTED_INITIAL_CANDIDATE_SHA256,
        "seed_bundle_hash": seeds.bundle_hash,
        "task_adapter_hash": DEFAULT_TASK.config_hash,
        "controller_source_hash": "4" * 64,
        "dependency_lock_hash": "5" * 64,
    }
    (root / "training_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (root / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_semantic_reproducibility_accepts_equal_checkpoint_states(
    monkeypatch, tmp_path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first)
    _write_run(second)
    monkeypatch.setattr(
        "scripts.compare_cuda_smoke_runs.validate_cuda_a40_training_output",
        lambda _path, *, expected_seed: {"expected_seed": expected_seed},
    )

    report = compare_cuda_smoke_runs(first, second)

    assert report["passed"] is True
    assert report["checks"]["model_state_tensors_equal"] is True
    assert report["checks"]["loss_sequence_equal"] is True


def test_semantic_reproducibility_rejects_changed_loss_trajectory(
    monkeypatch, tmp_path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first)
    _write_run(second, loss_delta=0.125)
    monkeypatch.setattr(
        "scripts.compare_cuda_smoke_runs.validate_cuda_a40_training_output",
        lambda _path, *, expected_seed: {"expected_seed": expected_seed},
    )

    report = compare_cuda_smoke_runs(first, second)

    assert report["passed"] is False
    assert report["checks"]["loss_sequence_equal"] is False
