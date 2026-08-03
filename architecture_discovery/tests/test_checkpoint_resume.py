from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from common.task_adapter import DEFAULT_TASK
from common.training_config import SMOKE_TRAIN_V1, TrainingSeedBundle
from common.trainer import (
    ResumeMismatchError,
    _checkpoint_payload,
    _restore_rng_state,
    _validate_resume,
    checkpoint_is_better,
    train_candidate_in_process,
)


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_selection_uses_only_public_development_tuple():
    assert checkpoint_is_better(
        accuracy=0.8,
        loss=2.0,
        step=20,
        best_accuracy=0.7,
        best_loss=1.0,
        best_step=10,
    )
    assert checkpoint_is_better(
        accuracy=0.8,
        loss=0.9,
        step=20,
        best_accuracy=0.8,
        best_loss=1.0,
        best_step=10,
    )
    assert checkpoint_is_better(
        accuracy=0.8,
        loss=1.0,
        step=10,
        best_accuracy=0.8,
        best_loss=1.0,
        best_step=20,
    )


@pytest.mark.parametrize(
    "field",
    [
        "candidate_source_hash",
        "profile_hash",
        "task_adapter_version",
        "task_adapter_hash",
        "seed_bundle_hash",
    ],
)
def test_resume_rejects_identity_mismatch(field):
    seeds = TrainingSeedBundle.from_run_seed(1)
    checkpoint = {
        "checkpoint_kind": "trusted_resume_state_v1",
        "candidate_source_hash": "candidate",
        "profile_hash": SMOKE_TRAIN_V1.profile_hash,
        "task_adapter_version": DEFAULT_TASK.version,
        "task_adapter_hash": DEFAULT_TASK.config_hash,
        "seed_bundle_hash": seeds.bundle_hash,
    }
    checkpoint[field] = "mismatch"
    with pytest.raises(ResumeMismatchError):
        _validate_resume(
            checkpoint,
            candidate_hash="candidate",
            profile=SMOKE_TRAIN_V1,
            task=DEFAULT_TASK,
            seeds=seeds,
        )


def test_nonempty_output_refuses_without_resume_and_preserves_sentinel(tmp_path):
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("unchanged")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = train_candidate_in_process(
        candidate_path=ROOT / "common" / "initial_candidate.py",
        output_dir=tmp_path,
        profile=replace(
            SMOKE_TRAIN_V1,
            max_steps=1,
            validation_interval=1,
            checkpoint_interval=1,
            validation_examples=2,
        ),
        seeds=TrainingSeedBundle.from_run_seed(1),
        requested_device="cpu",
        allow_cpu_for_tests=True,
    )
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert not result.success
    assert result.failure_stage == "checkpoint_write"
    assert before == after


def test_resume_checkpoint_loads_with_weights_only_and_plain_rng_state(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    seeds = TrainingSeedBundle.from_run_seed(9)
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        profile=SMOKE_TRAIN_V1,
        candidate_hash="candidate",
        task=DEFAULT_TASK,
        seeds=seeds,
        step=1,
        examples_processed=16,
        elapsed_seconds=0.1,
        best_step=1,
        best_accuracy=0.0,
        best_loss=1.0,
        final_training_loss=1.0,
    )
    path = tmp_path / "resume.pt"
    torch.save(payload, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    _validate_resume(
        loaded,
        candidate_hash="candidate",
        profile=SMOKE_TRAIN_V1,
        task=DEFAULT_TASK,
        seeds=seeds,
    )
    _restore_rng_state(loaded["rng_state"])


def test_resume_checkpoint_rejects_boolean_step_type() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    seeds = TrainingSeedBundle.from_run_seed(11)
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        profile=SMOKE_TRAIN_V1,
        candidate_hash="candidate",
        task=DEFAULT_TASK,
        seeds=seeds,
        step=1,
        examples_processed=SMOKE_TRAIN_V1.global_batch_size,
        elapsed_seconds=0.1,
        best_step=1,
        best_accuracy=0.0,
        best_loss=1.0,
        final_training_loss=1.0,
    )
    payload["global_step"] = True

    with pytest.raises(ResumeMismatchError, match="global_step"):
        _validate_resume(
            payload,
            candidate_hash="candidate",
            profile=SMOKE_TRAIN_V1,
            task=DEFAULT_TASK,
            seeds=seeds,
        )
