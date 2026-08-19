import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from common.runtime_context import ExecutionContextV1
from common.task_adapter import DEFAULT_TASK
from common.trainer import (
    ResumeMismatchError,
    _best_evaluation_checkpoint_payload,
    _checkpoint_payload,
    _restore_rng_state,
    _validate_resume,
    checkpoint_is_better,
    rng_state_sha256,
    train_candidate_in_process,
)
from common.training_config import (
    SMOKE_TRAIN_CUDA_V2,
    SMOKE_TRAIN_V1,
    TrainingSeedBundle,
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


def test_cuda_v2_checkpoints_bind_dependency_lock_without_changing_v1() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    seeds = TrainingSeedBundle.from_run_seed(12)
    lock_hash = "a" * 64
    resume = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        profile=SMOKE_TRAIN_CUDA_V2,
        candidate_hash="candidate",
        task=DEFAULT_TASK,
        seeds=seeds,
        step=1,
        examples_processed=SMOKE_TRAIN_CUDA_V2.global_batch_size,
        elapsed_seconds=0.1,
        best_step=1,
        best_accuracy=0.0,
        best_loss=1.0,
        final_training_loss=1.0,
        dependency_lock_hash=lock_hash,
    )
    best = _best_evaluation_checkpoint_payload(
        model=model,
        profile=SMOKE_TRAIN_CUDA_V2,
        candidate_hash="candidate",
        task=DEFAULT_TASK,
        seeds=seeds,
        step=1,
        examples_processed=SMOKE_TRAIN_CUDA_V2.global_batch_size,
        best_accuracy=0.0,
        best_loss=1.0,
        dependency_lock_hash=lock_hash,
    )
    assert resume["checkpoint_kind"] == "trusted_resume_state_v2"
    assert best["checkpoint_kind"] == "best_evaluation_weights_v2"
    assert resume["dependency_lock_hash"] == lock_hash
    assert best["dependency_lock_hash"] == lock_hash
    _validate_resume(
        resume,
        candidate_hash="candidate",
        profile=SMOKE_TRAIN_CUDA_V2,
        task=DEFAULT_TASK,
        seeds=seeds,
        dependency_lock_hash=lock_hash,
    )
    with pytest.raises(ResumeMismatchError, match="dependency_lock_hash"):
        _validate_resume(
            resume,
            candidate_hash="candidate",
            profile=SMOKE_TRAIN_CUDA_V2,
            task=DEFAULT_TASK,
            seeds=seeds,
            dependency_lock_hash="b" * 64,
        )

    historical = _checkpoint_payload(
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
        dependency_lock_hash=lock_hash,
    )
    assert historical["checkpoint_kind"] == "trusted_resume_state_v1"
    assert "dependency_lock_hash" not in historical


def test_v2_training_retains_one_nonterminal_resume_checkpoint(tmp_path) -> None:
    profile = replace(
        SMOKE_TRAIN_CUDA_V2,
        max_steps=2,
        validation_interval=1,
        checkpoint_interval=1,
    )
    seeds = TrainingSeedBundle.from_run_seed(73)
    result = train_candidate_in_process(
        candidate_path=ROOT / "common" / "initial_candidate.ir.json",
        output_dir=tmp_path,
        profile=profile,
        seeds=seeds,
        requested_device="cpu",
        allow_cpu_for_tests=True,
    )

    assert result.success
    partial = torch.load(
        tmp_path / "partial_resume_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    latest = torch.load(
        tmp_path / "latest_resume_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert partial["global_step"] == 1
    assert latest["global_step"] == 2
    assert partial["examples_processed"] < latest["examples_processed"]
    _validate_resume(
        partial,
        candidate_hash=result.candidate_source_hash,
        profile=profile,
        task=DEFAULT_TASK,
        seeds=seeds,
    )


def test_rng_state_digest_normalizes_tuples_and_hashes_tensor_bytes() -> None:
    tensor = torch.tensor([1, 2, 3], dtype=torch.uint8)
    left = {"python": (3, (1, 2), None), "torch_cpu": tensor}
    right = {"torch_cpu": tensor.clone(), "python": [3, [1, 2], None]}

    assert rng_state_sha256(left) == rng_state_sha256(right)
    changed = {"python": [3, [1, 2], None], "torch_cpu": tensor + 1}
    assert rng_state_sha256(left) != rng_state_sha256(changed)


def test_v2_resume_emits_create_only_rng_restore_attestation(tmp_path) -> None:
    profile = replace(
        SMOKE_TRAIN_CUDA_V2,
        max_steps=2,
        validation_interval=1,
        checkpoint_interval=1,
        validation_examples=2,
    )
    seeds = TrainingSeedBundle.from_run_seed(74)
    source = tmp_path / "source"
    first = train_candidate_in_process(
        candidate_path=ROOT / "common" / "initial_candidate.ir.json",
        output_dir=source,
        profile=profile,
        seeds=seeds,
        requested_device="cpu",
        allow_cpu_for_tests=True,
    )
    assert first.success

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    shutil.copy2(
        source / "partial_resume_checkpoint.pt",
        resumed / "latest_resume_checkpoint.pt",
    )
    shutil.copy2(source / "candidate_graph.json", resumed / "candidate_graph.json")
    shutil.copy2(source / "best_checkpoint.pt", resumed / "best_checkpoint.pt")
    context = ExecutionContextV1.local(run_id="resume-attestation-test")

    result = train_candidate_in_process(
        candidate_path=resumed / "candidate_graph.json",
        output_dir=resumed,
        profile=profile,
        seeds=seeds,
        requested_device="cpu",
        allow_cpu_for_tests=True,
        resume=resumed / "latest_resume_checkpoint.pt",
        execution_context=context,
    )

    assert result.success
    attestation_path = resumed / "rng_restore_attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert set(attestation) == {
        "schema_name",
        "schema_version",
        "source_checkpoint_sha256",
        "source_rng_state_sha256",
        "observed_post_restore_rng_state_sha256",
        "restored_exactly",
        "source_optimizer_step",
        "source_examples_processed",
        "final_checkpoint_sha256",
        "final_rng_state_sha256",
        "final_optimizer_step",
        "final_examples_processed",
        "rng_progressed",
        "execution_context",
    }
    assert attestation["schema_name"] == "RNGRestoreAttestation"
    assert attestation["schema_version"] == "1.0"
    assert attestation["restored_exactly"] is True
    assert attestation["rng_progressed"] is True
    assert attestation["source_optimizer_step"] == 1
    assert attestation["final_optimizer_step"] == 2
    assert attestation["source_rng_state_sha256"] == attestation[
        "observed_post_restore_rng_state_sha256"
    ]
    assert attestation["source_rng_state_sha256"] != attestation[
        "final_rng_state_sha256"
    ]
    assert attestation["execution_context"] == context.to_dict()

    original = attestation_path.read_bytes()
    second = train_candidate_in_process(
        candidate_path=resumed / "candidate_graph.json",
        output_dir=resumed,
        profile=profile,
        seeds=seeds,
        requested_device="cpu",
        allow_cpu_for_tests=True,
        resume=resumed / "latest_resume_checkpoint.pt",
        execution_context=context,
    )
    assert not second.success
    assert attestation_path.read_bytes() == original
