from pathlib import Path

import pytest
import torch

import common.trainer as trainer_module
from common.task_adapter import DEFAULT_TASK
from common.training_config import (
    SMOKE_TRAIN_CUDA_V1,
    TrainingSeedBundle,
)
from common.trainer import (
    _checkpoint_payload,
    _parameter_state_sha256,
    _restore_rng_state,
    _rng_state,
    _validate_resume,
    configure_deterministic_runtime,
    seed_everything,
    training_manifest,
)


def test_cuda_deterministic_runtime_disables_tf32_and_benchmark(monkeypatch):
    deterministic_calls = []
    precision_calls = []
    deterministic_state = {"enabled": False}
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def set_deterministic(enabled):
        deterministic_state["enabled"] = enabled
        deterministic_calls.append(enabled)

    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        set_deterministic,
    )
    monkeypatch.setattr(
        torch,
        "are_deterministic_algorithms_enabled",
        lambda: deterministic_state["enabled"],
    )
    monkeypatch.setattr(
        torch,
        "set_float32_matmul_precision",
        lambda value: precision_calls.append(value),
    )
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)

    configure_deterministic_runtime(
        torch.device("cuda:0"),
        deterministic=True,
    )

    assert deterministic_calls == [True]
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert precision_calls == ["highest"]


def test_cuda_deterministic_runtime_rejects_missing_cublas_precondition(monkeypatch):
    deterministic_state = {"enabled": False}
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    def set_deterministic(enabled):
        deterministic_state["enabled"] = enabled

    monkeypatch.setattr(torch, "use_deterministic_algorithms", set_deterministic)
    monkeypatch.setattr(
        torch,
        "are_deterministic_algorithms_enabled",
        lambda: deterministic_state["enabled"],
    )
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda _value: None)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)

    with pytest.raises(RuntimeError, match="deterministic CUDA runtime controls changed"):
        configure_deterministic_runtime(
            torch.device("cuda:0"),
            deterministic=True,
        )


def test_cuda_determinism_assertion_rejects_candidate_mutation(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        torch,
        "are_deterministic_algorithms_enabled",
        lambda: False,
    )
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")

    with pytest.raises(RuntimeError, match="controls changed"):
        trainer_module.assert_cuda_deterministic_runtime(torch.device("cuda:0"))


def test_cuda_seed_is_explicit_and_keeps_cpu_dataloader_generator(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("cpu", seed)))
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: calls.append(("cuda", seed)),
    )
    monkeypatch.setattr(
        trainer_module,
        "configure_deterministic_runtime",
        lambda device, *, deterministic: calls.append(
            (str(device), deterministic)
        ),
    )

    generator = seed_everything(
        19,
        deterministic=True,
        device=torch.device("cuda:0"),
    )

    assert ("cpu", 19) in calls
    assert ("cuda", 19) in calls
    assert ("cuda:0", True) in calls
    assert generator.device.type == "cpu"


def test_cuda_rng_state_round_trips_through_plain_tensor_list(monkeypatch):
    original = torch.tensor([1, 2, 3], dtype=torch.uint8)
    restored = []
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [original])
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda state: restored.extend(state),
    )

    state = _rng_state(torch.device("cuda:0"))
    original.zero_()
    _restore_rng_state(state)

    assert len(state["torch_cuda"]) == 1
    assert torch.equal(state["torch_cuda"][0], torch.tensor([1, 2, 3]))
    assert len(restored) == 1
    assert torch.equal(restored[0], torch.tensor([1, 2, 3]))


def test_cuda_rng_state_remains_weights_only_checkpoint_compatible(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([4, 5], dtype=torch.uint8)],
    )
    restored = []
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda state: restored.extend(state),
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    seeds = TrainingSeedBundle.from_run_seed(1)
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        profile=SMOKE_TRAIN_CUDA_V1,
        candidate_hash="candidate",
        task=DEFAULT_TASK,
        seeds=seeds,
        step=1,
        examples_processed=SMOKE_TRAIN_CUDA_V1.global_batch_size,
        elapsed_seconds=0.1,
        best_step=1,
        best_accuracy=0.0,
        best_loss=1.0,
        final_training_loss=1.0,
        device=torch.device("cuda:0"),
    )
    path = tmp_path / "cuda-resume.pt"
    torch.save(payload, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    _validate_resume(
        loaded,
        candidate_hash="candidate",
        profile=SMOKE_TRAIN_CUDA_V1,
        task=DEFAULT_TASK,
        seeds=seeds,
    )
    _restore_rng_state(loaded["rng_state"])

    assert torch.equal(restored[0], torch.tensor([4, 5], dtype=torch.uint8))


@pytest.mark.parametrize("cuda_state", [None, []])
def test_cuda_resume_rejects_missing_or_empty_cuda_rng_state(cuda_state):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    seeds = TrainingSeedBundle.from_run_seed(1)
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        profile=SMOKE_TRAIN_CUDA_V1,
        candidate_hash="candidate",
        task=DEFAULT_TASK,
        seeds=seeds,
        step=1,
        examples_processed=SMOKE_TRAIN_CUDA_V1.global_batch_size,
        elapsed_seconds=0.1,
        best_step=1,
        best_accuracy=0.0,
        best_loss=1.0,
        final_training_loss=1.0,
        device=None,
    )
    if cuda_state is not None:
        payload["rng_state"]["torch_cuda"] = cuda_state

    with pytest.raises(trainer_module.ResumeMismatchError, match="CUDA RNG"):
        _validate_resume(
            payload,
            candidate_hash="candidate",
            profile=SMOKE_TRAIN_CUDA_V1,
            task=DEFAULT_TASK,
            seeds=seeds,
        )


def test_parameter_hash_proves_state_changes():
    model = torch.nn.Linear(2, 2)
    before = _parameter_state_sha256(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    after = _parameter_state_sha256(model)

    assert len(before) == 64
    assert len(after) == 64
    assert before != after


def test_cuda_manifest_records_selected_hardware_and_determinism(
    monkeypatch,
    tmp_path,
):
    accelerator = {
        "schema_version": "accelerator_telemetry_v1",
        "backend": "cuda",
        "device": "cuda:0",
        "identity": {
            "name": "NVIDIA A40",
            "uuid": "GPU-1234",
            "total_memory_bytes": 48_000_000_000,
            "compute_capability": [8, 6],
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_runtime_version": "12.6",
            "cuda_driver_version": "55054",
            "cudnn_version": 90501,
        },
        "allocation": {"job_id": "12345", "visible_devices": "GPU-1234"},
    }
    monkeypatch.setattr(
        trainer_module,
        "accelerator_runtime_metadata",
        lambda _device: accelerator,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        torch,
        "are_deterministic_algorithms_enabled",
        lambda: True,
    )
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")

    manifest = training_manifest(
        candidate_path=Path(tmp_path / "candidate.py"),
        candidate_hash="a" * 64,
        profile=SMOKE_TRAIN_CUDA_V1,
        seeds=TrainingSeedBundle.from_run_seed(1),
        requested_device="cuda",
        selected_device="cuda:0",
        task=DEFAULT_TASK,
        allow_cpu_for_tests=False,
        containment_audit={},
        containment_decision={"allowed": True, "scientific": False},
    )

    assert manifest["requested_device"] == "cuda"
    assert manifest["selected_device"] == "cuda:0"
    assert manifest["runtime"]["accelerator"] == accelerator
    assert manifest["runtime"]["cuda_determinism"] == {
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "float32_matmul_precision": "highest",
    }
    assert "separate hardware conditions" in manifest["hardware_condition_note"]
