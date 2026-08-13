from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from common.runtime_context import ExecutionContextV1
from common.task_adapter import DEFAULT_TASK
from common.trainer import _checkpoint_payload, sha256_file
from common.training_config import (
    SMOKE_TRAIN_CUDA_V2,
    SMOKE_TRAIN_V1,
    TrainingSeedBundle,
)
from modal_boundary import (
    ModalBoundaryError,
    build_artifact_manifest,
    load_artifact_manifest,
    verify_artifact_manifest,
    write_artifact_manifest,
)
from scripts.verify_resume_contract import (
    PROBE_FIELDS,
    ResumeContractVerificationError,
    verify_resume_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "common" / "initial_candidate.ir.json"


def _checkpoint(profile, *, candidate_hash: str, seed: int, step: int = 1) -> dict:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    return _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        profile=profile,
        candidate_hash=candidate_hash,
        task=DEFAULT_TASK,
        seeds=TrainingSeedBundle.from_run_seed(seed),
        step=step,
        examples_processed=step * profile.global_batch_size,
        elapsed_seconds=0.1,
        best_step=step,
        best_accuracy=0.0,
        best_loss=1.0,
        final_training_loss=1.0,
    )


def test_v2_resume_contract_probes_are_canonical_and_read_only(tmp_path) -> None:
    checkpoint = tmp_path / "latest_resume_checkpoint.pt"
    evidence_path = tmp_path / "resume_contract_verification.json"
    torch.save(
        _checkpoint(
            SMOKE_TRAIN_CUDA_V2,
            candidate_hash=sha256_file(CANDIDATE),
            seed=1,
        ),
        checkpoint,
    )
    original = checkpoint.read_bytes()

    evidence = verify_resume_contract(
        checkpoint_path=checkpoint,
        candidate_path=CANDIDATE,
        profile_name=SMOKE_TRAIN_CUDA_V2.name,
        run_seed=1,
        output_path=evidence_path,
    )

    assert checkpoint.read_bytes() == original
    assert evidence_path.read_text(encoding="utf-8") == (
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    assert evidence["schema_name"] == "ResumeContractVerification"
    assert evidence["schema_version"] == "1.0"
    assert evidence["checkpoint_unchanged"] is True
    assert evidence["all_mismatches_rejected"] is True
    assert evidence["probe_count"] == len(PROBE_FIELDS)
    assert tuple(probe["field"] for probe in evidence["probes"]) == PROBE_FIELDS
    assert all(probe["rejected"] is True for probe in evidence["probes"])


def test_v1_checkpoint_is_refused_without_changing_bytes(tmp_path) -> None:
    checkpoint = tmp_path / "historical_resume_checkpoint.pt"
    evidence_path = tmp_path / "should-not-exist.json"
    torch.save(
        _checkpoint(
            SMOKE_TRAIN_V1,
            candidate_hash=sha256_file(CANDIDATE),
            seed=7,
        ),
        checkpoint,
    )
    original = checkpoint.read_bytes()

    with pytest.raises(
        ResumeContractVerificationError,
        match="restricted to version-2 CUDA profiles",
    ):
        verify_resume_contract(
            checkpoint_path=checkpoint,
            candidate_path=CANDIDATE,
            profile_name=SMOKE_TRAIN_V1.name,
            run_seed=7,
            output_path=evidence_path,
        )

    assert checkpoint.read_bytes() == original
    assert not evidence_path.exists()


def _source_candidate_run(tmp_path: Path, modal_app, *, seed: int = 1):
    source_run_id = "modal-candidate-source"
    source = tmp_path / "runs" / source_run_id
    training = source / "candidate_smoke" / f"seed_{seed}"
    training.mkdir(parents=True)
    candidate = training / "candidate_graph.json"
    candidate.write_bytes(CANDIDATE.read_bytes())
    candidate_hash = sha256_file(candidate)
    torch.save(
        _checkpoint(
            SMOKE_TRAIN_CUDA_V2,
            candidate_hash=candidate_hash,
            seed=seed,
            step=SMOKE_TRAIN_CUDA_V2.checkpoint_interval,
        ),
        training / "partial_resume_checkpoint.pt",
    )
    torch.save(
        _checkpoint(
            SMOKE_TRAIN_CUDA_V2,
            candidate_hash=candidate_hash,
            seed=seed,
            step=SMOKE_TRAIN_CUDA_V2.max_steps,
        ),
        training / "latest_resume_checkpoint.pt",
    )
    (training / "training_events.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "elapsed_seconds": float(step),
                    "examples_processed": (
                        step * SMOKE_TRAIN_CUDA_V2.global_batch_size
                    ),
                    "optimizer_step": step,
                },
                sort_keys=True,
            )
            + "\n"
            for step in range(1, SMOKE_TRAIN_CUDA_V2.max_steps + 1)
        ),
        encoding="utf-8",
    )
    (source / "remote_action_result.json").write_text(
        json.dumps(
            {
                "mode": "cuda_candidate_train_and_layer_a",
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "execution_context.json").write_text("{}\n", encoding="utf-8")
    manifest = build_artifact_manifest(
        source,
        run_id=source_run_id,
        image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
    )
    write_artifact_manifest(
        source,
        manifest,
        filename="artifact_manifest.checkpoint.json",
    )
    return source_run_id, source, training


def _context(modal_app, run_id: str) -> ExecutionContextV1:
    return ExecutionContextV1(
        execution_backend="modal",
        run_id=run_id,
        app_name="rl4rl-architecture-discovery",
        function_name="checkpoint_resume",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id=f"fc-{run_id}",
        modal_image_id=None,
        image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
        artifact_uri=f"volume://rl4rl-architecture-artifacts/runs/{run_id}",
    )


def _source_snapshot(source: Path) -> dict[str, bytes]:
    return {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


def _complete_fake_resume(command: list[str]) -> None:
    training = Path(command[command.index("--output-dir") + 1])
    latest = Path(command[command.index("--resume") + 1])
    latest.write_bytes(b"synthetic-resumed-latest")
    events = training / "training_events.jsonl"
    with events.open("a", encoding="utf-8") as handle:
        for step in range(
            SMOKE_TRAIN_CUDA_V2.checkpoint_interval + 1,
            SMOKE_TRAIN_CUDA_V2.max_steps + 1,
        ):
            handle.write(
                json.dumps(
                    {
                        "elapsed_seconds": float(step),
                        "examples_processed": (
                            step * SMOKE_TRAIN_CUDA_V2.global_batch_size
                        ),
                        "optimizer_step": step,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    (training / "best_checkpoint.pt").write_bytes(b"synthetic-best")
    (training / "training_summary.json").write_text(
        '{"success":true}\n', encoding="utf-8"
    )


def _configure_resume_test(monkeypatch, tmp_path, modal_app):
    class FakeVolume:
        def __init__(self):
            self.commits = 0
            self.reloads = 0

        def reload(self):
            self.reloads += 1

        def commit(self):
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setenv(
        modal_app.IMAGE_SOURCE_IDENTITY_ENV,
        modal_app.IMAGE_SOURCE_SHA256,
    )
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setattr(modal_app, "VOLUME_MOUNT_PATH", tmp_path)
    monkeypatch.setattr(
        modal_app,
        "_execution_context",
        lambda _function_name, run_id: _context(modal_app, run_id),
    )
    monkeypatch.setattr(
        modal_app,
        "_provider_free_network_denial_probe",
        lambda _context: {"denied": True},
    )
    return volume


def test_modal_resume_requires_one_checkpoint_manifest_and_rejects_mixed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import modal_app

    source_run_id, source, _training = _source_candidate_run(tmp_path, modal_app)
    checkpoint = source / "artifact_manifest.checkpoint.json"
    (source / "artifact_manifest.json").write_bytes(checkpoint.read_bytes())
    volume = _configure_resume_test(monkeypatch, tmp_path, modal_app)

    with pytest.raises(ModalBoundaryError, match="exactly one"):
        modal_app._resume_action(
            source_run_id,
            "modal-resume-mixed-manifests",
            seed=1,
        )

    assert volume.reloads == 1
    assert volume.commits == 0
    assert not (tmp_path / "runs" / "modal-resume-mixed-manifests").exists()


def test_modal_resume_rejects_final_manifest_as_checkpoint_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import modal_app

    source_run_id, source, _training = _source_candidate_run(tmp_path, modal_app)
    checkpoint = source / "artifact_manifest.checkpoint.json"
    checkpoint.rename(source / "artifact_manifest.json")
    volume = _configure_resume_test(monkeypatch, tmp_path, modal_app)

    with pytest.raises(
        modal_app.RemoteActionError,
        match="checkpoint artifact manifest",
    ):
        modal_app._resume_action(
            source_run_id,
            "modal-resume-final-only",
            seed=1,
        )

    assert volume.reloads == 1
    assert volume.commits == 0


def test_modal_resume_uses_fresh_attempt_and_leaves_source_immutable(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    source_run_id, source, _source_training = _source_candidate_run(
        tmp_path, modal_app
    )
    source_before = _source_snapshot(source)
    volume = _configure_resume_test(monkeypatch, tmp_path, modal_app)
    attempt_run_id = "modal-resume-attempt-1"
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        assert volume.commits == 1
        calls.append((command, kwargs))
        script = Path(command[1]).name
        if script == "verify_resume_contract.py":
            Path(command[command.index("--output") + 1]).write_text(
                '{"all_mismatches_rejected":true}\n', encoding="utf-8"
            )
        elif script == "train_candidate.py":
            training = Path(command[command.index("--output-dir") + 1])
            partial = training / "partial_resume_checkpoint.pt"
            latest = training / "latest_resume_checkpoint.pt"
            assert partial.read_bytes() == latest.read_bytes()
            prefix = [
                json.loads(line)
                for line in (training / "training_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            assert [item["optimizer_step"] for item in prefix] == [1, 2, 3, 4, 5]
            _complete_fake_resume(command)
        elif script == "verify_resume_progression.py":
            Path(command[command.index("--output") + 1]).write_text(
                '{"schema_name":"ResumeProgressionEvidence"}\n',
                encoding="utf-8",
            )
        return {"returncode": 0, "stdout_sha256": str(len(calls)) * 64}

    monkeypatch.setattr(modal_app, "_run_command", fake_run)

    result = modal_app._resume_action(
        source_run_id,
        attempt_run_id,
        seed=1,
    )

    attempt = tmp_path / "runs" / attempt_run_id
    assert [Path(call[0][1]).name for call in calls] == [
        "verify_resume_contract.py",
        "train_candidate.py",
        "verify_resume_progression.py",
    ]
    timeouts = [call[1]["timeout_seconds"] for call in calls]
    assert timeouts == [
        modal_app.RESUME_PROBE_TIMEOUT_SECONDS,
        modal_app.RESUME_TRAIN_TIMEOUT_SECONDS,
        modal_app.RESUME_PROGRESSION_TIMEOUT_SECONDS,
    ]
    assert sum(timeouts) + modal_app.RESUME_INTERNAL_RESERVE_SECONDS < (
        modal_app.RESUME_ACTION_DEADLINE_SECONDS
    )
    assert result["success"] is True
    assert result["source_run_id"] == source_run_id
    assert _source_snapshot(source) == source_before
    assert volume.commits == 2
    manifest = load_artifact_manifest(attempt / "artifact_manifest.json")
    assert verify_artifact_manifest(attempt, manifest)["verified"] is True
    artifact_paths = {item.relative_path for item in manifest.files}
    assert {
        "resume_contract_verification.json",
        "resume_progression_verification.json",
        "resume_source_binding.json",
        "resume_action_result.json",
    } <= artifact_paths
    training = attempt / "candidate_smoke" / "seed_1"
    assert (training / "partial_resume_checkpoint.pt").is_file()
    assert [
        json.loads(line)["optimizer_step"]
        for line in (training / "training_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ] == list(range(1, 11))


def test_resume_shared_deadline_fails_before_an_unbounded_stage(monkeypatch) -> None:
    import modal_app

    monkeypatch.setattr(modal_app.time, "monotonic", lambda: 100.0)

    with pytest.raises(modal_app.RemoteActionError, match="before training"):
        modal_app._resume_stage_timeout(
            130.0,
            maximum_seconds=modal_app.RESUME_TRAIN_TIMEOUT_SECONDS,
            reserve_after_seconds=30,
            stage="training",
        )


def test_modal_resume_failure_is_retryable_from_untouched_source(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    source_run_id, source, _source_training = _source_candidate_run(
        tmp_path, modal_app
    )
    source_before = _source_snapshot(source)
    volume = _configure_resume_test(monkeypatch, tmp_path, modal_app)
    failed_attempt = "modal-resume-failed"
    retry_attempt = "modal-resume-retry"

    def fake_run(command, **kwargs):
        script = Path(command[1]).name
        attempt_run_id = kwargs["context"].run_id
        if script == "verify_resume_contract.py":
            Path(command[command.index("--output") + 1]).write_text(
                '{"all_mismatches_rejected":true}\n', encoding="utf-8"
            )
        elif script == "train_candidate.py" and attempt_run_id == failed_attempt:
            raise RuntimeError("simulated resume failure")
        elif script == "train_candidate.py":
            _complete_fake_resume(command)
        elif script == "verify_resume_progression.py":
            Path(command[command.index("--output") + 1]).write_text(
                '{"schema_name":"ResumeProgressionEvidence"}\n',
                encoding="utf-8",
            )
        return {"returncode": 0, "stdout_sha256": "d" * 64}

    monkeypatch.setattr(modal_app, "_run_command", fake_run)

    with pytest.raises(RuntimeError, match="simulated resume failure"):
        modal_app._resume_action(source_run_id, failed_attempt, seed=1)

    assert _source_snapshot(source) == source_before
    failed = tmp_path / "runs" / failed_attempt
    failed_manifest = load_artifact_manifest(failed / "artifact_manifest.json")
    assert verify_artifact_manifest(failed, failed_manifest)["verified"] is True
    failed_result = json.loads(
        (failed / "resume_action_result.json").read_text(encoding="utf-8")
    )
    assert failed_result["success"] is False
    assert failed_result["source_run_id"] == source_run_id
    assert not (failed / "candidate_smoke").exists()
    assert not (failed / "provider_free_network_denial_probe.json").exists()

    retried = modal_app._resume_action(source_run_id, retry_attempt, seed=1)

    assert retried["success"] is True
    assert retried["source_run_id"] == source_run_id
    assert _source_snapshot(source) == source_before
    retry = tmp_path / "runs" / retry_attempt
    retry_manifest = load_artifact_manifest(retry / "artifact_manifest.json")
    assert verify_artifact_manifest(retry, retry_manifest)["verified"] is True
    assert volume.commits == 4


def test_modal_resume_setup_copy_failure_finalizes_only_fresh_attempt(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    source_run_id, source, _source_training = _source_candidate_run(
        tmp_path, modal_app
    )
    source_before = _source_snapshot(source)
    volume = _configure_resume_test(monkeypatch, tmp_path, modal_app)
    attempt_run_id = "modal-resume-copy-failure"

    def fail_copy(*_args, **_kwargs):
        raise RuntimeError("simulated setup copy failure")

    monkeypatch.setattr(modal_app, "_copy_manifest_artifact", fail_copy)

    with pytest.raises(RuntimeError, match="simulated setup copy failure"):
        modal_app._resume_action(source_run_id, attempt_run_id, seed=1)

    attempt = tmp_path / "runs" / attempt_run_id
    manifest = load_artifact_manifest(attempt / "artifact_manifest.json")
    assert verify_artifact_manifest(attempt, manifest)["verified"] is True
    result = json.loads(
        (attempt / "resume_action_result.json").read_text(encoding="utf-8")
    )
    assert result == {
        "error_type": "RuntimeError",
        "mode": "checkpoint_resume",
        "source_run_id": source_run_id,
        "success": False,
    }
    assert not (attempt / "candidate_smoke").exists()
    assert not (attempt / "provider_free_network_denial_probe.json").exists()
    assert _source_snapshot(source) == source_before
    assert volume.commits == 2


def test_volume_manifest_loader_falls_back_only_when_final_is_absent(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    source_run_id, source, _training = _source_candidate_run(tmp_path, modal_app)
    checkpoint_payload = (
        source / "artifact_manifest.checkpoint.json"
    ).read_bytes()
    calls: list[str] = []

    class CheckpointOnlyVolume:
        def read_file(self, path: str):
            calls.append(path)
            if path.endswith("/artifact_manifest.json"):
                raise FileNotFoundError(path)
            midpoint = len(checkpoint_payload) // 2
            return iter(
                (checkpoint_payload[:midpoint], checkpoint_payload[midpoint:])
            )

    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", CheckpointOnlyVolume())

    manifest = modal_app._load_volume_manifest(source_run_id)

    assert manifest.manifest.run_id == source_run_id
    assert calls == [
        f"/runs/{source_run_id}/artifact_manifest.json",
        f"/runs/{source_run_id}/artifact_manifest.checkpoint.json",
    ]


def test_volume_manifest_loader_never_downgrades_an_invalid_final_manifest(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    source_run_id, _source, _training = _source_candidate_run(tmp_path, modal_app)
    calls: list[str] = []

    class InvalidFinalVolume:
        def read_file(self, path: str):
            calls.append(path)
            if path.endswith("/artifact_manifest.checkpoint.json"):
                raise FileNotFoundError(path)
            return iter((b"{invalid-final",))

    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", InvalidFinalVolume())

    with pytest.raises(ModalBoundaryError, match="manifest is invalid"):
        modal_app._load_volume_manifest(source_run_id)

    assert calls == [
        f"/runs/{source_run_id}/artifact_manifest.json",
        f"/runs/{source_run_id}/artifact_manifest.checkpoint.json",
    ]


def test_volume_manifest_loader_rejects_final_and_checkpoint_mixed_state(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    source_run_id, source, _training = _source_candidate_run(tmp_path, modal_app)
    payload = (source / "artifact_manifest.checkpoint.json").read_bytes()
    calls: list[str] = []

    class MixedManifestVolume:
        def read_file(self, path: str):
            calls.append(path)
            return iter((payload,))

    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", MixedManifestVolume())

    with pytest.raises(ModalBoundaryError, match="exactly one"):
        modal_app._load_volume_manifest(source_run_id)

    assert calls == [
        f"/runs/{source_run_id}/artifact_manifest.json",
        f"/runs/{source_run_id}/artifact_manifest.checkpoint.json",
    ]
