from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from common.device import AcceleratorFingerprint
from common.runtime_context import ExecutionContextV1
from common.task_adapter import DEFAULT_TASK
from common.trainer import (
    _best_evaluation_checkpoint_payload,
    _checkpoint_payload,
    _dependency_lock_hash,
    rng_state_sha256,
    sha256_file,
    training_manifest,
)
from common.training_config import (
    SMOKE_TRAIN_CUDA_V2,
    TrainingResult,
    TrainingSeedBundle,
)
from containment.audit import audit_runtime
from containment.policy import (
    CandidateFormat,
    ScientificExecutionRequest,
    assess_scientific_execution,
)
from modal_boundary import (
    APP_NAME,
    ImageSourceManifestV1,
    SourceFileV1,
    volume_artifact_uri,
)
from scripts import verify_resume_progression as progression_module
from scripts.verify_resume_progression import (
    EVIDENCE_FIELDS,
    ResumeProgressionVerificationError,
    verify_resume_progression,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_SEED = 41


def _resume_checkpoint(
    *,
    step: int,
    candidate_hash: str,
    seeds: TrainingSeedBundle,
) -> dict:
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _step: 1.0,
    )
    inputs = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    for _ in range(step):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
        scheduler.step()
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        profile=SMOKE_TRAIN_CUDA_V2,
        candidate_hash=candidate_hash,
        task=DEFAULT_TASK,
        seeds=seeds,
        step=step,
        examples_processed=step * SMOKE_TRAIN_CUDA_V2.global_batch_size,
        elapsed_seconds=float(step),
        best_step=step,
        best_accuracy=0.0,
        best_loss=1.0,
        final_training_loss=1.0,
    )
    payload["rng_state"]["torch_mps"] = None
    payload["rng_state"]["torch_cuda"] = [
        torch.tensor([step, step + 1, step + 2], dtype=torch.uint8)
    ]
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    artifact_root = tmp_path / "resume-run"
    # The production action uses the frozen seed_1 path.  Keep RUN_SEED as the
    # logical seed below but bind the portable action roster exactly.
    training = artifact_root / "candidate_smoke" / "seed_1"
    training.mkdir(parents=True)
    candidate = training / "candidate_graph.json"
    candidate.write_bytes(
        (ROOT / "common" / "initial_candidate.ir.json").read_bytes()
    )
    candidate_hash = sha256_file(candidate)
    seeds = TrainingSeedBundle.from_run_seed(RUN_SEED)
    nonterminal = training / "partial_resume_checkpoint.pt"
    resumed = training / "latest_resume_checkpoint.pt"
    partial_payload = _resume_checkpoint(
        step=SMOKE_TRAIN_CUDA_V2.checkpoint_interval,
        candidate_hash=candidate_hash,
        seeds=seeds,
    )
    latest_payload = _resume_checkpoint(
        step=SMOKE_TRAIN_CUDA_V2.max_steps,
        candidate_hash=candidate_hash,
        seeds=seeds,
    )
    torch.save(partial_payload, nonterminal)
    torch.save(latest_payload, resumed)
    best = training / "best_checkpoint.pt"
    best_model = torch.nn.Linear(2, 2)
    best_model.load_state_dict(latest_payload["model_state"])
    torch.save(
        _best_evaluation_checkpoint_payload(
            model=best_model,
            profile=SMOKE_TRAIN_CUDA_V2,
            candidate_hash=candidate_hash,
            task=DEFAULT_TASK,
            seeds=seeds,
            step=SMOKE_TRAIN_CUDA_V2.max_steps,
            examples_processed=(
                SMOKE_TRAIN_CUDA_V2.max_steps
                * SMOKE_TRAIN_CUDA_V2.global_batch_size
            ),
            best_accuracy=0.0,
            best_loss=1.0,
        ),
        best,
    )
    events = training / "training_events.jsonl"
    events.write_text(
        "".join(
            json.dumps(
                {
                    "timestamp": f"2026-08-09T00:00:{step:02d}+00:00",
                    "checkpoint_decision": (
                        "best_development"
                        if step == SMOKE_TRAIN_CUDA_V2.max_steps
                        else "none"
                    ),
                    "elapsed_seconds": float(step),
                    "examples_processed": (
                        step * SMOKE_TRAIN_CUDA_V2.global_batch_size
                    ),
                    "loss": 1.0,
                    "learning_rate": 0.001,
                    "gradient_norm": 1.0,
                    "optimizer_step": step,
                    "validation_loss": (
                        1.0 if step == SMOKE_TRAIN_CUDA_V2.max_steps else None
                    ),
                    "validation_exact_match_accuracy": (
                        0.0 if step == SMOKE_TRAIN_CUDA_V2.max_steps else None
                    ),
                    "current_accelerator_allocated_bytes": 2048,
                    "reserved_accelerator_allocated_bytes": 4096,
                    "peak_accelerator_allocated_bytes": 4096,
                    "accelerator_total_memory_bytes": 16_000_000_000,
                },
                sort_keys=True,
            )
            + "\n"
            for step in range(1, SMOKE_TRAIN_CUDA_V2.max_steps + 1)
        ),
        encoding="utf-8",
    )
    fingerprint = AcceleratorFingerprint(
        requested_device="cuda",
        selected_device="cuda:0",
        accelerator_kind="cuda",
        gpu_name="NVIDIA T4",
        gpu_count=1,
        compute_capability="7.5",
        cuda_runtime="12.8",
        cuda_driver="550.54.15",
        torch_version="2.7.1+cu128",
        host_platform="Linux-test",
    )
    dependency_lock_hash = _dependency_lock_hash()
    image_manifest = ImageSourceManifestV1(
        dependency_lock_sha256=dependency_lock_hash,
        files=(
            SourceFileV1(
                relative_path="common/initial_candidate.ir.json",
                sha256=candidate_hash,
                size_bytes=candidate.stat().st_size,
            ),
        ),
    )
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=artifact_root.name,
        app_name=APP_NAME,
        function_name="checkpoint_resume",
        modal_app_id="ap-resume123",
        modal_function_id="fu-resume123",
        modal_call_id="fc-resume123",
        modal_image_id="im-resume123",
        image_source_sha256=image_manifest.manifest_sha256,
        artifact_uri=volume_artifact_uri(artifact_root.name),
    )
    _write_json(
        artifact_root / "resume_execution_context.json",
        context.to_dict(),
    )
    _write_json(
        artifact_root / "image_source_manifest.json",
        image_manifest.to_dict(),
    )
    audit = audit_runtime(environment={})
    decision = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARCHITECTURE_IR,
            requested_device="cuda",
            required_accelerator="cuda",
            scientific=False,
            ir_validated=True,
            trusted_ir_interpreter=True,
            candidate_artifact_hash=candidate_hash,
        ),
    )
    manifest = training_manifest(
        candidate_path=candidate,
        candidate_hash=candidate_hash,
        profile=SMOKE_TRAIN_CUDA_V2,
        seeds=seeds,
        requested_device="cuda",
        selected_device="cuda:0",
        task=DEFAULT_TASK,
        allow_cpu_for_tests=False,
        containment_audit=audit.to_dict(),
        containment_decision=decision.to_dict(),
        candidate_format=CandidateFormat.ARCHITECTURE_IR,
        candidate_graph_hash=(
            progression_module.validate_ir_candidate_json(
                candidate.read_text(encoding="utf-8")
            ).graph_hash
        ),
        accelerator_fingerprint=fingerprint.to_dict(),
        execution_context=context,
        dependency_lock_hash=dependency_lock_hash,
    )
    manifest["runtime"].update(
        {
            "mps_built": False,
            "mps_available": False,
            "cuda_runtime": "12.8",
            "cuda_available": True,
            "cuda_device_count": 1,
            "deterministic_algorithms": True,
            "pytorch_enable_mps_fallback": "0",
            "cublas_workspace_config": ":4096:8",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "accelerator_fingerprint": fingerprint.to_dict(),
        }
    )
    manifest_path = training / "training_manifest.json"
    _write_json(manifest_path, manifest)
    summary = TrainingResult(
        success=True,
        failure_stage="",
        error="",
        profile_name=SMOKE_TRAIN_CUDA_V2.name,
        profile_version=SMOKE_TRAIN_CUDA_V2.version,
        profile_hash=SMOKE_TRAIN_CUDA_V2.profile_hash,
        candidate_source_hash=candidate_hash,
        initialization_seed=seeds.model_initialization_seed,
        data_seed=seeds.training_data_seed,
        development_seed=seeds.development_set_seed,
        dataloader_seed=seeds.dataloader_seed,
        device="cuda:0",
        dtype=SMOKE_TRAIN_CUDA_V2.dtype,
        steps_completed=SMOKE_TRAIN_CUDA_V2.max_steps,
        examples_processed=(
            SMOKE_TRAIN_CUDA_V2.max_steps
            * SMOKE_TRAIN_CUDA_V2.global_batch_size
        ),
        best_development_step=SMOKE_TRAIN_CUDA_V2.max_steps,
        best_development_exact_match_accuracy=0.0,
        best_development_loss=1.0,
        final_training_loss=1.0,
        train_seconds=10.0,
        accelerator_kind="cuda",
        peak_accelerator_allocated_bytes=4096,
        current_accelerator_allocated_bytes=2048,
        reserved_accelerator_allocated_bytes=4096,
        accelerator_total_memory_bytes=16_000_000_000,
        accelerator_fingerprint=fingerprint.to_dict(),
        parameter_count_metadata=6,
        checkpoint_path=best.name,
        checkpoint_sha256=sha256_file(best),
        event_log_path=events.name,
        unsupported_operation_fallback=False,
        scientific=False,
        hardware_matched=True,
        cleanup_completed=True,
        schema_version="2.0",
    ).to_dict()
    summary_path = training / "training_summary.json"
    _write_json(summary_path, summary)
    attestation = {
        "schema_name": "RNGRestoreAttestation",
        "schema_version": "1.0",
        "source_checkpoint_sha256": sha256_file(nonterminal),
        "source_rng_state_sha256": rng_state_sha256(partial_payload["rng_state"]),
        "observed_post_restore_rng_state_sha256": rng_state_sha256(
            partial_payload["rng_state"]
        ),
        "restored_exactly": True,
        "source_optimizer_step": SMOKE_TRAIN_CUDA_V2.checkpoint_interval,
        "source_examples_processed": (
            SMOKE_TRAIN_CUDA_V2.checkpoint_interval
            * SMOKE_TRAIN_CUDA_V2.global_batch_size
        ),
        "final_checkpoint_sha256": sha256_file(resumed),
        "final_rng_state_sha256": rng_state_sha256(latest_payload["rng_state"]),
        "final_optimizer_step": SMOKE_TRAIN_CUDA_V2.max_steps,
        "final_examples_processed": (
            SMOKE_TRAIN_CUDA_V2.max_steps
            * SMOKE_TRAIN_CUDA_V2.global_batch_size
        ),
        "rng_progressed": True,
        "execution_context": context.to_dict(),
    }
    attestation_path = training / "rng_restore_attestation.json"
    _write_json(attestation_path, attestation)
    return {
        "artifact_root": artifact_root,
        "training": training,
        "candidate": candidate,
        "nonterminal": nonterminal,
        "resumed": resumed,
        "best": best,
        "events": events,
        "summary": summary_path,
        "manifest": manifest_path,
        "attestation": attestation_path,
        "output": artifact_root / "resume_progression_verification.json",
    }


def _verify(paths: dict[str, Path]) -> dict:
    return verify_resume_progression(
        artifact_root=paths["artifact_root"],
        training_output_dir=paths["training"],
        profile_name=SMOKE_TRAIN_CUDA_V2.name,
        run_seed=RUN_SEED,
        output_path=paths["output"],
    )


def test_resume_progression_binds_completion_with_portable_hash_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    original_load = torch.load
    loads: list[tuple[Path, dict]] = []

    def recorded_load(path, **kwargs):
        loads.append((Path(path), kwargs))
        return original_load(path, **kwargs)

    monkeypatch.setattr(progression_module.torch, "load", recorded_load)
    nonterminal_bytes = paths["nonterminal"].read_bytes()

    evidence = _verify(paths)

    assert set(evidence) == EVIDENCE_FIELDS
    assert evidence["schema_name"] == "ResumeProgressionEvidence"
    assert evidence["schema_version"] == "1.0"
    assert evidence["training_output_relative_path"] == "candidate_smoke/seed_1"
    assert evidence["evidence_relative_path"] == (
        "resume_progression_verification.json"
    )
    assert evidence["progression"]["nonterminal_optimizer_step"] == 5
    assert evidence["progression"]["resumed_optimizer_step"] == 10
    assert evidence["progression"]["events_after_nonterminal_checkpoint"] == 5
    assert evidence["progression"]["resumed_examples_processed"] == 160
    assert all(evidence["checks"].values())
    assert paths["nonterminal"].read_bytes() == nonterminal_bytes
    assert len(loads) == 3
    assert {path.name for path, _kwargs in loads} == {
        "partial_resume_checkpoint.pt",
        "latest_resume_checkpoint.pt",
        "best_checkpoint.pt",
    }
    assert all(
        kwargs == {"map_location": "cpu", "weights_only": True}
        for _path, kwargs in loads
    )
    assert paths["output"].read_text(encoding="utf-8") == (
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    serialized = json.dumps(evidence, sort_keys=True)
    assert str(tmp_path) not in serialized
    for artifact in evidence["artifacts"].values():
        assert not Path(artifact["relative_path"]).is_absolute()
        assert len(artifact["sha256"]) == 64


@pytest.mark.parametrize(
    "field",
    (
        "candidate_source_hash",
        "profile_hash",
        "seed_bundle_hash",
        "dependency_lock_hash",
    ),
)
def test_resume_progression_rejects_changed_checkpoint_identity(
    tmp_path,
    field,
) -> None:
    paths = _fixture(tmp_path)
    latest = torch.load(paths["resumed"], map_location="cpu", weights_only=True)
    latest[field] = "0" * 64
    torch.save(latest, paths["resumed"])

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="exact v2 identity contract|identity changed",
    ):
        _verify(paths)

    assert not paths["output"].exists()


def test_resume_progression_rejects_terminal_checkpoint_without_advancement(
    tmp_path,
) -> None:
    paths = _fixture(tmp_path)
    paths["resumed"].write_bytes(paths["nonterminal"].read_bytes())

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="did not advance",
    ):
        _verify(paths)

    assert not paths["output"].exists()


def test_resume_progression_rejects_noncontiguous_optimizer_events(tmp_path) -> None:
    paths = _fixture(tmp_path)
    records = [
        json.loads(line)
        for line in paths["events"].read_text(encoding="utf-8").splitlines()
    ]
    records[5]["optimizer_step"] = 7
    paths["events"].write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="event chain is not contiguous",
    ):
        _verify(paths)

    assert not paths["output"].exists()


def test_resume_progression_rejects_absolute_summary_artifact_path(tmp_path) -> None:
    paths = _fixture(tmp_path)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["event_log_path"] = str(paths["events"].resolve())
    _write_json(paths["summary"], summary)

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="non-portable absolute path fields.*event_log_path",
    ):
        _verify(paths)

    assert not paths["output"].exists()


def test_resume_progression_refuses_existing_evidence_destination(tmp_path) -> None:
    paths = _fixture(tmp_path)
    paths["output"].write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="destination already exists",
    ):
        _verify(paths)

    assert paths["output"].read_text(encoding="utf-8") == "operator-owned\n"


def test_resume_progression_rejects_duplicate_json_fields(tmp_path) -> None:
    paths = _fixture(tmp_path)
    raw = paths["summary"].read_text(encoding="utf-8")
    raw = raw.replace(
        '"steps_completed": 10',
        '"steps_completed": 10, "steps_completed": 10',
        1,
    )
    paths["summary"].write_text(raw, encoding="utf-8")

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="not canonical JSON evidence",
    ):
        _verify(paths)

    assert not paths["output"].exists()


@pytest.mark.parametrize(
    ("artifact", "mutation", "message"),
    (
        ("summary", "summary_extra", "summary fields differ"),
        ("manifest", "manifest_extra", "manifest fields differ"),
        ("events", "event_extra", "event line 1 differs"),
        ("best", "best_extra", "checkpoint fields differ"),
        ("attestation", "attestation_extra", "attestation fields differ"),
    ),
)
def test_resume_progression_rejects_schema_extensions(
    tmp_path,
    artifact,
    mutation,
    message,
) -> None:
    paths = _fixture(tmp_path)
    path = paths[artifact]
    if mutation == "event_extra":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        records[0]["unexpected"] = 1
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
    elif mutation == "best_extra":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["unexpected"] = 1
        torch.save(payload, path)
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        summary["checkpoint_sha256"] = sha256_file(path)
        _write_json(paths["summary"], summary)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unexpected"] = 1
        _write_json(path, payload)

    with pytest.raises(ResumeProgressionVerificationError, match=message):
        _verify(paths)

    assert not paths["output"].exists()


@pytest.mark.parametrize(
    ("artifact", "field", "message"),
    (
        ("summary", "steps_completed", "must be an integer"),
        ("manifest", "runtime.cuda_device_count", "must be an integer"),
        ("events", "examples_processed", "must be an integer"),
        ("best", "global_step", "must be an integer"),
        ("attestation", "source_optimizer_step", "must be an integer"),
    ),
)
def test_resume_progression_rejects_boolean_as_integer(
    tmp_path,
    artifact,
    field,
    message,
) -> None:
    paths = _fixture(tmp_path)
    path = paths[artifact]
    if artifact == "events":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        records[0][field] = True
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
    elif artifact == "best":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload[field] = True
        torch.save(payload, path)
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        summary["checkpoint_sha256"] = sha256_file(path)
        _write_json(paths["summary"], summary)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "." in field:
            parent, child = field.split(".", 1)
            payload[parent][child] = True
        else:
            payload[field] = True
        _write_json(path, payload)

    with pytest.raises(ResumeProgressionVerificationError, match=message):
        _verify(paths)

    assert not paths["output"].exists()


def test_resume_progression_rejects_credential_and_absolute_path_fields(
    tmp_path,
) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["candidate_path"] = "/mnt/discovery/private/candidate_graph.json"
    _write_json(paths["manifest"], manifest)

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="non-portable absolute path fields",
    ):
        _verify(paths)

    paths = _fixture(tmp_path / "second")
    attestation = json.loads(paths["attestation"].read_text(encoding="utf-8"))
    attestation["api_key"] = "redacted"
    _write_json(paths["attestation"], attestation)

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="credential-shaped fields",
    ):
        _verify(paths)


def test_resume_progression_rejects_mixed_context_and_image_bindings(tmp_path) -> None:
    paths = _fixture(tmp_path)
    context_path = paths["artifact_root"] / "resume_execution_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["modal_call_id"] = "fc-other123"
    _write_json(context_path, context)

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="outer run context",
    ):
        _verify(paths)

    paths = _fixture(tmp_path / "second")
    image_path = paths["artifact_root"] / "image_source_manifest.json"
    image = json.loads(image_path.read_text(encoding="utf-8"))
    image["dependency_lock_sha256"] = "0" * 64
    _write_json(image_path, image)

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="dependency lock differs",
    ):
        _verify(paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_cuda", "invalid exact CUDA RNG schema"),
        ("bool_numpy_position", "contains invalid state values"),
        ("stale_attestation_hash", "differs from persisted evidence"),
        ("no_rng_progression", "did not progress"),
    ),
)
def test_resume_progression_rejects_invalid_rng_schema_and_lineage(
    tmp_path,
    mutation,
    message,
) -> None:
    paths = _fixture(tmp_path)
    if mutation in {"missing_cuda", "bool_numpy_position"}:
        initial = torch.load(
            paths["nonterminal"],
            map_location="cpu",
            weights_only=True,
        )
        if mutation == "missing_cuda":
            del initial["rng_state"]["torch_cuda"]
        else:
            initial["rng_state"]["numpy"]["position"] = True
        torch.save(initial, paths["nonterminal"])
    elif mutation == "stale_attestation_hash":
        attestation = json.loads(paths["attestation"].read_text(encoding="utf-8"))
        attestation["observed_post_restore_rng_state_sha256"] = "0" * 64
        _write_json(paths["attestation"], attestation)
    else:
        initial = torch.load(
            paths["nonterminal"],
            map_location="cpu",
            weights_only=True,
        )
        latest = torch.load(paths["resumed"], map_location="cpu", weights_only=True)
        latest["rng_state"] = initial["rng_state"]
        torch.save(latest, paths["resumed"])
        attestation = json.loads(paths["attestation"].read_text(encoding="utf-8"))
        attestation["final_checkpoint_sha256"] = sha256_file(paths["resumed"])
        attestation["final_rng_state_sha256"] = rng_state_sha256(
            latest["rng_state"]
        )
        _write_json(paths["attestation"], attestation)

    with pytest.raises(ResumeProgressionVerificationError, match=message):
        _verify(paths)

    assert not paths["output"].exists()


def test_resume_progression_rejects_extra_and_symlinked_action_artifacts(
    tmp_path,
) -> None:
    paths = _fixture(tmp_path)
    (paths["training"] / "debug.log").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="artifact roster differs",
    ):
        _verify(paths)

    paths = _fixture(tmp_path / "second")
    paths["manifest"].unlink()
    paths["manifest"].symlink_to(paths["summary"])

    with pytest.raises(
        ResumeProgressionVerificationError,
        match="regular file",
    ):
        _verify(paths)
