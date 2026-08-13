from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
import yaml
from architecture_ir import load_and_build_ir_candidate, validate_ir_candidate_json
from common.candidate_artifact import build_candidate_artifact
from common.descriptor_schema import CATEGORY_CODES, SEMANTIC_METRIC_NAMES
from common.gpt56_sol import (
    API_MODE,
    OFFICIAL_OPENAI_API_BASE,
    TARGET_MODEL,
    GPT56SolProfile,
    resolve_provider_endpoint,
)
from common.provider_attempts import (
    PROVIDER_ATTEMPT_LEDGER_FILENAME,
    PROVIDER_ATTEMPT_SCHEMA,
    ProviderAttemptRecord,
    generation_settings_sha256,
)
from common.runtime_context import ExecutionContextV1
from common.task_adapter import DEFAULT_TASK
from common.trainer import (
    _best_evaluation_checkpoint_payload,
    _checkpoint_payload,
    _dependency_lock_hash,
    training_manifest,
)
from common.training_config import (
    SMOKE_TRAIN_CUDA_V2,
    SMOKE_TRAIN_V1,
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
    CANARY_ORDER,
    IMAGE_RECIPE_VERSION,
    MODAL_VERSION,
    PYTHON_VERSION,
    UV_VERSION,
    ImageSourceManifestV1,
    ModalLiveCohortIdentity,
    SourceFileV1,
    build_artifact_manifest,
    canonical_sha256,
    volume_artifact_uri,
    write_artifact_manifest,
)
from scripts.validate_engineering_canaries import (
    _MODAL_CANARY_GENERATOR_CONTRACT,
    HARNESSES,
    MAX_FAKE_RESPONSE_BYTES,
    DeterministicFakeProvider,
    build_report,
    create_modal_canary_selector,
    load_modal_canary_selector,
    validate_controller_surfaces,
    validate_downloaded_modal_canaries,
    validate_existing_cuda_smoke,
    validate_existing_mps_smoke,
    validate_private_canary_staging,
)
from scripts.validate_engineering_canaries import (
    main as validator_main,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _selector_identity(
    download_root: Path,
    prefix: str,
    *,
    cohort_id: str,
) -> ModalLiveCohortIdentity:
    run_id = _selected_canary_run_ids(download_root, prefix)[CANARY_ORDER[0]]
    context = json.loads(
        (download_root / run_id / "execution_context.json").read_text(
            encoding="utf-8"
        )
    )
    return ModalLiveCohortIdentity(
        source_tree_sha256="9" * 64,
        image_source_sha256=context["image_source_sha256"],
        cohort_id=cohort_id,
    )


def _project_fixture(
    tmp_path: Path,
    *,
    omit: str | None = None,
    active_cuda: bool = False,
) -> Path:
    project = tmp_path / "project"
    (project / "common").mkdir(parents=True)
    shutil.copy2(
        ROOT / "common" / "initial_candidate.ir.json",
        project / "common" / "initial_candidate.ir.json",
    )
    shutil.copy2(
        ROOT / "common" / "initial_candidate.py",
        project / "common" / "initial_candidate.py",
    )
    (project / "common" / "openevolve_runner.py").write_text(
        """import argparse

def run_controller(kind):
    parser = argparse.ArgumentParser()
    parser.add_argument('--engineering-pilot', action='store_true')
    parser.add_argument('--iterations')
    parser.add_argument('--seed')
    parser.add_argument('--output-dir')
    parser.add_argument('--training-profile')
    parser.add_argument('--evaluation-profile')
    parser.add_argument('--evaluation-cases')
    parser.add_argument('--device')
    return parser.parse_args()
""",
        encoding="utf-8",
    )
    for spec in HARNESSES:
        if spec.harness_id == omit:
            continue
        agent = project / "agents" / spec.agent_directory
        agent.mkdir(parents=True)
        if spec.delegated_controller_kind is None:
            entrypoint = """import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--engineering-pilot', action='store_true')
parser.add_argument('--iterations')
parser.add_argument('--seed')
parser.add_argument('--output-dir')
parser.add_argument('--training-profile')
parser.add_argument('--evaluation-profile')
parser.add_argument('--evaluation-cases')
parser.add_argument('--device')
if __name__ == '__main__':
    parser.parse_args()
"""
        else:
            entrypoint = f"""from common.openevolve_runner import run_controller

if __name__ == '__main__':
    run_controller({spec.delegated_controller_kind!r})
"""
        (agent / "run.py").write_text(entrypoint, encoding="utf-8")
        (agent / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "condition": spec.config_condition or spec.harness_id,
                    "training": {
                        "profile": (
                            "full_train_cuda_v2" if active_cuda else "full_train_v1"
                        ),
                        "profile_version": "2" if active_cuda else "1",
                        "device": "cuda" if active_cuda else "mps",
                        "allow_cpu_for_tests": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        (agent / "program.md").write_text(
            f"# {spec.display_name}\n\nFixture prompt.\n",
            encoding="utf-8",
        )
    return project


def _synthetic_mps_smoke(project: Path, tmp_path: Path) -> Path:
    output = tmp_path / "mps-smoke"
    output.mkdir()
    candidate = project / "common" / "initial_candidate.ir.json"
    stored_candidate = output / "candidate_graph.json"
    shutil.copy2(candidate, stored_candidate)

    initialization_seed = 17
    interpreted = load_and_build_ir_candidate(candidate, initialization_seed)
    model = interpreted.model
    trained_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    changed_name = next(
        name for name, value in trained_state.items() if value.is_floating_point()
    )
    trained_state[changed_name] = trained_state[changed_name] + 0.001
    checkpoint = output / "best_checkpoint.pt"
    torch.save({"model_state": trained_state}, checkpoint)

    events = output / "training_events.jsonl"
    events.write_text(
        "".join(
            json.dumps(
                {
                    "optimizer_step": step,
                    "loss": 1.0 / step,
                },
                sort_keys=True,
            )
            + "\n"
            for step in range(1, SMOKE_TRAIN_V1.max_steps + 1)
        ),
        encoding="utf-8",
    )
    candidate_hash = _sha256(candidate)
    candidate_validation = validate_ir_candidate_json(
        candidate.read_text(encoding="utf-8")
    )
    assert candidate_validation.valid
    manifest = {
        "allow_cpu_for_tests": False,
        "hardware_matched_scientific_run": False,
        "candidate_source_hash": candidate_hash,
        "candidate_artifact_hash": candidate_hash,
        "candidate_format": "architecture_ir",
        "candidate_graph_hash": candidate_validation.graph_hash,
        "profile_hash": SMOKE_TRAIN_V1.profile_hash,
        "requested_device": "mps",
        "selected_device": "mps",
        "parameter_count_role": "descriptive_metadata_only",
        "isolation_level": "engineering_only_or_scientific_gate_blocked",
        "runtime": {
            "mps_built": True,
            "mps_available": True,
            "pytorch_enable_mps_fallback": "0",
        },
        "containment_decision": {"allowed": True, "scientific": False},
        "containment_audit": {"visible_credential_names": []},
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    summary = {
        "success": True,
        "scientific": False,
        "hardware_matched": True,
        "unsupported_operation_fallback": False,
        "cleanup_completed": True,
        "profile_name": SMOKE_TRAIN_V1.name,
        "profile_version": SMOKE_TRAIN_V1.version,
        "profile_hash": SMOKE_TRAIN_V1.profile_hash,
        "candidate_source_hash": candidate_hash,
        "device": "mps",
        "dtype": "float32",
        "steps_completed": SMOKE_TRAIN_V1.max_steps,
        "examples_processed": (
            SMOKE_TRAIN_V1.max_steps * SMOKE_TRAIN_V1.global_batch_size
        ),
        "best_development_loss": 0.1,
        "final_training_loss": 0.1,
        "train_seconds": 1.0,
        "initialization_seed": initialization_seed,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "event_log_path": str(events),
    }
    (output / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return output


def _synthetic_pre_ir_mps_smoke(project: Path, tmp_path: Path) -> Path:
    """Reproduce the retained v1 arbitrary-Python artifact layout."""

    output = tmp_path / "pre-ir-mps-smoke"
    output.mkdir()
    candidate = project / "common" / "initial_candidate.py"
    stored_candidate = output / "candidate_source.py"
    shutil.copy2(candidate, stored_candidate)

    initialization_seed = 4541374973981895479
    model = build_candidate_artifact(candidate, seed=initialization_seed).model
    trained_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    changed_name = next(
        name for name, value in trained_state.items() if value.is_floating_point()
    )
    trained_state[changed_name] = trained_state[changed_name] + 0.001
    candidate_hash = _sha256(stored_candidate)
    checkpoint = output / "best_checkpoint.pt"
    torch.save(
        {
            "checkpoint_kind": "best_evaluation_weights_v1",
            "model_state": trained_state,
            "global_step": SMOKE_TRAIN_V1.max_steps,
            "examples_processed": (
                SMOKE_TRAIN_V1.max_steps * SMOKE_TRAIN_V1.global_batch_size
            ),
            "candidate_source_hash": candidate_hash,
            "profile_hash": SMOKE_TRAIN_V1.profile_hash,
        },
        checkpoint,
    )

    events = output / "training_events.jsonl"
    events.write_text(
        "".join(
            json.dumps({"optimizer_step": step, "loss": 1.0 / step}) + "\n"
            for step in range(1, SMOKE_TRAIN_V1.max_steps + 1)
        ),
        encoding="utf-8",
    )
    manifest = {
        "allow_cpu_for_tests": False,
        "hardware_matched_scientific_run": False,
        # Historical records omitted candidate_format and the IR-only fields.
        "candidate_source_hash": candidate_hash,
        "profile": SMOKE_TRAIN_V1.to_dict(),
        "profile_hash": SMOKE_TRAIN_V1.profile_hash,
        "requested_device": "mps",
        "selected_device": "mps",
        "parameter_count_role": "descriptive_metadata_only",
        "isolation_level": "engineering_only_or_scientific_gate_blocked",
        "runtime": {
            "mps_built": True,
            "mps_available": True,
            "pytorch_enable_mps_fallback": "0",
        },
        "containment_decision": {
            "allowed": True,
            "scientific": False,
            "candidate_format": "arbitrary_python",
        },
        "containment_audit": {"visible_credential_names": []},
    }
    _write_json(output / "training_manifest.json", manifest)
    summary = {
        "success": True,
        "scientific": False,
        "hardware_matched": True,
        "unsupported_operation_fallback": False,
        "cleanup_completed": True,
        "profile_name": SMOKE_TRAIN_V1.name,
        "profile_version": SMOKE_TRAIN_V1.version,
        "profile_hash": SMOKE_TRAIN_V1.profile_hash,
        "candidate_source_hash": candidate_hash,
        "device": "mps",
        "dtype": "float32",
        "steps_completed": SMOKE_TRAIN_V1.max_steps,
        "examples_processed": (
            SMOKE_TRAIN_V1.max_steps * SMOKE_TRAIN_V1.global_batch_size
        ),
        "best_development_loss": 0.1,
        "final_training_loss": 0.1,
        "train_seconds": 1.0,
        "initialization_seed": initialization_seed,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "event_log_path": str(events),
    }
    _write_json(output / "training_summary.json", summary)
    return output


def _synthetic_cuda_smoke(project: Path, tmp_path: Path) -> Path:
    run_root = tmp_path / "cuda-smoke-run"
    output = run_root / "candidate_smoke" / "seed_1"
    output.mkdir(parents=True)
    candidate = project / "common" / "initial_candidate.ir.json"
    stored_candidate = output / "candidate_graph.json"
    shutil.copy2(candidate, stored_candidate)

    seeds = TrainingSeedBundle.from_run_seed(1)
    initialization_seed = seeds.model_initialization_seed
    candidate_hash = _sha256(stored_candidate)
    dependency_lock_hash = _dependency_lock_hash()
    candidate_validation = validate_ir_candidate_json(
        candidate.read_text(encoding="utf-8")
    )
    assert candidate_validation.valid

    def trained_checkpoint(step: int) -> tuple[torch.nn.Module, dict]:
        interpreted = load_and_build_ir_candidate(candidate, initialization_seed)
        model = interpreted.model
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda _step: 1.0,
        )
        for _ in range(step):
            optimizer.zero_grad(set_to_none=True)
            loss = sum(parameter.square().mean() for parameter in model.parameters())
            loss.backward()
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
            dependency_lock_hash=dependency_lock_hash,
        )
        payload["rng_state"]["torch_mps"] = None
        payload["rng_state"]["torch_cuda"] = [
            torch.tensor([step, step + 1, step + 2], dtype=torch.uint8)
        ]
        return model, payload

    _partial_model, partial_payload = trained_checkpoint(
        SMOKE_TRAIN_CUDA_V2.checkpoint_interval
    )
    final_model, latest_payload = trained_checkpoint(SMOKE_TRAIN_CUDA_V2.max_steps)
    torch.save(partial_payload, output / "partial_resume_checkpoint.pt")
    torch.save(latest_payload, output / "latest_resume_checkpoint.pt")
    checkpoint = output / "best_checkpoint.pt"
    torch.save(
        _best_evaluation_checkpoint_payload(
            model=final_model,
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
            dependency_lock_hash=dependency_lock_hash,
        ),
        checkpoint,
    )

    events = output / "training_events.jsonl"
    events.write_text(
        "".join(
            json.dumps(
                {
                    "timestamp": f"2026-08-09T00:00:{step:02d}+00:00",
                    "optimizer_step": step,
                    "examples_processed": (
                        step * SMOKE_TRAIN_CUDA_V2.global_batch_size
                    ),
                    "loss": 1.0 / step,
                    "learning_rate": 0.001,
                    "gradient_norm": 1.0,
                    "validation_loss": 1.0 if step == 10 else None,
                    "validation_exact_match_accuracy": 0.0 if step == 10 else None,
                    "elapsed_seconds": float(step),
                    "current_accelerator_allocated_bytes": 1024,
                    "reserved_accelerator_allocated_bytes": 2048,
                    "peak_accelerator_allocated_bytes": 1536,
                    "accelerator_total_memory_bytes": 16_000_000_000,
                    "checkpoint_decision": (
                        "best_development" if step == 10 else "none"
                    ),
                },
                sort_keys=True,
            )
            + "\n"
            for step in range(1, SMOKE_TRAIN_CUDA_V2.max_steps + 1)
        ),
        encoding="utf-8",
    )
    fingerprint = {
        "requested_device": "cuda",
        "selected_device": "cuda:0",
        "accelerator_kind": "cuda",
        "gpu_name": "NVIDIA T4",
        "gpu_count": 1,
        "compute_capability": "7.5",
        "cuda_runtime": "12.8",
        "cuda_driver": "550.54",
        "torch_version": "2.7.1+cu128",
        "host_platform": "Linux-6.8-x86_64",
    }
    execution_context = ExecutionContextV1(
        execution_backend="modal",
        run_id="cuda-smoke-run",
        app_name="rl4rl-architecture-discovery",
        function_name="candidate_smoke",
        modal_app_id="ap-candidate123",
        modal_function_id="fu-candidate123",
        modal_call_id="fc-call123",
        modal_image_id="im-candidate123",
        image_source_sha256="a" * 64,
        artifact_uri=volume_artifact_uri("cuda-smoke-run"),
    )
    source_record = SourceFileV1(
        relative_path="common/initial_candidate.ir.json",
        sha256=candidate_hash,
        size_bytes=stored_candidate.stat().st_size,
    )
    image_manifest = ImageSourceManifestV1(
        dependency_lock_sha256=dependency_lock_hash,
        files=(source_record,),
    )
    execution_context = ExecutionContextV1(
        **{
            **execution_context.__dict__,
            "image_source_sha256": image_manifest.manifest_sha256,
        }
    )
    _write_json(run_root / "execution_context.json", execution_context.to_dict())
    _write_json(run_root / "image_source_manifest.json", image_manifest.to_dict())
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
        candidate_path=stored_candidate,
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
        candidate_graph_hash=candidate_validation.graph_hash,
        accelerator_fingerprint=fingerprint,
        execution_context=execution_context,
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
            "accelerator_fingerprint": fingerprint,
        }
    )
    (output / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    summary = TrainingResult(
        success=True,
        failure_stage="",
        error="",
        profile_name=SMOKE_TRAIN_CUDA_V2.name,
        profile_version=SMOKE_TRAIN_CUDA_V2.version,
        profile_hash=SMOKE_TRAIN_CUDA_V2.profile_hash,
        candidate_source_hash=candidate_hash,
        initialization_seed=initialization_seed,
        data_seed=seeds.training_data_seed,
        development_seed=seeds.development_set_seed,
        dataloader_seed=seeds.dataloader_seed,
        device="cuda:0",
        dtype="float32",
        steps_completed=SMOKE_TRAIN_CUDA_V2.max_steps,
        examples_processed=(
            SMOKE_TRAIN_CUDA_V2.max_steps * SMOKE_TRAIN_CUDA_V2.global_batch_size
        ),
        best_development_step=SMOKE_TRAIN_CUDA_V2.max_steps,
        best_development_exact_match_accuracy=0.0,
        best_development_loss=1.0,
        final_training_loss=1.0,
        train_seconds=10.0,
        accelerator_kind="cuda",
        peak_accelerator_allocated_bytes=1536,
        current_accelerator_allocated_bytes=1024,
        reserved_accelerator_allocated_bytes=2048,
        accelerator_total_memory_bytes=16_000_000_000,
        accelerator_fingerprint=fingerprint,
        parameter_count_metadata=sum(
            parameter.numel() for parameter in final_model.parameters()
        ),
        checkpoint_path=checkpoint.name,
        checkpoint_sha256=_sha256(checkpoint),
        event_log_path=events.name,
        unsupported_operation_fallback=False,
        scientific=False,
        hardware_matched=True,
        cleanup_completed=True,
    ).to_dict()
    (output / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_json(output / "runtime_validity.json", {"passed": True})
    return output


def _synthetic_modal_canary_bundle(
    tmp_path: Path,
    *,
    complete_controller_rosters: bool = True,
) -> tuple[Path, str]:
    download_root = tmp_path / "modal-downloads"
    download_root.mkdir()
    prefix = "modal-canary-20260809"
    image_source = {
        "schema_name": "ModalImageSourceManifest",
        "schema_version": "1.0",
        "recipe_version": IMAGE_RECIPE_VERSION,
        "python_version": PYTHON_VERSION,
        "uv_version": UV_VERSION,
        "modal_version": MODAL_VERSION,
        "dependency_lock_sha256": "d" * 64,
        "files": [],
    }
    image_source_sha256 = canonical_sha256(image_source)
    for index, harness in enumerate(CANARY_ORDER, start=1):
        suffix = harness.replace("_autoresearch", "-ar").replace("_", "-")
        run_id = f"{prefix}-{suffix}"
        run_directory = download_root / run_id
        controller = run_directory / "controller"
        controller.mkdir(parents=True)
        context = ExecutionContextV1(
            execution_backend="modal",
            run_id=run_id,
            app_name=APP_NAME,
            function_name=f"canary_{harness}",
            modal_app_id="ap-app123",
            modal_function_id=f"fu-function{index}",
            modal_call_id=f"fc-call{index}",
            modal_image_id="im-image123",
            image_source_sha256=image_source_sha256,
            artifact_uri=volume_artifact_uri(run_id),
        )
        _write_json(run_directory / "execution_context.json", context.to_dict())
        _write_json(run_directory / "image_source_manifest.json", image_source)
        _write_json(
            run_directory / "remote_action_result.json",
            {
                "success": True,
                "mode": "one_opportunity_engineering_canary",
                "harness": harness,
                "returncode": 0,
                "stdout_sha256": f"{index:x}" * 64,
                "stdout_size_bytes": 100 + index,
                "stderr_sha256": "0" * 64,
                "stderr_size_bytes": 0,
            },
        )
        controller_run_id = f"controller-{index}"
        run_manifest = {
            "schema_name": "ControllerRunManifest",
            "schema_version": "2.0",
            "run_id": controller_run_id,
            "condition": harness,
            "seed": 1,
            "candidate_budget": 2,
            "mutation_budget": 1,
            "maximum_provider_attempts": 1,
            "candidate_training_budget": 2,
            "provider_attempt_ledger": PROVIDER_ATTEMPT_LEDGER_FILENAME,
            "provider_attempt_schema": PROVIDER_ATTEMPT_SCHEMA,
            "authoritative_scientific_evidence": False,
            "initial_candidate_hash": "1" * 64,
            "initial_architecture_hash": "2" * 64,
            "architecture_hash_schema": "architecture_graph_v1",
            "architecture_deduplication": {
                "scope": "run",
                "identity": "normalized_executable_architecture_hash",
                "duplicate_proposals_train": False,
                "duplicate_proposals_consume_opportunity": True,
            },
            "evaluator_hash": "3" * 64,
            "trusted_executable_component_hashes": {},
            "trusted_component_set_sha256": "4" * 64,
            "config_hash": "5" * 64,
            "generator": {
                "provider_identity": "openai_official",
                "api_endpoint": OFFICIAL_OPENAI_API_BASE,
                "model": TARGET_MODEL,
                "api_mode": API_MODE,
                "api_base_configured": True,
                "reasoning_effort": "high",
                "max_completion_tokens": 16_384,
                "request_timeout_seconds": 180,
                "retries": 0,
                "retry_delay_seconds": 0,
                "temperature": None,
                "top_p": None,
                "request_seed": 1,
                "generation_seed_support": "best_effort_api_seed",
                "request_settings_source": "environment_overrides_permitted",
            },
            "training": {
                "profile": SMOKE_TRAIN_CUDA_V2.name,
                "profile_version": SMOKE_TRAIN_CUDA_V2.version,
                "profile_hash": SMOKE_TRAIN_CUDA_V2.profile_hash,
                "device": "cuda",
                "allow_cpu_for_tests": False,
            },
            "evaluation": {
                "profile": "smoke_eval_v1",
                "case_count": 24,
                "scientific": False,
            },
        }
        if harness in {"greedy_autoresearch", "semantic_autoresearch"}:
            run_manifest.update(
                {
                    "initial_candidate_is_evaluated": True,
                    "candidate_format": "architecture_tensor_graph@1.0",
                    "max_ir_bytes": 40_000,
                    "run_mode": "engineering_pilot",
                    "exploratory_only": True,
                    "selection_semantics": "mechanics_only_transformer_validity",
                    "prompt_protocol": {},
                    "preflight": {},
                    "evidence_scope": "secondary_native_replication",
                }
            )
            if harness == "greedy_autoresearch":
                run_manifest["greedy_retention"] = {
                    "requires_parent_eligibility": True,
                    "rejects_search_score_regressions": True,
                    "accept_valid_plateau_moves": True,
                }
            else:
                run_manifest["semantic_archive"] = {
                    "axes": [],
                    "parent_policy": "least_used_cell_then_accuracy",
                    "novelty_role": "exploratory_coverage_tiebreak_only",
                    "scientific_novelty_claim": False,
                    "parameter_count_role": "descriptive_metadata_only",
                }
            summary = {
                "schema_name": "ControllerRunSummary",
                "schema_version": "2.0",
                "run_id": controller_run_id,
                "condition": harness,
                "proposal_opportunities_requested": 1,
                "proposal_opportunities_terminal": 1,
            }
            if harness == "greedy_autoresearch":
                summary.update(
                    {
                        "lineage_path": "lineage.jsonl",
                        "incumbent_path": "incumbent.ir.json",
                        "authoritative_scientific_evidence": False,
                    }
                )
            else:
                summary.update(
                    {
                        "semantic_archive_cells": 1,
                        "lineage_path": "lineage.jsonl",
                        "archive_path": "semantic_archive.json",
                        "scientific_novelty_claim": False,
                    }
                )
            _write_json(controller / "run_summary.json", summary)
        else:
            run_manifest.update(
                {
                    "initial_program_is_evaluated": True,
                    "engineering_pilot": True,
                    "proposal_opportunities": 1,
                    "candidate_format": "architecture_ir_json",
                    "proposal_format": "strict_full_document_json",
                    "generated_python_execution": False,
                    "containment_bypass": False,
                    "parent_relative_architecture_change_required": True,
                    "proposal_terminal_ledger": "proposal_terminal_outcomes.jsonl",
                    "evidence_scope": "exploratory_engineering_pilot",
                    "eligibility_threshold": 0.0,
                    "limitations": ["engineering fixture"],
                }
            )
            summary = {
                "schema_name": "ControllerRunResult",
                "schema_version": "2.0",
                "run_id": controller_run_id,
                "condition": harness,
                "completed": True,
                "eligible_best_program_found": True,
                "best_program_id": f"program-{index}",
                "engineering_pilot": True,
                "authoritative_scientific_evidence": False,
                "proposal_opportunities_requested": 1,
                "proposal_opportunities_completed": 1,
                "proposal_terminal_iterations": [1],
                "proposal_terminal_status_counts": {"candidate": 1},
                "proposal_accounting_errors": [],
                "failure_stage": "",
            }
            _write_json(controller / "run_result.json", summary)
        _write_json(controller / "run_manifest.json", run_manifest)
        attempt = ProviderAttemptRecord(
            schema_name=ProviderAttemptRecord.SCHEMA_NAME,
            schema_version=ProviderAttemptRecord.SCHEMA_VERSION,
            harness=harness,
            action="one_opportunity_engineering_canary",
            controller_run_id=controller_run_id,
            execution_backend="modal",
            action_run_id=run_id,
            modal_call_id=context.modal_call_id,
            attempt_ordinal=1,
            started_at_utc=f"2026-08-09T00:00:0{index}.000000Z",
            ended_at_utc=f"2026-08-09T00:00:0{index}.500000Z",
            status="success",
            api_endpoint=OFFICIAL_OPENAI_API_BASE,
            model=TARGET_MODEL,
            generation_settings_sha256=generation_settings_sha256(
                {
                    "model": TARGET_MODEL,
                    "reasoning_effort": "high",
                    "max_completion_tokens": 16_384,
                    "seed": 1,
                }
            ),
            provider_response_id=f"chatcmpl-canary{index}",
            provider_request_id=f"req_canary{index}",
            usage_known=True,
            input_tokens=100 + index,
            output_tokens=20 + index,
            total_tokens=120 + (2 * index),
            error_class=None,
        )
        (controller / PROVIDER_ATTEMPT_LEDGER_FILENAME).write_text(
            json.dumps(attempt.to_dict(), sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        if complete_controller_rosters:
            expected_roster = {
                "greedy_autoresearch": {
                    "accepted_lineage",
                    "architecture_hash_registry",
                    "artifacts",
                    "candidate_training",
                    "incumbent.ir.json",
                    "lineage.jsonl",
                    "prompt_snapshot",
                },
                "semantic_autoresearch": {
                    "architecture_hash_registry",
                    "artifacts",
                    "candidate_training",
                    "lineage.jsonl",
                    "prompt_snapshot",
                    "semantic_archive.json",
                },
                "openevolve_generic": {
                    "architecture_hash_registry",
                    "best",
                    "candidate_training",
                    "checkpoints",
                    "database",
                    "evolution_trace.jsonl",
                    "logs",
                    "proposal_terminal_outcomes.jsonl",
                },
                "openevolve_semantic": {
                    "architecture_hash_registry",
                    "best",
                    "candidate_training",
                    "checkpoints",
                    "database",
                    "evolution_trace.jsonl",
                    "logs",
                    "proposal_terminal_outcomes.jsonl",
                },
            }[harness]
            for name in expected_roster:
                path = controller / name
                if path.suffix in {".json", ".jsonl"}:
                    path.write_text("{}\n", encoding="utf-8")
                else:
                    path.mkdir()
            if harness in {"greedy_autoresearch", "semantic_autoresearch"}:
                lineage = [
                    _native_lineage_record(
                        harness=harness,
                        controller_run_id=controller_run_id,
                        candidate_hash=character * 64,
                        opportunity=opportunity,
                        input_tokens=attempt.input_tokens if opportunity else 0,
                        output_tokens=attempt.output_tokens if opportunity else 0,
                    )
                    for opportunity, character in enumerate(("8", "9"))
                ]
                (controller / "lineage.jsonl").write_text(
                    "".join(
                        json.dumps(item, sort_keys=True) + "\n" for item in lineage
                    ),
                    encoding="utf-8",
                )
            else:
                trace = {
                    "iteration": 1,
                    "timestamp": 1_786_224_001.0,
                    "parent_id": "parent-program",
                    "child_id": "child-program",
                    "parent_metrics": {},
                    "child_metrics": {},
                    "parent_code": "{}",
                    "child_code": "{}",
                    "parent_changes_description": "",
                    "prompt": {"system": "system", "user": "user"},
                    "llm_response": "{}",
                    "improvement_delta": {},
                    "island_id": 0,
                    "generation": 1,
                    "artifacts": {
                        "candidate_architecture_hash": "a" * 64,
                        "candidate_graph_hash": "b" * 64,
                        "failure_stage": "",
                        "infrastructure_failure": False,
                        "layer_a_record_id": "evaluation-1",
                        "parent_architecture_hash": "c" * 64,
                    },
                    "metadata": {"iteration_time": 1.0, "changes": ""},
                }
                (controller / "evolution_trace.jsonl").write_text(
                    json.dumps(trace, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                checkpoint = controller / "checkpoints" / "checkpoint_1"
                checkpoint.mkdir()
                _write_json(
                    checkpoint / "metadata.json",
                    {
                        "archive": ["child-program"],
                        "best_program_id": "child-program",
                        "current_island": 0,
                        "feature_stats": {},
                        "island_best_programs": ["child-program"],
                        "island_feature_maps": [{}],
                        "island_generations": [1],
                        "islands": [["child-program"]],
                        "last_iteration": 1,
                        "last_migration_generation": 0,
                    },
                )
        artifact_manifest = build_artifact_manifest(
            run_directory,
            run_id=run_id,
            image_source_sha256=image_source_sha256,
        )
        write_artifact_manifest(run_directory, artifact_manifest)
    return download_root, prefix


def _canonical_modal_canary_bundle(project_root: Path) -> tuple[Path, str]:
    development = project_root / "outputs" / "development"
    development.mkdir(parents=True)
    temporary_root, prefix = _synthetic_modal_canary_bundle(development)
    download_root = development / "modal_downloads"
    temporary_root.rename(download_root)
    return download_root, prefix


def _selected_canary_run_ids(download_root: Path, prefix: str) -> dict[str, str]:
    return {
        harness: (
            download_root
            / f"{prefix}-{harness.replace('_autoresearch', '-ar').replace('_', '-')}"
        ).name
        for harness in CANARY_ORDER
    }


def _refresh_artifact_manifest(run_directory: Path) -> None:
    manifest_path = run_directory / "artifact_manifest.json"
    manifest_path.unlink()
    image_source = json.loads(
        (run_directory / "image_source_manifest.json").read_text(encoding="utf-8")
    )
    manifest = build_artifact_manifest(
        run_directory,
        run_id=run_directory.name,
        image_source_sha256=canonical_sha256(image_source),
    )
    write_artifact_manifest(run_directory, manifest)


def _rename_canary_attempt(run_directory: Path, new_run_id: str) -> Path:
    renamed = run_directory.with_name(new_run_id)
    run_directory.rename(renamed)
    context_path = renamed / "execution_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["run_id"] = new_run_id
    context["artifact_uri"] = volume_artifact_uri(new_run_id)
    _write_json(context_path, context)
    ledger_path = renamed / "controller" / PROVIDER_ATTEMPT_LEDGER_FILENAME
    attempt = json.loads(ledger_path.read_text(encoding="utf-8"))
    attempt["action_run_id"] = new_run_id
    ledger_path.write_text(
        json.dumps(attempt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_artifact_manifest(renamed)
    return renamed


def _rewrite_private_candidate(
    source: Path,
    destination: Path,
    *,
    context: ExecutionContextV1,
    mutate_architecture: bool,
) -> dict[str, str]:
    shutil.copytree(source, destination)
    graph_path = destination / "candidate_graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if mutate_architecture:
        norm = next(
            node for node in graph["nodes"] if node["node_id"] == "block1_norm1"
        )
        norm["attributes"]["epsilon"] = 0.0001
        graph_path.write_text(
            json.dumps(graph, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    candidate_hash = _sha256(graph_path)
    validation = validate_ir_candidate_json(graph_path.read_text(encoding="utf-8"))
    assert validation.valid and validation.graph is not None
    for name in (
        "best_checkpoint.pt",
        "partial_resume_checkpoint.pt",
        "latest_resume_checkpoint.pt",
    ):
        path = destination / name
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["candidate_source_hash"] = candidate_hash
        torch.save(payload, path)
    manifest_path = destination / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "candidate_source_hash": candidate_hash,
            "candidate_artifact_hash": candidate_hash,
            "candidate_graph_hash": validation.graph_hash,
            "execution_context": context.to_dict(),
        }
    )
    _write_json(manifest_path, manifest)
    summary_path = destination / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["candidate_source_hash"] = candidate_hash
    summary["checkpoint_sha256"] = _sha256(destination / "best_checkpoint.pt")
    _write_json(summary_path, summary)
    return {
        "candidate_hash": candidate_hash,
        "graph_hash": validation.graph_hash,
        "architecture_hash": validation.graph.architecture_hash,
    }


def _native_lineage_record(
    *,
    harness: str,
    controller_run_id: str,
    candidate_hash: str,
    opportunity: int,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    lineage_id = f"lineage-{opportunity}"
    parent_hash = None if opportunity == 0 else "parent-candidate"
    record = {
        "run_id": controller_run_id,
        "condition": harness,
        "seed": 1,
        "candidate_id": candidate_hash,
        "parent_id": parent_hash,
        "lineage_record_id": lineage_id,
        "proposal_id": "" if opportunity == 0 else "proposal-1",
        "parent_lineage_record_id": None if opportunity == 0 else "lineage-0",
        "inspiration_ids": [],
        "proposal_text": "seed" if opportunity == 0 else "proposal",
        "mechanism_hypothesis": "fixture",
        "prompt_hash": "" if opportunity == 0 else "6" * 64,
        "response_hash": "" if opportunity == 0 else "7" * 64,
        "code_hash": candidate_hash,
        "diff": "",
        "proposal_timestamp": "2026-08-09T00:00:00+00:00",
        "completion_timestamp": "2026-08-09T00:00:01+00:00",
        "retention_decision": "seed_parent" if opportunity == 0 else "accept",
        "archive_cells": [],
        "rollback_target": None,
        "future_parent_count": 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "schema_name": "search_evaluation",
        "schema_version": "1.0",
        "record_id": f"evaluation-{opportunity}",
        "evaluation_run_id": controller_run_id,
        "condition_id": f"native-{harness.replace('_', '-')}",
        "evaluation_candidate_id": f"candidate-{candidate_hash}",
        "execution_ok": True,
        "transformer_valid": True,
        "public_accuracy": 0.5,
        "search_score": 0.5,
        "eligible_for_parent": True,
        "failure_stage": "",
        "infrastructure_failure": False,
        "online_descriptor_codes": [],
    }
    if harness == "greedy_autoresearch":
        record.update(
            {
                "proposal_opportunity": opportunity,
                "candidate_role": "initial_seed" if opportunity == 0 else "proposed_ir",
            }
        )
    else:
        record["opportunity_index"] = opportunity
    return record


def _synthetic_private_greedy_canary(
    tmp_path: Path,
) -> tuple[Path, ExecutionContextV1]:
    project = _project_fixture(tmp_path / "project-fixture", active_cuda=True)
    bundle_fixture = tmp_path / "bundle-fixture"
    bundle_fixture.mkdir()
    download_root, prefix = _synthetic_modal_canary_bundle(
        bundle_fixture,
        complete_controller_rosters=False,
    )
    run_root = download_root / f"{prefix}-greedy-ar"
    context = ExecutionContextV1.from_dict(
        json.loads((run_root / "execution_context.json").read_text(encoding="utf-8"))
    )
    controller = run_root / "controller"

    smoke = _synthetic_cuda_smoke(project, tmp_path / "candidate-fixture")
    training_root = controller / "candidate_training"
    seed_report = _rewrite_private_candidate(
        smoke,
        training_root / "seed-placeholder",
        context=context,
        mutate_architecture=False,
    )
    proposal_report = _rewrite_private_candidate(
        smoke,
        training_root / "proposal-placeholder",
        context=context,
        mutate_architecture=True,
    )
    seed_dir = training_root / f"0000_{seed_report['candidate_hash'][:12]}"
    proposal_dir = training_root / f"0001_{proposal_report['candidate_hash'][:12]}"
    (training_root / "seed-placeholder").rename(seed_dir)
    (training_root / "proposal-placeholder").rename(proposal_dir)

    registry = controller / "architecture_hash_registry"
    registry.mkdir()
    for digest in (
        seed_report["architecture_hash"],
        proposal_report["architecture_hash"],
    ):
        (registry / digest).write_text(digest + "\n", encoding="ascii")

    snapshot = controller / "prompt_snapshot"
    snapshot.mkdir()
    sources = (
        ("shared_system", ROOT / "common" / "prompts" / "shared_system.md"),
        ("shared_task", ROOT / "common" / "prompts" / "shared_task.md"),
        (
            "architecture_ir_contract",
            ROOT / "common" / "prompts" / "architecture_ir_contract.md",
        ),
        (
            "greedy_autoresearch_program",
            ROOT / "agents" / "greedy_autoresearch" / "program.md",
        ),
    )
    component_records = []
    combined_parts = []
    for name, source in sources:
        content = source.read_text(encoding="utf-8")
        combined_parts.append(content)
        (snapshot / f"{name}.md").write_text(content, encoding="utf-8")
        component_records.append(
            {
                "name": name,
                "source_path": source.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    combined = "\n\n".join(combined_parts)
    (snapshot / "combined_system_prompt.md").write_text(combined, encoding="utf-8")

    manifest_path = controller / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "initial_candidate_hash": seed_report["candidate_hash"],
            "initial_architecture_hash": seed_report["architecture_hash"],
            "prompt_protocol": {
                "components": component_records,
                "combined_system_prompt_sha256": hashlib.sha256(
                    combined.encode("utf-8")
                ).hexdigest(),
                "message_hash": "sha256_canonical_json_v1",
                "snapshot_directory": "prompt_snapshot",
            },
        }
    )
    _write_json(manifest_path, manifest)

    user_prompt = "Return exactly one changed Architecture IR JSON document."
    messages = [
        {"role": "system", "content": combined},
        {"role": "user", "content": user_prompt},
    ]
    artifacts = controller / "artifacts"
    artifacts.mkdir()
    (artifacts / "0001.messages.json").write_text(
        json.dumps(messages, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifacts / "0001.prompt.md").write_text(user_prompt, encoding="utf-8")
    proposal_text = (proposal_dir / "candidate_graph.json").read_text(encoding="utf-8")
    (artifacts / "0001.response.txt").write_text(proposal_text, encoding="utf-8")
    proposal_artifact = artifacts / (
        f"0001_{proposal_report['candidate_hash'][:12]}.ir.json"
    )
    proposal_artifact.write_text(proposal_text, encoding="utf-8")
    (controller / "incumbent.ir.json").write_text(proposal_text, encoding="utf-8")

    attempt = json.loads(
        (controller / PROVIDER_ATTEMPT_LEDGER_FILENAME).read_text(encoding="utf-8")
    )
    controller_run_id = manifest["run_id"]
    lineage = []
    for opportunity, report in enumerate((seed_report, proposal_report)):
        lineage.append(
            _native_lineage_record(
                harness="greedy_autoresearch",
                controller_run_id=controller_run_id,
                candidate_hash=report["candidate_hash"],
                opportunity=opportunity,
                input_tokens=attempt["input_tokens"] if opportunity else 0,
                output_tokens=attempt["output_tokens"] if opportunity else 0,
            )
        )
    (controller / "lineage.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in lineage),
        encoding="utf-8",
    )
    accepted = controller / "accepted_lineage"
    (accepted / ".git").mkdir(parents=True)
    (accepted / "candidate.ir.json").write_text(proposal_text, encoding="utf-8")
    (accepted / ".git" / "HEAD").write_text(
        "ref: refs/heads/main\n", encoding="utf-8"
    )
    return controller, context


def _synthetic_private_openevolve_canary(
    tmp_path: Path,
    *,
    harness: str = "openevolve_generic",
) -> tuple[Path, ExecutionContextV1]:
    native_controller, _native_context = _synthetic_private_greedy_canary(tmp_path)
    download_root = native_controller.parent.parent
    target_run = download_root / f"modal-canary-20260809-{harness.replace('_', '-')}"
    controller = target_run / "controller"
    context = ExecutionContextV1.from_dict(
        json.loads((target_run / "execution_context.json").read_text(encoding="utf-8"))
    )

    training_root = controller / "candidate_training"
    shutil.copytree(native_controller / "candidate_training", training_root)
    candidate_records = []
    for index, candidate in enumerate(sorted(training_root.iterdir())):
        manifest_path = candidate / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["execution_context"] = context.to_dict()
        _write_json(manifest_path, manifest)
        validation = validate_ir_candidate_json(
            (candidate / "candidate_graph.json").read_text(encoding="utf-8")
        )
        assert validation.valid and validation.graph is not None
        candidate_hash = _sha256(candidate / "candidate_graph.json")
        renamed = training_root / f"{candidate_hash[:12]}_{index + 1:08x}"
        candidate.rename(renamed)
        candidate_records.append(
            {
                "hash": candidate_hash,
                "graph_hash": validation.graph_hash,
                "architecture_hash": validation.graph.architecture_hash,
                "code": (renamed / "candidate_graph.json").read_text(
                    encoding="utf-8"
                ),
            }
        )
    seed, child = candidate_records

    registry = controller / "architecture_hash_registry"
    registry.mkdir()
    for record in candidate_records:
        digest = record["architecture_hash"]
        (registry / digest).write_text(digest + "\n", encoding="ascii")

    manifest_path = controller / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "initial_candidate_hash": seed["hash"],
            "initial_architecture_hash": seed["architecture_hash"],
        }
    )
    _write_json(manifest_path, manifest)

    program_ids = ("11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222")

    def program_payload(index: int, *, checkpoint: bool) -> dict:
        record = candidate_records[index]
        parent_architecture = None if index == 0 else seed["architecture_hash"]
        return {
            "id": program_ids[index],
            "code": record["code"],
            "changes_description": "initial" if index == 0 else "changed epsilon",
            "language": "json",
            "parent_id": None if index == 0 else program_ids[0],
            "generation": index,
            "timestamp": 1_786_224_000.0 + index,
            "iteration_found": index,
            "metrics": {
                "execution_ok": 1.0,
                "transformer_valid": 1.0,
                "eligible_for_parent": 1.0,
                "public_accuracy": 0.5,
                "search_score": 0.5,
                "combined_score": 0.5,
            },
            "complexity": float(len(record["code"])),
            "diversity": float(index),
            "metadata": {"island": 0},
            "prompts": (
                {"full_rewrite_user": {"system": "system", "user": "user"}}
                if checkpoint and index == 1
                else None
            ),
            "artifacts_json": (
                json.dumps(
                    {
                        "candidate_graph_hash": record["graph_hash"],
                        "candidate_architecture_hash": record["architecture_hash"],
                        "parent_architecture_hash": parent_architecture,
                        "failure_stage": "",
                        "infrastructure_failure": False,
                    },
                    sort_keys=True,
                )
                if checkpoint
                else None
            ),
            "artifact_dir": None,
            "embedding": None,
        }

    database_programs = controller / "database" / "programs"
    database_programs.mkdir(parents=True)
    for index, program_id in enumerate(program_ids):
        _write_json(
            database_programs / f"{program_id}.json",
            program_payload(index, checkpoint=False),
        )

    checkpoint = controller / "checkpoints" / "checkpoint_1"
    checkpoint_programs = checkpoint / "programs"
    checkpoint_programs.mkdir(parents=True)
    for index, program_id in enumerate(program_ids):
        _write_json(
            checkpoint_programs / f"{program_id}.json",
            program_payload(index, checkpoint=True),
        )
    _write_json(
        checkpoint / "metadata.json",
        {
            "archive": list(program_ids),
            "best_program_id": program_ids[1],
            "current_island": 0,
            "feature_stats": {},
            "island_best_programs": [program_ids[1], None, None, None],
            "island_feature_maps": [{}, {}, {}, {}],
            "island_generations": [1, 0, 0, 0],
            "islands": [list(program_ids), [], [], []],
            "last_iteration": 1,
            "last_migration_generation": 0,
        },
    )
    child_code = child["code"]
    (checkpoint / "best_program.json").write_text(child_code, encoding="utf-8")
    best_info = {
        "id": program_ids[1],
        "generation": 1,
        "iteration": 1,
        "metrics": program_payload(1, checkpoint=True)["metrics"],
        "language": "json",
        "timestamp": 1_786_224_001.0,
    }
    _write_json(checkpoint / "best_program_info.json", best_info)

    best = controller / "best"
    best.mkdir()
    (best / "best_program.json").write_text(child_code, encoding="utf-8")
    _write_json(
        best / "best_program_info.json",
        {
            **best_info,
            "parent_id": program_ids[0],
            "saved_at": 1_786_224_002.0,
        },
    )
    (controller / "proposal_terminal_outcomes.jsonl").write_text(
        json.dumps(
            {"candidate_produced": True, "iteration": 1, "status": "candidate"},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (controller / "evolution_trace.jsonl").write_text(
        json.dumps(
            {
                "iteration": 1,
                "timestamp": 1_786_224_001.0,
                "parent_id": program_ids[0],
                "child_id": program_ids[1],
                "parent_metrics": program_payload(0, checkpoint=True)["metrics"],
                "child_metrics": program_payload(1, checkpoint=True)["metrics"],
                "parent_code": seed["code"],
                "child_code": child_code,
                "parent_changes_description": "initial",
                "prompt": {"system": "system", "user": "user"},
                "llm_response": child_code,
                "improvement_delta": {
                    "execution_ok": 0.0,
                    "transformer_valid": 0.0,
                    "eligible_for_parent": 0.0,
                    "public_accuracy": 0.0,
                    "search_score": 0.0,
                    "combined_score": 0.0,
                },
                "island_id": 0,
                "generation": 1,
                "artifacts": {
                    "candidate_architecture_hash": child["architecture_hash"],
                    "candidate_graph_hash": child["graph_hash"],
                    "failure_stage": "",
                    "infrastructure_failure": False,
                    "layer_a_record_id": "evaluation-1",
                    "parent_architecture_hash": seed["architecture_hash"],
                },
                "metadata": {"iteration_time": 1.0, "changes": ""},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    logs = controller / "logs"
    logs.mkdir()
    (logs / "openevolve_20260809_000000.log").write_text(
        "one opportunity complete\n", encoding="utf-8"
    )
    return controller, context


def _synthetic_private_semantic_canary(
    tmp_path: Path,
) -> tuple[Path, ExecutionContextV1]:
    greedy, _greedy_context = _synthetic_private_greedy_canary(tmp_path)
    download_root = greedy.parent.parent
    target_run = download_root / "modal-canary-20260809-semantic-ar"
    controller = target_run / "controller"
    context = ExecutionContextV1.from_dict(
        json.loads((target_run / "execution_context.json").read_text(encoding="utf-8"))
    )
    shutil.copytree(greedy / "candidate_training", controller / "candidate_training")
    for candidate in (controller / "candidate_training").iterdir():
        manifest_path = candidate / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["execution_context"] = context.to_dict()
        _write_json(manifest_path, manifest)
    shutil.copytree(
        greedy / "architecture_hash_registry",
        controller / "architecture_hash_registry",
    )
    candidates = sorted((controller / "candidate_training").iterdir())
    seed_hash = _sha256(candidates[0] / "candidate_graph.json")
    child_hash = _sha256(candidates[1] / "candidate_graph.json")
    seed_validation = validate_ir_candidate_json(
        (candidates[0] / "candidate_graph.json").read_text(encoding="utf-8")
    )
    assert seed_validation.valid and seed_validation.graph is not None

    sources = (
        ("shared_system", ROOT / "common" / "prompts" / "shared_system.md"),
        ("shared_task", ROOT / "common" / "prompts" / "shared_task.md"),
        (
            "architecture_ir_contract",
            ROOT / "common" / "prompts" / "architecture_ir_contract.md",
        ),
        (
            "semantic_autoresearch_program",
            ROOT / "agents" / "semantic_autoresearch" / "program.md",
        ),
    )
    snapshot = controller / "prompt_snapshot"
    snapshot.mkdir()
    combined_parts = []
    component_records = []
    for name, source in sources:
        content = source.read_text(encoding="utf-8")
        combined_parts.append(content)
        (snapshot / f"{name}.md").write_text(content, encoding="utf-8")
        component_records.append(
            {
                "name": name,
                "source_path": source.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    combined = "\n\n".join(combined_parts)
    (snapshot / "combined_system_prompt.md").write_text(combined, encoding="utf-8")
    manifest_path = controller / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "initial_candidate_hash": seed_hash,
            "initial_architecture_hash": seed_validation.graph.architecture_hash,
            "prompt_protocol": {
                "components": component_records,
                "combined_system_prompt_sha256": hashlib.sha256(
                    combined.encode("utf-8")
                ).hexdigest(),
                "message_hash": "sha256_canonical_json_v1",
                "snapshot_directory": "prompt_snapshot",
            },
        }
    )
    _write_json(manifest_path, manifest)

    seed_code = (candidates[0] / "candidate_graph.json").read_text(encoding="utf-8")
    child_code = (candidates[1] / "candidate_graph.json").read_text(encoding="utf-8")
    artifacts = controller / "artifacts"
    artifacts.mkdir()
    (artifacts / "0000_seed.ir.json").write_text(seed_code, encoding="utf-8")
    user_prompt = "Return one semantically indexed Architecture IR."
    (artifacts / "0001.messages.json").write_text(
        json.dumps(
            [
                {"role": "system", "content": combined},
                {"role": "user", "content": user_prompt},
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifacts / "0001.prompt.md").write_text(user_prompt, encoding="utf-8")
    (artifacts / "0001.response.txt").write_text(child_code, encoding="utf-8")
    (artifacts / f"0001_{child_hash[:12]}.ir.json").write_text(
        child_code, encoding="utf-8"
    )
    attempt = json.loads(
        (controller / PROVIDER_ATTEMPT_LEDGER_FILENAME).read_text(encoding="utf-8")
    )
    lineage = []
    for opportunity, candidate_hash in enumerate((seed_hash, child_hash)):
        lineage.append(
            _native_lineage_record(
                harness="semantic_autoresearch",
                controller_run_id=manifest["run_id"],
                candidate_hash=candidate_hash,
                opportunity=opportunity,
                input_tokens=attempt["input_tokens"] if opportunity else 0,
                output_tokens=attempt["output_tokens"] if opportunity else 0,
            )
        )
    (controller / "lineage.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in lineage),
        encoding="utf-8",
    )
    axes = [SEMANTIC_METRIC_NAMES[axis] for axis in CATEGORY_CODES]
    _write_json(
        controller / "semantic_archive.json",
        {
            "schema_name": "semantic_autoresearch_archive",
            "schema_version": "2.0",
            "axes": axes,
            "coverage_cells": 1,
            "novelty_role": "exploratory_coverage_tiebreak_only",
            "scientific_novelty_claim": False,
            "cells": [
                {
                    "cell": "seed-cell",
                    "signature": [0] * len(axes),
                    "candidate_id": seed_hash,
                    "lineage_record_id": "lineage-seed",
                    "source_path": "artifacts/0000_seed.ir.json",
                    "search_score": 0.5,
                    "public_accuracy": 0.5,
                    "discovered_opportunity": 0,
                    "parent_uses": 1,
                }
            ],
        },
    )
    summary_path = controller / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["semantic_archive_cells"] = 1
    _write_json(summary_path, summary)
    return controller, context


def test_private_provider_canary_staging_validates_before_publication(tmp_path):
    controller, context = _synthetic_private_greedy_canary(tmp_path)

    report = validate_private_canary_staging(
        controller,
        harness="greedy_autoresearch",
        execution_context=context,
    )

    assert report["valid"] is True
    assert report["harness"] == "greedy_autoresearch"
    assert report["candidate_count"] == 2
    assert report["provider_attempt_count"] == 1
    assert report["file_count"] > 20
    assert report["total_bytes"] > 0


def test_private_semantic_staging_reconciles_archive_and_training(tmp_path):
    controller, context = _synthetic_private_semantic_canary(tmp_path)

    report = validate_private_canary_staging(
        controller,
        harness="semantic_autoresearch",
        execution_context=context,
    )

    assert report["valid"] is True
    assert report["candidate_count"] == 2
    assert report["provider_attempt_count"] == 1


@pytest.mark.parametrize(
    "harness",
    ("openevolve_generic", "openevolve_semantic"),
)
def test_private_openevolve_staging_reconciles_programs_and_training(
    tmp_path,
    harness,
):
    controller, context = _synthetic_private_openevolve_canary(
        tmp_path,
        harness=harness,
    )

    report = validate_private_canary_staging(
        controller,
        harness=harness,
        execution_context=context,
    )

    assert report["valid"] is True
    assert report["candidate_count"] == 2
    assert report["provider_attempt_count"] == 1


def test_private_provider_canary_staging_rejects_adversarial_trees(tmp_path):
    controller, context = _synthetic_private_greedy_canary(tmp_path)

    symlink_tree = tmp_path / "symlink-tree"
    shutil.copytree(controller, symlink_tree)
    (symlink_tree / "artifacts" / "link.txt").symlink_to(
        symlink_tree / "artifacts" / "0001.prompt.md"
    )
    with pytest.raises(ValueError, match="symlinks"):
        validate_private_canary_staging(
            symlink_tree,
            harness="greedy_autoresearch",
            execution_context=context,
        )

    extra_tree = tmp_path / "extra-tree"
    shutil.copytree(controller, extra_tree)
    (extra_tree / "unapproved.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level roster"):
        validate_private_canary_staging(
            extra_tree,
            harness="greedy_autoresearch",
            execution_context=context,
        )

    incomplete_tree = tmp_path / "incomplete-tree"
    shutil.copytree(controller, incomplete_tree)
    candidate = next((incomplete_tree / "candidate_training").iterdir())
    (candidate / "latest_resume_checkpoint.pt").unlink()
    with pytest.raises(ValueError, match="artifact roster"):
        validate_private_canary_staging(
            incomplete_tree,
            harness="greedy_autoresearch",
            execution_context=context,
        )

    context_tree = tmp_path / "context-tree"
    shutil.copytree(controller, context_tree)
    candidate = next((context_tree / "candidate_training").iterdir())
    manifest_path = candidate / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_context"]["run_id"] = "substituted-run"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="context differs"):
        validate_private_canary_staging(
            context_tree,
            harness="greedy_autoresearch",
            execution_context=context,
        )

    credential_tree = tmp_path / "credential-tree"
    shutil.copytree(controller, credential_tree)
    summary_path = credential_tree / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["api_key"] = "not-a-real-key"
    _write_json(summary_path, summary)
    with pytest.raises(ValueError, match="credential-shaped fields"):
        validate_private_canary_staging(
            credential_tree,
            harness="greedy_autoresearch",
            execution_context=context,
        )


@pytest.mark.parametrize("harness", CANARY_ORDER)
def test_private_canary_staging_rejects_every_frozen_schema_extension(
    tmp_path,
    harness,
):
    fixture_root = tmp_path / harness
    if harness == "greedy_autoresearch":
        controller, context = _synthetic_private_greedy_canary(fixture_root)
    elif harness == "semantic_autoresearch":
        controller, context = _synthetic_private_semantic_canary(fixture_root)
    else:
        controller, context = _synthetic_private_openevolve_canary(
            fixture_root,
            harness=harness,
        )

    manifest_path = controller / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected_extension"] = True
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="ControllerRunManifest fields differ"):
        validate_private_canary_staging(
            controller,
            harness=harness,
            execution_context=context,
        )
    del manifest["unexpected_extension"]
    _write_json(manifest_path, manifest)

    summary_name = (
        "run_summary.json"
        if harness in {"greedy_autoresearch", "semantic_autoresearch"}
        else "run_result.json"
    )
    summary_path = controller / summary_name
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["unexpected_extension"] = True
    _write_json(summary_path, summary)
    with pytest.raises(ValueError, match="controller summary fields differ"):
        validate_private_canary_staging(
            controller,
            harness=harness,
            execution_context=context,
        )
    del summary["unexpected_extension"]
    _write_json(summary_path, summary)

    if harness in {"greedy_autoresearch", "semantic_autoresearch"}:
        lineage_path = controller / "lineage.jsonl"
        lineage = [
            json.loads(line)
            for line in lineage_path.read_text(encoding="utf-8").splitlines()
        ]
        lineage[0]["unexpected_extension"] = True
        lineage_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in lineage),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="native lineage record fields differ"):
            validate_private_canary_staging(
                controller,
                harness=harness,
                execution_context=context,
            )
    else:
        trace_path = controller / "evolution_trace.jsonl"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trace["unexpected_extension"] = True
        trace_path.write_text(json.dumps(trace, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="evolution trace fields differ"):
            validate_private_canary_staging(
                controller,
                harness=harness,
                execution_context=context,
            )
        del trace["unexpected_extension"]
        trace["metadata"]["unexpected_extension"] = True
        trace_path.write_text(json.dumps(trace, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="evolution trace metadata fields differ"):
            validate_private_canary_staging(
                controller,
                harness=harness,
                execution_context=context,
            )
        del trace["metadata"]["unexpected_extension"]
        trace_path.write_text(json.dumps(trace, sort_keys=True) + "\n", encoding="utf-8")

        metadata_path = controller / "checkpoints" / "checkpoint_1" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["unexpected_extension"] = True
        _write_json(metadata_path, metadata)
        with pytest.raises(ValueError, match="checkpoint metadata fields differ"):
            validate_private_canary_staging(
                controller,
                harness=harness,
                execution_context=context,
            )


def test_static_controller_surfaces_are_four_harness_and_nonexecuting(
    tmp_path,
):
    project = _project_fixture(tmp_path)
    report = validate_controller_surfaces(project)

    assert report["passed"]
    assert report["real_provider_calls"] == 0
    assert report["local_fixture_calls"] == 4
    assert report["entrypoint_execution_runs"] == 0
    assert report["candidate_execution_runs"] == 0
    assert report["training_runs"] == 0
    assert [item["harness_id"] for item in report["harnesses"]] == [
        "normal_autoresearch",
        "semantic_autoresearch",
        "openevolve",
        "semantic_openevolve",
    ]
    assert report["candidate_format"] == "architecture_ir"
    assert report["trusted_candidate"] == "common/initial_candidate.ir.json"
    assert not Path(report["trusted_candidate"]).is_absolute()
    assert all(
        item["candidate_executable_structure_unchanged"] for item in report["harnesses"]
    )
    assert all(item["candidate_graph_hash_changed"] for item in report["harnesses"])
    assert all(item["candidate_ir_valid"] for item in report["harnesses"])
    assert all(
        item["fixed_response_format"] == "complete_architecture_ir_json"
        for item in report["harnesses"]
    )
    assert all(
        0 < item["fixed_response_bytes"] <= MAX_FAKE_RESPONSE_BYTES
        for item in report["harnesses"]
    )
    assert all(item["static_cli_contract_passed"] for item in report["harnesses"])
    assert not any(item["entrypoint_executed"] for item in report["harnesses"])
    assert not any(item["candidate_executed"] for item in report["harnesses"])


def test_static_controller_surfaces_accept_active_cuda_v2_bindings(tmp_path):
    project = _project_fixture(tmp_path, active_cuda=True)
    report = validate_controller_surfaces(project)

    assert report["passed"], report["errors"]
    assert {item["training_contract"] for item in report["harnesses"]} == {
        "active_v2_cuda"
    }


def test_static_controller_surfaces_reject_a_mixed_profile_device_binding(tmp_path):
    project = _project_fixture(tmp_path, active_cuda=True)
    config_path = project / "agents" / "greedy_autoresearch" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["device"] = "mps"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = validate_controller_surfaces(project)

    assert not report["passed"]
    assert any("neither active CUDA v2" in error for error in report["errors"])


def test_fake_provider_rejects_an_unbounded_response_before_ir_decoding():
    with pytest.raises(ValueError, match="byte limit"):
        DeterministicFakeProvider("x" * (MAX_FAKE_RESPONSE_BYTES + 1))


def test_static_controller_surfaces_fail_when_one_named_harness_is_missing(tmp_path):
    project = _project_fixture(tmp_path, omit="semantic_autoresearch")
    report = validate_controller_surfaces(project)

    assert not report["passed"]
    semantic = next(
        item
        for item in report["harnesses"]
        if item["harness_id"] == "semantic_autoresearch"
    )
    assert not semantic["passed"]
    assert any("missing entrypoint" in error for error in semantic["errors"])


def test_static_controller_surface_requires_explicit_engineering_pilot_flag(tmp_path):
    project = _project_fixture(tmp_path)
    entrypoint = project / "agents" / "greedy_autoresearch" / "run.py"
    entrypoint.write_text(
        entrypoint.read_text(encoding="utf-8").replace(
            "parser.add_argument('--engineering-pilot', action='store_true')\n",
            "",
        ),
        encoding="utf-8",
    )

    report = validate_controller_surfaces(project)

    assert not report["passed"]
    greedy = next(
        item
        for item in report["harnesses"]
        if item["harness_id"] == "normal_autoresearch"
    )
    assert not greedy["static_cli_contract_passed"]
    assert any("--engineering-pilot" in error for error in greedy["errors"])


def test_static_surface_inspection_never_executes_entrypoint_code(tmp_path):
    project = _project_fixture(tmp_path)
    marker = project / "entrypoint-executed.txt"
    entrypoint = project / "agents" / "greedy_autoresearch" / "run.py"
    entrypoint.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('unsafe')\n"
        + entrypoint.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    report = validate_controller_surfaces(project)

    assert report["passed"]
    assert not marker.exists()
    assert report["entrypoint_execution_runs"] == 0


def test_report_never_claims_scientific_or_generated_candidate_readiness(tmp_path):
    project = _project_fixture(tmp_path)
    report = build_report(project_root=project)

    assert report["static_controller_surfaces_passed"]
    assert report["status"] == "static_controller_surfaces_passed"
    assert report["provider_calls"] == 0
    assert report["training_runs"] == 0
    assert report["scientific"] is False
    assert report["scientific_pilot_ready"] is False
    assert report["autonomous_generated_candidate_execution_ready"] is False
    assert report["mps_smoke_artifacts_self_consistent"] is False
    assert report["mps_execution_origin_attested"] is False


def test_existing_smoke_artifacts_are_consistent_without_attesting_execution(tmp_path):
    project = _project_fixture(tmp_path)
    output = _synthetic_mps_smoke(project, tmp_path)

    evidence = validate_existing_mps_smoke(output, project_root=project)

    assert evidence["valid"]
    assert evidence["artifact_self_consistent"]
    assert evidence["execution_origin_attested"] is False
    assert evidence["claim_scope"] == "self_authored_artifact_consistency_only"
    assert evidence["training_started_by_validator"] is False
    assert evidence["parameters_changed"]
    assert evidence["scientific"] is False


def test_historical_mps_smoke_validation_is_read_only_and_hash_stable(tmp_path):
    project = _project_fixture(tmp_path)
    output = _synthetic_mps_smoke(project, tmp_path)
    before = {path.name: _sha256(path) for path in output.iterdir() if path.is_file()}

    evidence = validate_existing_mps_smoke(output, project_root=project)

    after = {path.name: _sha256(path) for path in output.iterdir() if path.is_file()}
    assert evidence["valid"]
    assert evidence["profile"] == "smoke_train_v1"
    assert evidence["accelerator_kind"] == "mps"
    assert after == before


def test_pre_ir_python_mps_smoke_is_readable_and_hash_stable(tmp_path):
    project = _project_fixture(tmp_path)
    output = _synthetic_pre_ir_mps_smoke(project, tmp_path)
    before = {path.name: _sha256(path) for path in output.iterdir() if path.is_file()}

    evidence = validate_existing_mps_smoke(output, project_root=project)

    after = {path.name: _sha256(path) for path in output.iterdir() if path.is_file()}
    assert evidence["valid"], evidence["errors"]
    assert evidence["profile"] == "smoke_train_v1"
    assert evidence["accelerator_kind"] == "mps"
    assert evidence["parameters_changed"]
    assert after == before


def test_cuda_v2_smoke_records_are_validated_without_rewriting_artifacts(tmp_path):
    project = _project_fixture(tmp_path, active_cuda=True)
    output = _synthetic_cuda_smoke(project, tmp_path)
    before = {path.name: _sha256(path) for path in output.iterdir() if path.is_file()}

    evidence = validate_existing_cuda_smoke(output, project_root=project)

    after = {path.name: _sha256(path) for path in output.iterdir() if path.is_file()}
    assert evidence["valid"], evidence["errors"]
    assert evidence["profile"] == "smoke_train_cuda_v2"
    assert evidence["accelerator_kind"] == "cuda"
    assert evidence["execution_origin_attested"] is False
    assert after == before


def test_cuda_v2_smoke_rejects_nondeterministic_runtime_evidence(tmp_path):
    project = _project_fixture(tmp_path, active_cuda=True)
    output = _synthetic_cuda_smoke(project, tmp_path)
    manifest_path = output / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"]["cuda_matmul_allow_tf32"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evidence = validate_existing_cuda_smoke(output, project_root=project)

    assert not evidence["valid"]
    assert any("cuda_matmul_allow_tf32" in error for error in evidence["errors"])


def test_cuda_v2_smoke_rejects_a_v1_checkpoint_kind(tmp_path):
    project = _project_fixture(tmp_path, active_cuda=True)
    output = _synthetic_cuda_smoke(project, tmp_path)
    checkpoint_path = output / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["checkpoint_kind"] = "best_evaluation_weights_v1"
    checkpoint["dependency_lock_hash"] = "e" * 64
    torch.save(checkpoint, checkpoint_path)
    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_sha256"] = _sha256(checkpoint_path)
    _write_json(summary_path, summary)

    evidence = validate_existing_cuda_smoke(output, project_root=project)

    assert not evidence["valid"]
    assert any("checkpoint_kind" in error for error in evidence["errors"])


def test_cuda_v2_smoke_rejects_checkpoint_dependency_lock_mismatch(tmp_path):
    project = _project_fixture(tmp_path, active_cuda=True)
    output = _synthetic_cuda_smoke(project, tmp_path)
    checkpoint_path = output / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["dependency_lock_hash"] = "e" * 64
    torch.save(checkpoint, checkpoint_path)
    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_sha256"] = _sha256(checkpoint_path)
    _write_json(summary_path, summary)

    evidence = validate_existing_cuda_smoke(output, project_root=project)

    assert not evidence["valid"]
    assert any("dependency_lock_hash" in error for error in evidence["errors"])


def test_cuda_v2_smoke_rejects_duplicate_json_fields(tmp_path):
    project = _project_fixture(tmp_path, active_cuda=True)
    output = _synthetic_cuda_smoke(project, tmp_path)
    summary_path = output / "training_summary.json"
    raw = summary_path.read_text(encoding="utf-8")
    raw = raw.replace(
        '"steps_completed": 10',
        '"steps_completed": 10, "steps_completed": 10',
        1,
    )
    summary_path.write_text(raw, encoding="utf-8")

    evidence = validate_existing_cuda_smoke(output, project_root=project)

    assert not evidence["valid"]
    assert any("duplicate JSON field" in error for error in evidence["errors"])


@pytest.mark.parametrize(
    ("artifact", "mutation", "message"),
    (
        ("training_summary.json", "extra", "summary fields differ"),
        ("training_manifest.json", "missing_runtime", "lacks runtime evidence"),
        ("training_events.jsonl", "event_extra", "event line 1 differs"),
        ("best_checkpoint.pt", "checkpoint_extra", "checkpoint fields differ"),
    ),
)
def test_cuda_v2_smoke_rejects_schema_extensions_and_omissions(
    tmp_path,
    artifact,
    mutation,
    message,
):
    project = _project_fixture(tmp_path, active_cuda=True)
    output = _synthetic_cuda_smoke(project, tmp_path)
    path = output / artifact
    if mutation == "extra":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unexpected"] = "extension"
        _write_json(path, payload)
    elif mutation == "missing_runtime":
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["runtime"]
        _write_json(path, payload)
    elif mutation == "event_extra":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        records[0]["unexpected"] = 1
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["unexpected"] = 1
        torch.save(payload, path)
        summary_path = output / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["checkpoint_sha256"] = _sha256(path)
        _write_json(summary_path, summary)

    evidence = validate_existing_cuda_smoke(output, project_root=project)

    assert not evidence["valid"]
    assert any(message in error for error in evidence["errors"])


@pytest.mark.parametrize(
    ("artifact", "field", "message"),
    (
        ("training_summary.json", "steps_completed", "must be an integer"),
        ("training_manifest.json", "runtime.cuda_device_count", "must be an integer"),
        ("training_events.jsonl", "examples_processed", "must be an integer"),
        ("best_checkpoint.pt", "global_step", "must be an integer"),
    ),
)
def test_cuda_v2_smoke_rejects_boolean_as_integer(
    tmp_path,
    artifact,
    field,
    message,
):
    project = _project_fixture(tmp_path, active_cuda=True)
    output = _synthetic_cuda_smoke(project, tmp_path)
    path = output / artifact
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload[field] = True
        torch.save(payload, path)
        summary_path = output / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["checkpoint_sha256"] = _sha256(path)
        _write_json(summary_path, summary)
    elif path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        records[0][field] = True
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "." in field:
            parent, child = field.split(".", 1)
            payload[parent][child] = True
        else:
            payload[field] = True
        _write_json(path, payload)

    evidence = validate_existing_cuda_smoke(output, project_root=project)

    assert not evidence["valid"]
    assert any(message in error for error in evidence["errors"])


def test_cuda_v2_smoke_rejects_absolute_and_credential_bearing_fields(tmp_path):
    project = _project_fixture(tmp_path, active_cuda=True)
    output = _synthetic_cuda_smoke(project, tmp_path)
    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["event_log_path"] = "/mnt/discovery/private/events.jsonl"
    _write_json(summary_path, summary)

    absolute = validate_existing_cuda_smoke(output, project_root=project)
    assert not absolute["valid"]
    assert any("non-portable absolute path" in error for error in absolute["errors"])

    output = _synthetic_cuda_smoke(project, tmp_path / "second")
    manifest_path = output / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["api_key"] = "redacted"
    _write_json(manifest_path, manifest)

    credential = validate_existing_cuda_smoke(output, project_root=project)
    assert not credential["valid"]
    assert any("credential-shaped fields" in error for error in credential["errors"])


def test_cuda_v2_smoke_rejects_mixed_outer_run_and_image_bindings(tmp_path):
    project = _project_fixture(tmp_path, active_cuda=True)
    output = _synthetic_cuda_smoke(project, tmp_path)
    run_root = output.parent.parent
    context_path = run_root / "execution_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["modal_call_id"] = "fc-other123"
    _write_json(context_path, context)

    mixed_context = validate_existing_cuda_smoke(output, project_root=project)
    assert not mixed_context["valid"]
    assert any("outer run context" in error for error in mixed_context["errors"])

    output = _synthetic_cuda_smoke(project, tmp_path / "second")
    run_root = output.parent.parent
    image_path = run_root / "image_source_manifest.json"
    image = json.loads(image_path.read_text(encoding="utf-8"))
    image["dependency_lock_sha256"] = "0" * 64
    _write_json(image_path, image)

    mixed_image = validate_existing_cuda_smoke(output, project_root=project)
    assert not mixed_image["valid"]
    assert any("dependency lock differs" in error for error in mixed_image["errors"])


def test_cuda_v2_smoke_rejects_extra_and_symlinked_action_artifacts(tmp_path):
    project = _project_fixture(tmp_path, active_cuda=True)
    output = _synthetic_cuda_smoke(project, tmp_path)
    (output / "debug.log").write_text("unexpected\n", encoding="utf-8")

    extra = validate_existing_cuda_smoke(output, project_root=project)
    assert not extra["valid"]
    assert any("artifact roster differs" in error for error in extra["errors"])

    output = _synthetic_cuda_smoke(project, tmp_path / "second")
    runtime = output / "runtime_validity.json"
    runtime.unlink()
    runtime.symlink_to(output / "training_summary.json")

    symlinked = validate_existing_cuda_smoke(output, project_root=project)
    assert not symlinked["valid"]
    assert any("regular file" in error for error in symlinked["errors"])


def test_downloaded_modal_canary_bundle_is_hash_verified_and_read_only(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    before = {
        path.relative_to(download_root).as_posix(): _sha256(path)
        for path in download_root.rglob("*")
        if path.is_file()
    }

    evidence = validate_downloaded_modal_canaries(download_root)
    prefix_evidence = validate_downloaded_modal_canaries(download_root / prefix)

    after = {
        path.relative_to(download_root).as_posix(): _sha256(path)
        for path in download_root.rglob("*")
        if path.is_file()
    }
    assert evidence["valid"], evidence["errors"]
    assert prefix_evidence["valid"], prefix_evidence["errors"]
    assert evidence["all_four_canaries_validated"]
    assert [run["harness"] for run in evidence["runs"]] == list(CANARY_ORDER)
    assert all(run["profile"] == "smoke_train_cuda_v2" for run in evidence["runs"])
    assert all(run["proposal_opportunities"] == 1 for run in evidence["runs"])
    assert all(run["provider_attempt_count"] == 1 for run in evidence["runs"])
    assert evidence["provider_attempts_observed"] == 4
    assert evidence["provider_input_tokens"] == 410
    assert evidence["provider_output_tokens"] == 90
    assert evidence["provider_total_tokens"] == 500
    assert evidence["remote_calls_started_by_validator"] == 0
    assert evidence["provider_calls_started_by_validator"] == 0
    assert evidence["training_runs_started_by_validator"] == 0
    assert after == before


def test_create_only_selector_supports_flat_partial_and_recovery_downloads(
    tmp_path,
):
    project = tmp_path / "project"
    project.mkdir()
    download_root, prefix = _canonical_modal_canary_bundle(project)
    run_ids = _selected_canary_run_ids(download_root, prefix)
    generic = download_root / run_ids["openevolve_generic"]
    recovered = _rename_canary_attempt(
        generic,
        "modal-canary-recovery-openevolve-generic",
    )
    run_ids["openevolve_generic"] = recovered.name
    (download_root / "modal-canary-failed-earlier-attempt").mkdir()

    selector_path, selector = create_modal_canary_selector(
        selector_id="provider-canary-cohort-01",
        run_ids=run_ids,
        identity=_selector_identity(
            download_root,
            prefix,
            cohort_id="provider-canary-cohort-01",
        ),
        project_root=project,
    )
    discovery = validate_downloaded_modal_canaries(download_root)
    selected = validate_downloaded_modal_canaries(
        None,
        modal_canary_selector=selector_path,
        project_root=project,
    )

    assert not discovery["valid"]
    assert selected["valid"], selected["errors"]
    assert selected["selection_mode"] == "exact_create_only_selector"
    assert selected["recovery_bundle"] is True
    assert [item["run_id"] for item in selected["runs"]] == [
        run_ids[harness] for harness in CANARY_ORDER
    ]
    assert selector["harness_order"] == list(CANARY_ORDER)
    assert selector_path.relative_to(project).as_posix() == (
        "outputs/readiness/modal_only_final/modal_live_cohorts/"
        + ("9" * 64)
        + "/"
        + selector["image_source_sha256"]
        + "/provider-canary-cohort-01/provider_canary_selection/"
        "provider-canary-cohort-01/canary_run_selector.json"
    )
    with pytest.raises(FileExistsError):
        create_modal_canary_selector(
            selector_id="provider-canary-cohort-01",
            run_ids=run_ids,
            identity=_selector_identity(
                download_root,
                prefix,
                cohort_id="provider-canary-cohort-01",
            ),
            project_root=project,
        )


def test_modal_canary_selector_rejects_hash_drift_and_schema_extensions(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    download_root, prefix = _canonical_modal_canary_bundle(project)
    selector_path, _selector = create_modal_canary_selector(
        selector_id="provider-canary-cohort-02",
        run_ids=_selected_canary_run_ids(download_root, prefix),
        identity=_selector_identity(
            download_root,
            prefix,
            cohort_id="provider-canary-cohort-02",
        ),
        project_root=project,
    )
    payload = json.loads(selector_path.read_text(encoding="utf-8"))
    payload["runs"]["greedy_autoresearch"][
        "execution_context_sha256"
    ] = "0" * 64
    _write_json(selector_path, payload)

    drifted = validate_downloaded_modal_canaries(
        None,
        modal_canary_selector=selector_path,
        project_root=project,
    )

    assert not drifted["valid"]
    assert any("does not match" in error for error in drifted["errors"])

    payload["runs"]["greedy_autoresearch"][
        "execution_context_sha256"
    ] = _sha256(
        download_root
        / _selected_canary_run_ids(download_root, prefix)["greedy_autoresearch"]
        / "execution_context.json"
    )
    payload["unexpected"] = True
    _write_json(selector_path, payload)
    extended = validate_downloaded_modal_canaries(
        None,
        modal_canary_selector=selector_path,
        project_root=project,
    )
    assert not extended["valid"]
    assert any("fields differ" in error for error in extended["errors"])


def test_modal_canary_selector_namespace_isolates_same_named_cohorts(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    download_root, prefix = _canonical_modal_canary_bundle(project)
    run_ids = _selected_canary_run_ids(download_root, prefix)
    first_identity = _selector_identity(
        download_root,
        prefix,
        cohort_id="selector-isolation-a",
    )
    second_identity = ModalLiveCohortIdentity(
        source_tree_sha256=first_identity.source_tree_sha256,
        image_source_sha256=first_identity.image_source_sha256,
        cohort_id="selector-isolation-b",
    )

    first_path, _ = create_modal_canary_selector(
        selector_id="same-selector",
        run_ids=run_ids,
        identity=first_identity,
        project_root=project,
    )
    second_path, _ = create_modal_canary_selector(
        selector_id="same-selector",
        run_ids=run_ids,
        identity=second_identity,
        project_root=project,
    )

    assert first_path != second_path
    load_modal_canary_selector(
        first_path,
        project_root=project,
        expected_identity=first_identity,
    )
    load_modal_canary_selector(
        second_path,
        project_root=project,
        expected_identity=second_identity,
    )
    with pytest.raises(ValueError, match="selected cohort"):
        load_modal_canary_selector(
            first_path,
            project_root=project,
            expected_identity=second_identity,
        )


def test_modal_canary_selector_cli_is_create_only(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()
    download_root, prefix = _canonical_modal_canary_bundle(project)
    run_ids = _selected_canary_run_ids(download_root, prefix)
    arguments = [
        "validate_engineering_canaries.py",
        "--project-root",
        str(project),
        "--create-modal-canary-selector",
        "provider-canary-cohort-03",
    ]
    identity = _selector_identity(
        download_root,
        prefix,
        cohort_id="provider-canary-cohort-03",
    )
    arguments.extend(
        [
            "--source-tree-sha256",
            identity.source_tree_sha256,
            "--image-source-sha256",
            identity.image_source_sha256,
            "--cohort-id",
            identity.cohort_id,
        ]
    )
    for harness in CANARY_ORDER:
        arguments.extend(
            ["--modal-canary-run", f"{harness}={run_ids[harness]}"]
        )
    monkeypatch.setattr(sys, "argv", arguments)

    assert validator_main() == 0
    selector = json.loads(capsys.readouterr().out)
    assert selector["schema_name"] == "ModalProviderCanaryRunSelector"
    selector_path = (
        project
        / "outputs/readiness/modal_only_final/modal_live_cohorts"
        / identity.source_tree_sha256
        / identity.image_source_sha256
        / identity.cohort_id
        / "provider_canary_selection/provider-canary-cohort-03/"
        "canary_run_selector.json"
    )
    assert selector_path.is_file()


def test_modal_canary_selector_rejects_duplicate_and_symlinked_runs(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    download_root, prefix = _canonical_modal_canary_bundle(project)
    run_ids = _selected_canary_run_ids(download_root, prefix)
    duplicated = dict(run_ids)
    duplicated["semantic_autoresearch"] = duplicated["greedy_autoresearch"]

    with pytest.raises(ValueError):
        create_modal_canary_selector(
            selector_id="provider-canary-cohort-duplicate",
            run_ids=duplicated,
            identity=_selector_identity(
                download_root,
                prefix,
                cohort_id="provider-canary-cohort-duplicate",
            ),
            project_root=project,
        )

    greedy = download_root / run_ids["greedy_autoresearch"]
    backing = download_root / "greedy-backing"
    greedy.rename(backing)
    greedy.symlink_to(backing, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        create_modal_canary_selector(
            selector_id="provider-canary-cohort-symlink",
            run_ids=run_ids,
            identity=_selector_identity(
                download_root,
                prefix,
                cohort_id="provider-canary-cohort-symlink",
            ),
            project_root=project,
        )


@pytest.mark.parametrize("harness", CANARY_ORDER)
def test_downloaded_modal_canaries_reject_remanifested_roster_extensions(
    tmp_path,
    harness,
):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    suffix = harness.replace("_autoresearch", "-ar").replace("_", "-")
    run = download_root / f"{prefix}-{suffix}"

    outer_extra = run / "debug.json"
    _write_json(outer_extra, {"debug": True})
    _refresh_artifact_manifest(run)
    evidence = validate_downloaded_modal_canaries(download_root)
    assert not evidence["valid"]
    assert any("outer roster differs" in error for error in evidence["errors"])

    outer_extra.unlink()
    controller_extra = run / "controller" / "debug.json"
    _write_json(controller_extra, {"debug": True})
    _refresh_artifact_manifest(run)
    evidence = validate_downloaded_modal_canaries(download_root)
    assert not evidence["valid"]
    assert any("controller roster differs" in error for error in evidence["errors"])


@pytest.mark.parametrize("harness", CANARY_ORDER)
def test_downloaded_modal_canaries_reject_remanifested_schema_extensions(
    tmp_path,
    harness,
):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    suffix = harness.replace("_autoresearch", "-ar").replace("_", "-")
    run = download_root / f"{prefix}-{suffix}"
    manifest_path = run / "controller" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected_extension"] = True
    _write_json(manifest_path, manifest)
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)
    assert not evidence["valid"]
    assert any("ControllerRunManifest fields differ" in error for error in evidence["errors"])

    del manifest["unexpected_extension"]
    _write_json(manifest_path, manifest)
    summary_name = (
        "run_summary.json"
        if harness in {"greedy_autoresearch", "semantic_autoresearch"}
        else "run_result.json"
    )
    summary_path = run / "controller" / summary_name
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["unexpected_extension"] = True
    _write_json(summary_path, summary)
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)
    assert not evidence["valid"]
    assert any("controller summary fields differ" in error for error in evidence["errors"])
    del summary["unexpected_extension"]
    _write_json(summary_path, summary)

    if harness in {"greedy_autoresearch", "semantic_autoresearch"}:
        lineage_path = run / "controller" / "lineage.jsonl"
        lineage = [
            json.loads(line)
            for line in lineage_path.read_text(encoding="utf-8").splitlines()
        ]
        lineage[0]["unexpected_extension"] = True
        lineage_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in lineage),
            encoding="utf-8",
        )
        _refresh_artifact_manifest(run)
        evidence = validate_downloaded_modal_canaries(download_root)
        assert not evidence["valid"]
        assert any("native lineage record fields differ" in error for error in evidence["errors"])
    else:
        trace_path = run / "controller" / "evolution_trace.jsonl"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trace["unexpected_extension"] = True
        trace_path.write_text(json.dumps(trace, sort_keys=True) + "\n", encoding="utf-8")
        _refresh_artifact_manifest(run)
        evidence = validate_downloaded_modal_canaries(download_root)
        assert not evidence["valid"]
        assert any("OpenEvolve trace fields differ" in error for error in evidence["errors"])

        del trace["unexpected_extension"]
        trace_path.write_text(json.dumps(trace, sort_keys=True) + "\n", encoding="utf-8")
        metadata_path = (
            run
            / "controller"
            / "checkpoints"
            / "checkpoint_1"
            / "metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["unexpected_extension"] = True
        _write_json(metadata_path, metadata)
        _refresh_artifact_manifest(run)
        evidence = validate_downloaded_modal_canaries(download_root)
        assert not evidence["valid"]
        assert any("checkpoint metadata fields differ" in error for error in evidence["errors"])


def test_downloaded_modal_canaries_reject_a_missing_provider_ledger(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-greedy-ar"
    (run / "controller" / PROVIDER_ATTEMPT_LEDGER_FILENAME).unlink()
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any("lacks required artifacts" in error for error in evidence["errors"])


def test_downloaded_modal_canaries_reject_more_than_one_actual_attempt(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-openevolve-generic"
    ledger = run / "controller" / PROVIDER_ATTEMPT_LEDGER_FILENAME
    attempt = json.loads(ledger.read_text(encoding="utf-8"))
    attempt["attempt_ordinal"] = 2
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(attempt, sort_keys=True, separators=(",", ":")) + "\n"
        )
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any(
        "exactly one actual API attempt" in error for error in evidence["errors"]
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("usage_known", False, "usage_known"),
        ("provider_response_id", None, "response ID"),
        ("provider_request_id", None, "request ID"),
        ("status", "error", "status"),
        ("generation_settings_sha256", "0" * 64, "generation_settings_sha256"),
    ],
)
def test_downloaded_modal_canaries_reject_unattributable_provider_attempts(
    tmp_path,
    field,
    value,
    expected_error,
):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-semantic-ar"
    ledger = run / "controller" / PROVIDER_ATTEMPT_LEDGER_FILENAME
    attempt = json.loads(ledger.read_text(encoding="utf-8"))
    attempt[field] = value
    if field == "usage_known":
        attempt["input_tokens"] = None
        attempt["output_tokens"] = None
        attempt["total_tokens"] = None
    if field == "status":
        attempt["provider_response_id"] = None
        attempt["usage_known"] = False
        attempt["input_tokens"] = None
        attempt["output_tokens"] = None
        attempt["total_tokens"] = None
        attempt["error_class"] = "OfflineProviderFailure"
    ledger.write_text(
        json.dumps(attempt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any(expected_error in error for error in evidence["errors"])


def test_downloaded_modal_canaries_reject_non_reconciling_token_usage(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-openevolve-semantic"
    ledger = run / "controller" / PROVIDER_ATTEMPT_LEDGER_FILENAME
    attempt = json.loads(ledger.read_text(encoding="utf-8"))
    attempt["total_tokens"] += 1
    ledger.write_text(
        json.dumps(attempt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any("token totals do not reconcile" in error for error in evidence["errors"])


def test_downloaded_modal_canaries_reject_extended_or_message_bearing_ledger_schema(
    tmp_path,
):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-greedy-ar"
    ledger = run / "controller" / PROVIDER_ATTEMPT_LEDGER_FILENAME
    attempt = json.loads(ledger.read_text(encoding="utf-8"))
    attempt["error_message"] = "raw provider body is forbidden"
    ledger.write_text(
        json.dumps(attempt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any("unexpected or missing fields" in error for error in evidence["errors"])


@pytest.mark.parametrize(
    ("config_relative_path", "config_kind"),
    [
        ("agents/greedy_autoresearch/config.yaml", "native"),
        ("agents/semantic_autoresearch/config.yaml", "native"),
        ("agents/openevolve_generic/config.yaml", "openevolve"),
        ("agents/openevolve_semantic/config.yaml", "openevolve"),
    ],
    ids=(
        "greedy-autoresearch",
        "semantic-autoresearch",
        "openevolve-generic",
        "openevolve-semantic",
    ),
)
def test_modal_canary_controller_configs_emit_the_frozen_generator_contract(
    config_relative_path: str,
    config_kind: str,
) -> None:
    config = yaml.safe_load((ROOT / config_relative_path).read_text(encoding="utf-8"))
    if config_kind == "native":
        generation_config = config
        model = TARGET_MODEL
        api_base = OFFICIAL_OPENAI_API_BASE
        timeout_field = "timeout_seconds"
        retry_delay_field = "retry_delay_seconds"
    else:
        generation_config = config["llm"]
        models = generation_config["models"]
        assert models == [{"name": TARGET_MODEL, "weight": 1.0}]
        model = models[0]["name"]
        api_base = generation_config["api_base"]
        timeout_field = "timeout"
        retry_delay_field = "retry_delay"

    generation = GPT56SolProfile.resolve(
        model=model,
        seed=1,
        default_reasoning_effort=str(generation_config["reasoning_effort"]),
        default_max_completion_tokens=int(generation_config["max_tokens"]),
        default_timeout_seconds=int(generation_config[timeout_field]),
        default_retries=int(generation_config["retries"]),
        default_retry_delay_seconds=int(generation_config[retry_delay_field]),
        environ={},
        allow_environment_overrides=True,
    )
    endpoint = resolve_provider_endpoint(api_base, scientific=False)
    emitted_contract = {
        **generation.manifest_fields(),
        "api_base_configured": bool(api_base),
        **endpoint.manifest_fields(),
    }

    assert emitted_contract == _MODAL_CANARY_GENERATOR_CONTRACT


def test_downloaded_modal_canary_bundle_accepts_one_retried_attempt(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    original = download_root / f"{prefix}-openevolve-generic"
    _rename_canary_attempt(
        original,
        "modal-canary-retry-1-openevolve-generic",
    )

    evidence = validate_downloaded_modal_canaries(download_root)

    assert evidence["valid"], evidence["errors"]
    assert evidence["recovery_bundle"]
    assert evidence["run_id_prefix"] is None
    assert [run["harness"] for run in evidence["runs"]] == list(CANARY_ORDER)


def test_modal_canary_cli_requirement_accepts_a_valid_local_bundle(
    tmp_path, monkeypatch, capsys
):
    project = _project_fixture(tmp_path)
    download_root, _prefix = _synthetic_modal_canary_bundle(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_engineering_canaries.py",
            "--project-root",
            str(project),
            "--modal-canary-download-root",
            str(download_root),
            "--require-modal-canaries",
        ],
    )

    assert validator_main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["modal_canaries_validated"] is True
    assert report["provider_calls"] == 0
    assert report["training_runs"] == 0


def test_downloaded_modal_canaries_require_exactly_four_frozen_suffixes(tmp_path):
    download_root, _prefix = _synthetic_modal_canary_bundle(tmp_path)
    (download_root / "modal-canary-20260809-extra-harness").mkdir()

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any("exactly four" in error for error in evidence["errors"])


def test_downloaded_modal_canaries_reject_hash_tampering(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-greedy-ar"
    result_path = run / "remote_action_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["stdout_size_bytes"] += 1
    _write_json(result_path, result)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any("mismatch" in error for error in evidence["errors"])


def test_downloaded_modal_canaries_reject_harness_substitution(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-semantic-ar"
    result_path = run / "remote_action_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["harness"] = "greedy_autoresearch"
    _write_json(result_path, result)
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any("substituted harness" in error for error in evidence["errors"])


def test_downloaded_modal_canaries_reject_credential_fields(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-openevolve-generic"
    manifest_path = run / "controller" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator"]["api_key"] = "redacted-test-value"
    _write_json(manifest_path, manifest)
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any("credential fields" in error for error in evidence["errors"])


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("api_endpoint", "https://provider.invalid/v1"),
        ("provider_identity", "openai_compatible_deadbeef"),
        ("model", "gpt-5.6"),
        ("max_completion_tokens", 32_768),
        ("retries", 3),
    ],
    ids=(
        "endpoint",
        "provider-identity",
        "model",
        "completion-ceiling",
        "retries",
    ),
)
def test_downloaded_modal_canaries_reject_generator_contract_tampering(
    tmp_path,
    field,
    tampered_value,
):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-greedy-ar"
    manifest_path = run / "controller" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator"][field] = tampered_value
    _write_json(manifest_path, manifest)
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any(f"generator {field}" in error for error in evidence["errors"])


def test_downloaded_modal_canaries_reject_missing_null_generator_field(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-semantic-ar"
    manifest_path = run / "controller" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["generator"]["top_p"]
    _write_json(manifest_path, manifest)
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any("generator top_p is missing" in error for error in evidence["errors"])


def test_downloaded_modal_canaries_reject_unknown_generator_field(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-openevolve-semantic"
    manifest_path = run / "controller" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator"]["effective_model"] = "different-model"
    _write_json(manifest_path, manifest)
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any("generator fields differ" in error for error in evidence["errors"])


def test_downloaded_modal_canaries_reject_executor_absolute_paths(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-semantic-ar"
    summary_path = run / "controller" / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["lineage_path"] = "/mnt/discovery/runs/private/lineage.jsonl"
    _write_json(summary_path, summary)
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any("executor-absolute path" in error for error in evidence["errors"])


def test_downloaded_modal_canaries_reject_exact_path_key_file_uri(tmp_path):
    download_root, prefix = _synthetic_modal_canary_bundle(tmp_path)
    run = download_root / f"{prefix}-openevolve-semantic"
    result_path = run / "controller" / "run_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["path"] = "file:///mnt/discovery/private/result.json"
    _write_json(result_path, result)
    _refresh_artifact_manifest(run)

    evidence = validate_downloaded_modal_canaries(download_root)

    assert not evidence["valid"]
    assert any("executor-absolute path" in error for error in evidence["errors"])


def test_existing_smoke_accepts_controller_canonicalized_trusted_seed(tmp_path):
    project = _project_fixture(tmp_path)
    output = _synthetic_mps_smoke(project, tmp_path)
    graph_path = output / "candidate_graph.json"
    validation = validate_ir_candidate_json(graph_path.read_text(encoding="utf-8"))
    assert validation.valid and validation.graph is not None
    graph_path.write_text(validation.graph.canonical_json, encoding="utf-8")
    canonical_hash = _sha256(graph_path)

    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["candidate_source_hash"] = canonical_hash
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    manifest_path = output / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_source_hash"] = canonical_hash
    manifest["candidate_artifact_hash"] = canonical_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evidence = validate_existing_mps_smoke(output, project_root=project)

    assert evidence["valid"]
    assert evidence["artifact_self_consistent"]
    assert evidence["parameters_changed"]


def test_smoke_artifact_check_rejects_wrong_checkpoint_shape(tmp_path):
    project = _project_fixture(tmp_path)
    output = _synthetic_mps_smoke(project, tmp_path)
    checkpoint_path = output / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    name = next(iter(checkpoint["model_state"]))
    checkpoint["model_state"][name] = checkpoint["model_state"][name].reshape(-1)
    torch.save(checkpoint, checkpoint_path)
    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_sha256"] = _sha256(checkpoint_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    evidence = validate_existing_mps_smoke(output, project_root=project)

    assert not evidence["valid"]
    assert any("wrong shape" in error for error in evidence["errors"])


def test_mps_smoke_rejects_a_candidate_other_than_the_trusted_seed(tmp_path):
    project = _project_fixture(tmp_path)
    output = _synthetic_mps_smoke(project, tmp_path)
    graph_path = output / "candidate_graph.json"
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    payload["metadata"]["untrusted_change"] = True
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = validate_existing_mps_smoke(output, project_root=project)

    assert not evidence["valid"]
    assert not evidence["artifact_self_consistent"]
    assert any("trusted initial candidate" in error for error in evidence["errors"])
