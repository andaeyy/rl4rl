from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import modal_app
from common.openevolve_runner import _BoundedOpenEvolveClient
from common.provider_attempts import (
    ProviderAttemptRecord,
    generation_settings_sha256,
)
from common.runtime_context import ExecutionContextV1
from modal_boundary import (
    OPENEVOLVE_60_ACTION,
    OPENEVOLVE_60_FUNCTION_NAME,
    OPENEVOLVE_60_FUNCTION_TIMEOUT_SECONDS,
    OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST,
    OPENEVOLVE_60_ITERATIONS,
    ModalLiveCohortIdentity,
    build_image_source_manifest,
    build_modal_cli_command,
    function_spec,
    modal_live_cohort_root,
)
from openevolve.config import LLMModelConfig
from scripts import launch_modal
from scripts.openevolve_60_plan import (
    build_openevolve_60_approval_plan,
    verify_openevolve_60_approval_plan,
)
from scripts.record_local_engineering_evidence import source_tree_sha256
from scripts.validate_openevolve_60 import (
    validate_private_openevolve_60_staging,
)

ROOT = Path(__file__).resolve().parents[1]


def _context() -> ExecutionContextV1:
    return ExecutionContextV1(
        execution_backend="modal",
        run_id="oe60-run-1",
        app_name="rl4rl-architecture-discovery",
        function_name=OPENEVOLVE_60_FUNCTION_NAME,
        modal_app_id="ap-oe60",
        modal_function_id="fu-oe60",
        modal_call_id="fc-oe60",
        modal_image_id="im-oe60",
        image_source_sha256="a" * 64,
        artifact_uri="volume://rl4rl-architecture-artifacts/runs/oe60-run-1",
    )


def test_modal_contract_exposes_a_separate_bounded_60_action() -> None:
    spec = function_spec(OPENEVOLVE_60_FUNCTION_NAME)
    assert spec.timeout_seconds == OPENEVOLVE_60_FUNCTION_TIMEOUT_SECONDS
    assert spec.gpu == "T4"
    assert spec.provider_secret is True
    assert spec.retries == 0
    assert spec.max_containers == 1
    assert launch_modal.expected_outer_cli_timeout_seconds(OPENEVOLVE_60_ACTION) == 16_200
    profile = launch_modal.modal_resource_profile(OPENEVOLVE_60_ACTION)
    assert profile["runtime_function_calls"] == [
        {
            "function_name": OPENEVOLVE_60_FUNCTION_NAME,
            "call_count": 1,
            "cpu_request_cores": 2.0,
            "cpu_soft_limit_cores": 2.0,
            "memory_request_mib": 8192,
            "memory_limit_mib": 8192,
            "gpu": "T4",
            "region": None,
            "timeout_seconds": OPENEVOLVE_60_FUNCTION_TIMEOUT_SECONDS,
            "max_containers": 1,
            "min_containers": 0,
            "retries": 0,
            "provider_secret_attached": True,
            "network_mode": "provider_egress_enabled",
        }
    ]


def test_modal_command_requires_provider_approval_for_60_action() -> None:
    kwargs = {
        "python_executable": ROOT / ".venv" / "bin" / "python",
        "project_root": ROOT,
        "action": OPENEVOLVE_60_ACTION,
        "run_id": "oe60-run-1",
        "source_tree_sha256": "a" * 64,
        "cohort_id": "oe60-cohort-1",
        "image_source_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="provider approval differs"):
        build_modal_cli_command(**kwargs, provider_approved=False)
    command = build_modal_cli_command(**kwargs, provider_approved=True)
    assert "--provider-approved" in command
    assert command[command.index("--action") + 1] == OPENEVOLVE_60_ACTION


def test_modal_action_constructs_the_frozen_60_iteration_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_command(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return {"returncode": 0}

    monkeypatch.setattr(modal_app, "_run_command", fake_run_command)
    result = modal_app._openevolve_generic_60_action(tmp_path, _context())

    command = captured["command"]
    assert command[command.index("--iterations") + 1] == "60"
    assert command[command.index("--seed") + 1] == "1"
    assert command[command.index("--training-profile") + 1] == (
        "smoke_train_cuda_v2"
    )
    assert command[command.index("--evaluation-profile") + 1] == "smoke_eval_v1"
    assert "--modal-openevolve-60" in command
    assert captured["kwargs"]["provider"] is True
    assert captured["kwargs"]["timeout_seconds"] == 15_000
    assert captured["kwargs"]["function_timeout_seconds"] == 15_300
    assert result["iterations"] == 60
    assert result["scientific"] is False


def test_bounded_client_rejects_oversized_prompt_before_transport() -> None:
    client = _BoundedOpenEvolveClient(
        LLMModelConfig(
            name="gpt-5.6-sol",
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            max_tokens=16_384,
            timeout=180,
            retries=0,
            retry_delay=0,
            reasoning_effort="high",
        )
    )
    params = {
        "model": "gpt-5.6-sol",
        "messages": [
            {"role": "user", "content": "x" * OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST}
        ],
    }
    with pytest.raises(ValueError, match="input-byte ceiling"):
        asyncio.run(client._call_api(params))


def test_private_staging_validator_accepts_complete_60_call_run(
    tmp_path: Path,
) -> None:
    context = _context()
    controller = tmp_path / "controller"
    controller.mkdir()
    controller_run_id = "openevolve-generic-seed-1-deadbeef"
    manifest = {
        "run_id": controller_run_id,
        "condition": "openevolve_generic",
        "seed": 1,
        "candidate_budget": 61,
        "mutation_budget": 60,
        "proposal_opportunities": 60,
        "maximum_provider_attempts": 60,
        "candidate_training_budget": 61,
        "engineering_pilot": True,
        "modal_openevolve_60": True,
        "provider_input_bytes_per_request_ceiling": (
            OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST
        ),
        "authoritative_scientific_evidence": False,
        "training": {"profile": "smoke_train_cuda_v2", "device": "cuda"},
        "evaluation": {
            "profile": "smoke_eval_v1",
            "case_count": 24,
            "scientific": False,
        },
    }
    result = {
        "run_id": controller_run_id,
        "condition": "openevolve_generic",
        "completed": True,
        "proposal_opportunities_requested": 60,
        "proposal_opportunities_completed": 60,
        "proposal_terminal_iterations": list(range(1, 61)),
        "engineering_pilot": True,
        "authoritative_scientific_evidence": False,
        "failure_stage": "",
    }
    (controller / "run_manifest.json").write_text(json.dumps(manifest))
    (controller / "run_result.json").write_text(json.dumps(result))
    settings_sha256 = generation_settings_sha256(
        {
            "model": "gpt-5.6-sol",
            "max_completion_tokens": 16_384,
            "reasoning_effort": "high",
            "seed": 1,
        }
    )
    records = []
    for ordinal in range(1, 61):
        records.append(
            ProviderAttemptRecord(
                schema_name="ProviderAttemptRecord",
                schema_version="1.0",
                harness="openevolve_generic",
                action="openevolve_generic_60",
                controller_run_id=controller_run_id,
                execution_backend="modal",
                action_run_id=context.run_id,
                modal_call_id=context.modal_call_id,
                attempt_ordinal=ordinal,
                started_at_utc="2026-08-18T00:00:00.000000Z",
                ended_at_utc="2026-08-18T00:00:01.000000Z",
                status="success",
                api_endpoint="https://api.openai.com/v1",
                model="gpt-5.6-sol",
                generation_settings_sha256=settings_sha256,
                provider_response_id=f"response-{ordinal}",
                provider_request_id=f"request-{ordinal}",
                usage_known=True,
                input_tokens=100,
                output_tokens=100,
                total_tokens=200,
                error_class=None,
            ).to_dict()
        )
    (controller / "provider_attempts.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    validated = validate_private_openevolve_60_staging(
        controller,
        execution_context=context,
    )
    assert validated["validated"] is True
    assert validated["iterations"] == OPENEVOLVE_60_ITERATIONS
    assert validated["provider_attempts"] == 60


def test_approval_plan_is_source_bound_and_self_hashing() -> None:
    source = build_image_source_manifest(ROOT)
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=source_tree_sha256(ROOT),
        image_source_sha256=source.manifest_sha256,
        cohort_id="oe60-plan-cohort",
    )
    preflight = (
        modal_live_cohort_root(identity)
        / "components"
        / "candidate_resume_preflight_receipts"
        / "v2.0"
        / f"{'c' * 64}.json"
    )
    plan = build_openevolve_60_approval_plan(
        ROOT,
        source_tree_sha256=identity.source_tree_sha256,
        cohort_id=identity.cohort_id,
        candidate_resume_preflight_receipt_path=preflight.as_posix(),
        candidate_resume_preflight_receipt_sha256="d" * 64,
    )
    assert verify_openevolve_60_approval_plan(plan) == plan["approval_plan_sha256"]
    assert plan["cost_ceiling"]["maximum_requests"] == 60
    assert plan["controller"]["scientific"] is False
