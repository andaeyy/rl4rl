"""Validate a private arbitrary-length engineering evolution run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.evolution_run import (
    EVOLUTION_ACTION,
    EVOLUTION_COMPLETION_TOKENS_PER_REQUEST,
    EVOLUTION_FUNCTION_NAME,
    EVOLUTION_INPUT_BYTES_PER_REQUEST,
    EvolutionRunSpec,
)
from common.gpt56_sol import OFFICIAL_OPENAI_API_BASE, TARGET_MODEL
from common.provider_attempts import (
    PROVIDER_ATTEMPT_LEDGER_FILENAME,
    generation_settings_sha256,
    load_provider_attempt_ledger,
)
from common.runtime_context import ExecutionContextV1
from scripts.validate_engineering_canaries import (
    _safe_json_object,
    _validate_modal_canary_generator,
    _validate_private_canary_tree,
    _validate_private_cuda_candidate,
)

_NATIVE = frozenset({"greedy_autoresearch", "semantic_autoresearch"})


def validate_private_evolution_staging(
    controller_directory: str | Path,
    *,
    spec: EvolutionRunSpec,
    execution_context: ExecutionContextV1,
) -> dict[str, Any]:
    """Fail closed unless staging proves the requested engineering run."""

    if not isinstance(spec, EvolutionRunSpec):
        raise TypeError("evolution staging requires an EvolutionRunSpec")
    if (
        execution_context.execution_backend != "modal"
        or execution_context.function_name != EVOLUTION_FUNCTION_NAME
        or execution_context.modal_call_id is None
        or execution_context.modal_image_id is None
    ):
        raise ValueError("evolution staging context is incomplete")
    controller = Path(controller_directory).resolve()
    file_count, total_bytes = _validate_private_canary_tree(controller)
    manifest = _safe_json_object(controller / "run_manifest.json")
    summary_name = "run_summary.json" if spec.harness in _NATIVE else "run_result.json"
    summary = _safe_json_object(controller / summary_name)
    controller_run_id = manifest.get("run_id")
    if (
        not isinstance(controller_run_id, str)
        or summary.get("run_id") != controller_run_id
        or manifest.get("condition") != spec.harness
        or summary.get("condition") != spec.harness
        or manifest.get("modal_evolution_run") is not True
        or manifest.get("provider_input_bytes_per_request_ceiling")
        != EVOLUTION_INPUT_BYTES_PER_REQUEST
    ):
        raise ValueError("evolution controller identity changed")
    for field, expected in {
        "candidate_budget": spec.iterations + 1,
        "mutation_budget": spec.iterations,
        "maximum_provider_attempts": spec.iterations,
        "candidate_training_budget": spec.iterations + 1,
    }.items():
        if manifest.get(field) != expected:
            raise ValueError(f"evolution manifest {field} changed")
    if manifest.get("authoritative_scientific_evidence") is not False:
        raise ValueError("evolution run claims scientific authority")
    training = manifest.get("training")
    evaluation = manifest.get("evaluation")
    if (
        not isinstance(training, dict)
        or training.get("profile") != "smoke_train_cuda_v2"
        or training.get("device") != "cuda"
        or training.get("allow_cpu_for_tests") is not False
        or not isinstance(evaluation, dict)
        or evaluation.get("profile") != "smoke_eval_v1"
        or evaluation.get("case_count") != 24
        or evaluation.get("scientific") is not False
    ):
        raise ValueError("evolution smoke training/evaluation contract changed")
    _validate_modal_canary_generator(manifest.get("generator"))

    if spec.harness in _NATIVE:
        if (
            manifest.get("run_mode") != "engineering_pilot"
            or manifest.get("exploratory_only") is not True
            or summary.get("proposal_opportunities_requested") != spec.iterations
            or summary.get("proposal_opportunities_terminal") != spec.iterations
        ):
            raise ValueError("Autoresearch opportunity accounting changed")
    elif (
        manifest.get("engineering_pilot") is not True
        or manifest.get("proposal_opportunities") != spec.iterations
        or summary.get("completed") is not True
        or summary.get("proposal_opportunities_requested") != spec.iterations
        or summary.get("proposal_opportunities_completed") != spec.iterations
        or summary.get("proposal_accounting_errors") != []
        or summary.get("failure_stage") != ""
    ):
        raise ValueError("OpenEvolve opportunity accounting changed")

    records = load_provider_attempt_ledger(
        controller / PROVIDER_ATTEMPT_LEDGER_FILENAME
    )
    expected_settings = generation_settings_sha256(
        {
            "model": TARGET_MODEL,
            "max_completion_tokens": (
                EVOLUTION_COMPLETION_TOKENS_PER_REQUEST
            ),
            "reasoning_effort": "high",
            "seed": 1,
        }
    )
    if len(records) != spec.iterations:
        raise ValueError("evolution run lacks one provider attempt per iteration")
    for ordinal, record in enumerate(records, start=1):
        if (
            record.harness != spec.harness
            or record.action != EVOLUTION_ACTION.replace("-", "_")
            or record.controller_run_id != controller_run_id
            or record.execution_backend != "modal"
            or record.action_run_id != execution_context.run_id
            or record.modal_call_id != execution_context.modal_call_id
            or record.attempt_ordinal != ordinal
            or record.status != "success"
            or record.api_endpoint != OFFICIAL_OPENAI_API_BASE
            or record.model != TARGET_MODEL
            or record.generation_settings_sha256 != expected_settings
            or record.usage_known is not True
            or record.input_tokens is None
            or record.output_tokens is None
            or record.input_tokens > EVOLUTION_INPUT_BYTES_PER_REQUEST
            or record.output_tokens > EVOLUTION_COMPLETION_TOKENS_PER_REQUEST
        ):
            raise ValueError("evolution provider-attempt contract changed")

    training_root = controller / "candidate_training"
    candidate_directories = tuple(sorted(training_root.iterdir()))
    if not 1 <= len(candidate_directories) <= spec.iterations + 1:
        raise ValueError("evolution candidate-training count is invalid")
    for candidate in candidate_directories:
        _validate_private_cuda_candidate(
            candidate,
            execution_context=execution_context,
        )
    return {
        "schema_name": "PrivateEvolutionStagingValidation",
        "schema_version": "1.0",
        "valid": True,
        "evolution_spec": spec.token,
        "controller_run_id": controller_run_id,
        "candidate_count": len(candidate_directories),
        "provider_attempt_count": len(records),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }
