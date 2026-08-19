"""Validate a private, completed 60-iteration OpenEvolve staging tree."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

from common.gpt56_sol import OFFICIAL_OPENAI_API_BASE, TARGET_MODEL
from common.provider_attempts import (
    PROVIDER_ATTEMPT_LEDGER_FILENAME,
    generation_settings_sha256,
    load_provider_attempt_ledger,
)
from common.runtime_context import ExecutionContextV1
from modal_boundary import (
    OPENEVOLVE_60_ACTION,
    OPENEVOLVE_60_FUNCTION_NAME,
    OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST,
    OPENEVOLVE_60_ITERATIONS,
)

_MAX_JSON_BYTES = 4 * 1024 * 1024
_PROVIDER_LEDGER_ACTION = OPENEVOLVE_60_ACTION.replace("-", "_")


def _read_json_object(path: Path) -> dict[str, Any]:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError(f"OpenEvolve staging artifact is unsafe: {path.name}")
    if details.st_size > _MAX_JSON_BYTES:
        raise ValueError(f"OpenEvolve staging artifact is oversized: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"OpenEvolve staging artifact is not an object: {path.name}")
    return payload


def validate_private_openevolve_60_staging(
    controller_directory: str | Path,
    *,
    execution_context: ExecutionContextV1,
) -> dict[str, Any]:
    """Fail closed unless the staging tree proves the frozen 60-run contract."""

    root = Path(controller_directory)
    details = root.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("OpenEvolve 60 staging root is unsafe")
    if (
        execution_context.execution_backend != "modal"
        or execution_context.function_name != OPENEVOLVE_60_FUNCTION_NAME
        or execution_context.modal_call_id is None
    ):
        raise ValueError("OpenEvolve 60 staging context is not the Modal action")

    manifest = _read_json_object(root / "run_manifest.json")
    result = _read_json_object(root / "run_result.json")
    controller_run_id = manifest.get("run_id")
    if not isinstance(controller_run_id, str) or not controller_run_id:
        raise ValueError("OpenEvolve 60 run manifest lacks its controller run ID")
    expected_manifest = {
        "condition": "openevolve_generic",
        "seed": 1,
        "candidate_budget": OPENEVOLVE_60_ITERATIONS + 1,
        "mutation_budget": OPENEVOLVE_60_ITERATIONS,
        "proposal_opportunities": OPENEVOLVE_60_ITERATIONS,
        "maximum_provider_attempts": OPENEVOLVE_60_ITERATIONS,
        "candidate_training_budget": OPENEVOLVE_60_ITERATIONS + 1,
        "engineering_pilot": True,
        "modal_openevolve_60": True,
        "provider_input_bytes_per_request_ceiling": (
            OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST
        ),
        "authoritative_scientific_evidence": False,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"OpenEvolve 60 manifest field changed: {field}")
    training = manifest.get("training")
    evaluation = manifest.get("evaluation")
    if (
        not isinstance(training, dict)
        or training.get("profile") != "smoke_train_cuda_v2"
        or training.get("device") != "cuda"
        or not isinstance(evaluation, dict)
        or evaluation.get("profile") != "smoke_eval_v1"
        or evaluation.get("case_count") != 24
        or evaluation.get("scientific") is not False
    ):
        raise ValueError("OpenEvolve 60 smoke training/evaluation contract changed")

    expected_result = {
        "run_id": controller_run_id,
        "condition": "openevolve_generic",
        "completed": True,
        "proposal_opportunities_requested": OPENEVOLVE_60_ITERATIONS,
        "proposal_opportunities_completed": OPENEVOLVE_60_ITERATIONS,
        "engineering_pilot": True,
        "authoritative_scientific_evidence": False,
        "failure_stage": "",
    }
    for field, expected in expected_result.items():
        if result.get(field) != expected:
            raise ValueError(f"OpenEvolve 60 result field changed: {field}")
    terminal_iterations = result.get("proposal_terminal_iterations")
    if terminal_iterations != list(range(1, OPENEVOLVE_60_ITERATIONS + 1)):
        raise ValueError("OpenEvolve 60 terminal opportunity roster is incomplete")

    records = load_provider_attempt_ledger(
        root / PROVIDER_ATTEMPT_LEDGER_FILENAME
    )
    if len(records) != OPENEVOLVE_60_ITERATIONS:
        raise ValueError(
            "OpenEvolve 60 did not record exactly one provider attempt per iteration"
        )
    expected_generation_settings = generation_settings_sha256(
        {
            "model": TARGET_MODEL,
            "max_completion_tokens": 16_384,
            "reasoning_effort": "high",
            "seed": 1,
        }
    )
    for record in records:
        if (
            record.harness != "openevolve_generic"
            or record.action != _PROVIDER_LEDGER_ACTION
            or record.controller_run_id != controller_run_id
            or record.execution_backend != "modal"
            or record.action_run_id != execution_context.run_id
            or record.modal_call_id != execution_context.modal_call_id
            or record.api_endpoint != OFFICIAL_OPENAI_API_BASE
            or record.model != TARGET_MODEL
            or record.generation_settings_sha256 != expected_generation_settings
        ):
            raise ValueError("OpenEvolve 60 provider-attempt identity changed")
        if record.usage_known and (
            record.input_tokens is None
            or record.output_tokens is None
            or record.input_tokens > OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST
            or record.output_tokens > 16_384
        ):
            raise ValueError("OpenEvolve 60 provider usage exceeded approval ceilings")

    return {
        "schema_name": "OpenEvolve60StagingValidation",
        "schema_version": "1.0",
        "run_id": execution_context.run_id,
        "controller_run_id": controller_run_id,
        "iterations": OPENEVOLVE_60_ITERATIONS,
        "provider_attempts": len(records),
        "validated": True,
    }
