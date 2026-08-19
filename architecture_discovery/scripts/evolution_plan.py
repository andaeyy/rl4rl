"""Build a source-bound provider approval plan for one evolution run."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from common.evolution_run import (
    EVOLUTION_ACTION,
    EVOLUTION_COMPLETION_TOKENS_PER_REQUEST,
    EVOLUTION_INPUT_BYTES_PER_REQUEST,
    EvolutionRunSpec,
)
from common.gpt56_sol import API_MODE, OFFICIAL_OPENAI_API_BASE, TARGET_MODEL
from modal_boundary import (
    ModalLiveCohortIdentity,
    build_image_source_manifest,
    canonical_sha256,
    modal_live_cohort_root,
)
from scripts.record_local_engineering_evidence import (
    source_tree_sha256 as compute_source_tree_sha256,
)
from study.serialization import create_json_exclusive

ROOT = Path(__file__).resolve().parents[1]
_HASH_SCOPE = "canonical_json_sha256_excluding_approval_plan_sha256"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provider_settings(root: Path, harness: str) -> tuple[Path, dict[str, Any]]:
    config_path = root / "agents" / harness / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("evolution controller config is invalid")
    if harness.startswith("openevolve_"):
        llm = config.get("llm")
        models = llm.get("models") if isinstance(llm, dict) else None
        if not isinstance(models, list) or len(models) != 1:
            raise ValueError("OpenEvolve model roster is not frozen")
        settings = {
            "model": models[0].get("name"),
            "api_base": llm.get("api_base"),
            "max_completion_tokens": llm.get("max_tokens"),
            "timeout_seconds": llm.get("timeout"),
            "retries": llm.get("retries"),
            "retry_delay_seconds": llm.get("retry_delay"),
            "reasoning_effort": llm.get("reasoning_effort"),
        }
    else:
        settings = {
            "model": TARGET_MODEL,
            "api_base": OFFICIAL_OPENAI_API_BASE,
            "max_completion_tokens": config.get("max_tokens"),
            "timeout_seconds": config.get("timeout_seconds"),
            "retries": config.get("retries"),
            "retry_delay_seconds": config.get("retry_delay_seconds"),
            "reasoning_effort": config.get("reasoning_effort"),
        }
    if settings != {
        "model": TARGET_MODEL,
        "api_base": OFFICIAL_OPENAI_API_BASE,
        "max_completion_tokens": EVOLUTION_COMPLETION_TOKENS_PER_REQUEST,
        "timeout_seconds": 180,
        "retries": 0,
        "retry_delay_seconds": 0,
        "reasoning_effort": "high",
    }:
        raise ValueError("evolution provider settings are not frozen")
    return config_path, settings


def build_evolution_approval_plan(
    project_root: str | Path = ROOT,
    *,
    source_tree_sha256: str,
    cohort_id: str,
    candidate_resume_preflight_receipt_path: str,
    candidate_resume_preflight_receipt_sha256: str,
    evolution_spec: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if root != ROOT.resolve():
        raise ValueError("evolution approval is bound to this checkout")
    spec = EvolutionRunSpec.parse(evolution_spec)
    source = build_image_source_manifest(root)
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=source_tree_sha256,
        image_source_sha256=source.manifest_sha256,
        cohort_id=cohort_id,
    )
    if compute_source_tree_sha256(root) != identity.source_tree_sha256:
        raise ValueError("evolution approval uses a stale source tree")
    if re.fullmatch(r"[0-9a-f]{64}", candidate_resume_preflight_receipt_sha256) is None:
        raise ValueError("candidate preflight digest must be a SHA-256")
    preflight = PurePosixPath(candidate_resume_preflight_receipt_path)
    expected_parent = (
        modal_live_cohort_root(identity)
        / "components"
        / "candidate_resume_preflight_receipts"
        / "v2.0"
    )
    if (
        preflight.is_absolute()
        or preflight.parent != expected_parent
        or re.fullmatch(r"[0-9a-f]{64}\.json", preflight.name) is None
    ):
        raise ValueError("candidate preflight path is outside the selected cohort")
    config_path, settings = _provider_settings(root, spec.harness)
    runner_path = (
        root / "common" / "openevolve_runner.py"
        if spec.harness.startswith("openevolve_")
        else root / "agents" / spec.harness / "run.py"
    )
    plan: dict[str, Any] = {
        "schema_name": "EvolutionProviderApprovalPlan",
        "schema_version": "1.0",
        "approval_plan_sha256_scope": _HASH_SCOPE,
        "action": EVOLUTION_ACTION,
        "evolution_spec": spec.token,
        "source_tree_sha256": identity.source_tree_sha256,
        "image_source_sha256": identity.image_source_sha256,
        "cohort_id": identity.cohort_id,
        "dependency_lock_sha256": source.dependency_lock_sha256,
        "candidate_resume_preflight_receipt": {
            "path": preflight.as_posix(),
            "sha256": candidate_resume_preflight_receipt_sha256,
        },
        "provider": {
            "identity": "openai_official",
            "api_mode": API_MODE,
            "api_endpoint": OFFICIAL_OPENAI_API_BASE,
            "model": TARGET_MODEL,
        },
        "controller": {
            "harness": spec.harness,
            "iterations": spec.iterations,
            "seed": 1,
            "training_profile": "smoke_train_cuda_v2",
            "evaluation_profile": "smoke_eval_v1",
            "evaluation_cases": 24,
            "device": "cuda",
            "scientific": False,
            "config_path": config_path.relative_to(root).as_posix(),
            "config_sha256": _file_sha256(config_path),
            "runner_path": runner_path.relative_to(root).as_posix(),
            "runner_sha256": _file_sha256(runner_path),
        },
        "request_settings": {
            "reasoning_effort": settings["reasoning_effort"],
            "max_completion_tokens": settings["max_completion_tokens"],
            "timeout_seconds": settings["timeout_seconds"],
            "retries": 0,
            "retry_delay_seconds": 0,
            "temperature": None,
            "top_p": None,
            "seed": 1,
        },
        "execution_deadlines": {
            "function_timeout_seconds": spec.function_timeout_seconds,
            "controller_subprocess_timeout_seconds": (
                spec.controller_timeout_seconds
            ),
            "outer_cli_timeout_seconds": spec.outer_cli_timeout_seconds,
        },
        "cost_ceiling": {
            "maximum_requests": spec.iterations,
            "input_bytes_per_request_ceiling": (
                EVOLUTION_INPUT_BYTES_PER_REQUEST
            ),
            "conservative_input_token_ceiling": (
                spec.iterations * EVOLUTION_INPUT_BYTES_PER_REQUEST
            ),
            "requested_completion_token_ceiling": (
                spec.iterations * EVOLUTION_COMPLETION_TOKENS_PER_REQUEST
            ),
        },
        "provider_calls_started": 0,
        "modal_calls_started": 0,
        "openai_clients_initialized": 0,
        "claim_scope": "cost_free_pre_request_non_scientific_approval_only",
    }
    plan["approval_plan_sha256"] = canonical_sha256(plan)
    return plan

def verify_evolution_approval_plan(plan: dict[str, Any]) -> str:
    if not isinstance(plan, dict):
        raise TypeError("evolution approval plan must be an object")
    digest = plan.get("approval_plan_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("evolution approval plan lacks its SHA-256")
    unsigned = dict(plan)
    del unsigned["approval_plan_sha256"]
    if unsigned.get("approval_plan_sha256_scope") != _HASH_SCOPE:
        raise ValueError("evolution approval plan hash scope changed")
    if canonical_sha256(unsigned) != digest:
        raise ValueError("evolution approval plan SHA-256 does not reconstruct")
    return digest


def create_evolution_approval_plan(output: str | Path, **kwargs: Any) -> dict[str, Any]:
    plan = build_evolution_approval_plan(**kwargs)
    verify_evolution_approval_plan(plan)
    create_json_exclusive(Path(output).expanduser(), plan)
    return plan
