"""Build the cost-free approval plan for the bounded OpenEvolve 60 action."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from common.gpt56_sol import API_MODE, OFFICIAL_OPENAI_API_BASE, TARGET_MODEL
from modal_boundary import (
    OPENEVOLVE_60_ACTION,
    OPENEVOLVE_60_CONTROLLER_TIMEOUT_SECONDS,
    OPENEVOLVE_60_FUNCTION_TIMEOUT_SECONDS,
    OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST,
    OPENEVOLVE_60_ITERATIONS,
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
CONFIG_PATH = Path("agents/openevolve_generic/config.yaml")
_HASH_SCOPE = "canonical_json_sha256_excluding_approval_plan_sha256"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_openevolve_60_approval_plan(
    project_root: str | Path = ROOT,
    *,
    source_tree_sha256: str,
    cohort_id: str,
    candidate_resume_preflight_receipt_path: str,
    candidate_resume_preflight_receipt_sha256: str,
) -> dict[str, Any]:
    """Return the exact source-bound plan without provider or Modal I/O."""

    root = Path(project_root).resolve()
    if root != ROOT.resolve():
        raise ValueError("OpenEvolve 60 approval is bound to this checkout")
    source = build_image_source_manifest(root)
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=source_tree_sha256,
        image_source_sha256=source.manifest_sha256,
        cohort_id=cohort_id,
    )
    if compute_source_tree_sha256(root) != identity.source_tree_sha256:
        raise ValueError("OpenEvolve 60 approval uses a stale source tree")
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

    config_path = root / CONFIG_PATH
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    llm = config.get("llm") if isinstance(config, dict) else None
    models = llm.get("models") if isinstance(llm, dict) else None
    if (
        not isinstance(models, list)
        or len(models) != 1
        or models[0].get("name") != TARGET_MODEL
        or llm.get("api_base") != OFFICIAL_OPENAI_API_BASE
        or llm.get("max_tokens") != 16_384
        or llm.get("timeout") != 180
        or llm.get("retries") != 0
        or llm.get("retry_delay") != 0
        or llm.get("reasoning_effort") != "high"
    ):
        raise ValueError("OpenEvolve 60 provider settings are not frozen")

    total_input = OPENEVOLVE_60_ITERATIONS * OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST
    total_completion = OPENEVOLVE_60_ITERATIONS * int(llm["max_tokens"])
    plan: dict[str, Any] = {
        "schema_name": "OpenEvolve60ProviderApprovalPlan",
        "schema_version": "1.0",
        "approval_plan_sha256_scope": _HASH_SCOPE,
        "action": OPENEVOLVE_60_ACTION,
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
            "harness": "openevolve_generic",
            "iterations": OPENEVOLVE_60_ITERATIONS,
            "seed": 1,
            "training_profile": "smoke_train_cuda_v2",
            "evaluation_profile": "smoke_eval_v1",
            "evaluation_cases": 24,
            "device": "cuda",
            "scientific": False,
            "config_path": CONFIG_PATH.as_posix(),
            "config_sha256": _file_sha256(config_path),
            "input_guard_path": "common/openevolve_runner.py",
            "input_guard_sha256": _file_sha256(
                root / "common" / "openevolve_runner.py"
            ),
        },
        "request_settings": {
            "reasoning_effort": "high",
            "max_completion_tokens": int(llm["max_tokens"]),
            "timeout_seconds": int(llm["timeout"]),
            "retries": 0,
            "retry_delay_seconds": 0,
            "temperature": None,
            "top_p": None,
            "seed": 1,
        },
        "execution_deadlines": {
            "function_timeout_seconds": OPENEVOLVE_60_FUNCTION_TIMEOUT_SECONDS,
            "controller_subprocess_timeout_seconds": (
                OPENEVOLVE_60_CONTROLLER_TIMEOUT_SECONDS
            ),
            "finalization_reserve_seconds": (
                OPENEVOLVE_60_FUNCTION_TIMEOUT_SECONDS
                - OPENEVOLVE_60_CONTROLLER_TIMEOUT_SECONDS
            ),
        },
        "cost_ceiling": {
            "maximum_requests": OPENEVOLVE_60_ITERATIONS,
            "input_bytes_per_request_ceiling": (
                OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST
            ),
            "conservative_input_tokens_per_request_ceiling": (
                OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST
            ),
            "completion_tokens_per_request_ceiling": int(llm["max_tokens"]),
            "conservative_input_token_ceiling": total_input,
            "requested_completion_token_ceiling": total_completion,
            "input_token_bound_basis": (
                "pre-transport canonical request byte guard; token count cannot "
                "exceed UTF-8 request bytes"
            ),
        },
        "provider_calls_started": 0,
        "modal_calls_started": 0,
        "openai_clients_initialized": 0,
        "claim_scope": "cost_free_pre_request_non_scientific_approval_only",
    }
    plan["approval_plan_sha256"] = canonical_sha256(plan)
    return plan


def verify_openevolve_60_approval_plan(plan: dict[str, Any]) -> str:
    if not isinstance(plan, dict):
        raise TypeError("OpenEvolve 60 approval plan must be an object")
    digest = plan.get("approval_plan_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("OpenEvolve 60 approval plan lacks its SHA-256")
    unsigned = dict(plan)
    del unsigned["approval_plan_sha256"]
    if unsigned.get("approval_plan_sha256_scope") != _HASH_SCOPE:
        raise ValueError("OpenEvolve 60 approval plan hash scope changed")
    if canonical_sha256(unsigned) != digest:
        raise ValueError("OpenEvolve 60 approval plan SHA-256 does not reconstruct")
    return digest


def create_openevolve_60_approval_plan(output: str | Path, **kwargs: Any) -> dict[str, Any]:
    plan = build_openevolve_60_approval_plan(**kwargs)
    verify_openevolve_60_approval_plan(plan)
    create_json_exclusive(Path(output).expanduser(), plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--candidate-resume-preflight-receipt-path", required=True)
    parser.add_argument("--candidate-resume-preflight-receipt-sha256", required=True)
    args = parser.parse_args()
    kwargs = {
        "project_root": args.project_root,
        "source_tree_sha256": args.source_tree_sha256,
        "cohort_id": args.cohort_id,
        "candidate_resume_preflight_receipt_path": (
            args.candidate_resume_preflight_receipt_path
        ),
        "candidate_resume_preflight_receipt_sha256": (
            args.candidate_resume_preflight_receipt_sha256
        ),
    }
    plan = (
        create_openevolve_60_approval_plan(args.output, **kwargs)
        if args.output is not None
        else build_openevolve_60_approval_plan(**kwargs)
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
