"""Build the four-request provider canary approval plan without provider I/O.

The plan executes the checked-in first-opportunity prompt constructors against
two bounded seed-evaluation endpoint states.  Those states differ in every
live numeric value that can affect prompt text.  Equal canonical request sizes
prove the byte count is invariant for the frozen engineering contract; the
hash remains explicitly scoped to the deterministic zero-state approval
template because the live seed evaluation has not happened yet.

No OpenAI client is constructed, no credential is read, and no network or
Modal operation is available from this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
VENDORED_OPENEVOLVE = ROOT / "vendor" / "openevolve"
for import_root in (ROOT, VENDORED_OPENEVOLVE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from architecture_ir import validate_ir_candidate_json  # noqa: E402
from common.descriptor_extractor import extract_ir_descriptors  # noqa: E402
from common.descriptor_schema import SEMANTIC_METRIC_NAMES  # noqa: E402
from common.gpt56_sol import (  # noqa: E402
    API_MODE,
    OFFICIAL_OPENAI_API_BASE,
    TARGET_MODEL,
    GPT56SolProfile,
)
from common.openevolve_policy import canonical_combined_score  # noqa: E402
from common.provider_attempts import generation_settings_sha256  # noqa: E402
from evaluation.records import ControllerSearchView  # noqa: E402
from modal_boundary import (  # noqa: E402
    CANARY_ORDER,
    CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS,
    FUNCTION_TIMEOUT_SECONDS,
    ModalLiveCohortIdentity,
    PROVIDER_ATTEMPT_FINALIZATION_RESERVE_SECONDS,
    PROVIDER_REQUEST_TIMEOUT_SECONDS,
    build_image_source_manifest,
    canonical_sha256,
    modal_live_cohort_root,
)
from scripts.record_local_engineering_evidence import (  # noqa: E402
    source_tree_sha256 as compute_source_tree_sha256,
)
from study.serialization import create_json_exclusive  # noqa: E402

_INPUT_TOKEN_QUANTUM = 8_192
_EXPECTED_HARNESS_CONFIGS = {
    "greedy_autoresearch": Path("agents/greedy_autoresearch/config.yaml"),
    "semantic_autoresearch": Path("agents/semantic_autoresearch/config.yaml"),
    "openevolve_generic": Path("agents/openevolve_generic/config.yaml"),
    "openevolve_semantic": Path("agents/openevolve_semantic/config.yaml"),
}
_APPROVAL_PLAN_SHA256_SCOPE = (
    "canonical_json_sha256_of_complete_payload_excluding_approval_plan_sha256"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_request(request: dict[str, Any]) -> bytes:
    return json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _message_content_bytes(messages: list[dict[str, str]]) -> int:
    return sum(len(message["content"].encode("utf-8")) for message in messages)


def _config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"provider canary config must be an object: {path}")
    return payload


def _resolved_profile(config: dict[str, Any], *, openevolve: bool) -> GPT56SolProfile:
    generation = config.get("llm") if openevolve else config
    if not isinstance(generation, dict):
        raise ValueError("provider canary configuration lacks generation settings")
    model = TARGET_MODEL
    if openevolve:
        models = generation.get("models")
        if (
            not isinstance(models, list)
            or len(models) != 1
            or not isinstance(models[0], dict)
            or models[0].get("name") != TARGET_MODEL
        ):
            raise ValueError("OpenEvolve canary must contain exactly the pinned model")
        if generation.get("api_base") != OFFICIAL_OPENAI_API_BASE:
            raise ValueError("OpenEvolve canary is not pinned to the official endpoint")
    profile = GPT56SolProfile.resolve(
        model=model,
        seed=1,
        default_reasoning_effort=str(generation["reasoning_effort"]),
        default_max_completion_tokens=int(
            generation["max_tokens"]
        ),
        default_timeout_seconds=int(
            generation["timeout"] if openevolve else generation["timeout_seconds"]
        ),
        default_retries=int(generation["retries"]),
        default_retry_delay_seconds=int(
            generation["retry_delay"]
            if openevolve
            else generation["retry_delay_seconds"]
        ),
        environ={},
        allow_environment_overrides=False,
    )
    if (
        profile.seed != 1
        or profile.reasoning_effort != "high"
        or profile.retries != 0
        or profile.retry_delay_seconds != 0
        or profile.max_completion_tokens != 16_384
        or profile.timeout_seconds != PROVIDER_REQUEST_TIMEOUT_SECONDS
    ):
        raise ValueError("provider canary generation settings are not frozen")
    if generation.get("temperature") is not None or generation.get("top_p") is not None:
        raise ValueError("provider canary may not enable sampling controls")
    return profile


def _native_messages(
    harness: str,
    *,
    endpoint_value: float,
) -> list[dict[str, str]]:
    candidate = ROOT / "common" / "initial_candidate.ir.json"
    if harness == "greedy_autoresearch":
        from agents.greedy_autoresearch.run import (
            _load_initial_ir,
            _load_prompt_protocol,
            _prompt_for_incumbent,
        )

        initial_ir = _load_initial_ir(candidate, max_ir_bytes=40_000)
        system_prompt, _components, _manifest = _load_prompt_protocol()
        return _prompt_for_incumbent(
            system_prompt=system_prompt,
            incumbent_ir=initial_ir,
            incumbent_score=endpoint_value,
            opportunity=1,
        )
    if harness != "semantic_autoresearch":
        raise ValueError("native prompt requested for a non-native harness")
    from agents.semantic_autoresearch.run import (
        ArchiveCandidate,
        FrozenSemanticArchive,
        _load_initial_ir,
        _load_prompt_protocol,
        _prompt_for_parent,
    )

    initial_ir = _load_initial_ir(candidate, max_ir_bytes=40_000)
    validation = validate_ir_candidate_json(initial_ir)
    if not validation.valid or validation.graph is None:
        raise ValueError("semantic canary seed is not valid Architecture IR")
    descriptors = extract_ir_descriptors(validation.graph)
    archive = FrozenSemanticArchive()
    view = ControllerSearchView(
        schema_name="search_evaluation",
        schema_version="1.0",
        record_id="approval-template-record",
        run_id="approval-template-run",
        condition_id="native-semantic-autoresearch",
        candidate_id="candidate-" + ("0" * 64),
        execution_ok=True,
        transformer_valid=True,
        public_accuracy=endpoint_value,
        search_score=endpoint_value,
        eligible_for_parent=True,
        failure_stage="",
        infrastructure_failure=False,
        online_descriptor_codes=tuple(sorted(descriptors.codes.items())),
    )
    archive.consider(
        candidate_id="0" * 64,
        lineage_record_id="approval-template-lineage",
        source_path=candidate,
        view=view,
        opportunity=0,
    )
    parent: ArchiveCandidate = archive.select_parent()
    system_prompt, _components, _manifest = _load_prompt_protocol()
    return _prompt_for_parent(
        system_prompt=system_prompt,
        parent=parent,
        parent_ir=initial_ir,
        archive=archive,
        opportunity=1,
    )


def _openevolve_messages(
    harness: str,
    *,
    endpoint_value: float,
    record_character: str,
) -> list[dict[str, str]]:
    from openevolve.config import load_config
    from openevolve.database import Program
    from openevolve.prompt.sampler import PromptSampler

    kind = harness.removeprefix("openevolve_")
    if kind not in {"generic", "semantic"}:
        raise ValueError("OpenEvolve prompt requested for a non-OpenEvolve harness")
    agent = ROOT / "agents" / f"openevolve_{kind}"
    config = load_config(agent / "config.yaml")
    config.prompt.system_message = "\n\n".join(
        (
            (agent / "system_prompt.md").read_text(encoding="utf-8"),
            (ROOT / "common" / "prompts" / "architecture_ir_contract.md").read_text(
                encoding="utf-8"
            ),
        )
    )
    config.prompt.template_dir = str(agent / "templates")
    config.prompt.use_template_stochasticity = False
    if config.diff_based_evolution or config.language != "json":
        raise ValueError("OpenEvolve canary is not a full-document JSON constructor")

    code = (ROOT / "common" / "initial_candidate.ir.json").read_text(
        encoding="utf-8"
    )
    validation = validate_ir_candidate_json(code)
    if not validation.valid or validation.graph is None:
        raise ValueError("OpenEvolve canary seed is not valid Architecture IR")
    descriptors = extract_ir_descriptors(validation.graph)
    metrics: dict[str, float] = {
        "execution_ok": 1.0,
        "transformer_valid": 1.0,
        "public_accuracy": endpoint_value,
        "search_score": endpoint_value,
        "eligible_for_parent": 1.0,
    }
    metrics.update({name: 0.0 for name in SEMANTIC_METRIC_NAMES.values()})
    metrics.update(descriptors.codes)
    metrics["combined_score"] = canonical_combined_score(metrics)
    artifacts: dict[str, object] = {
        "layer_a_record_id": "search_evaluation-" + (record_character * 32),
        "candidate_graph_hash": validation.graph_hash,
        "candidate_architecture_hash": validation.graph.architecture_hash,
        "parent_architecture_hash": None,
        "failure_stage": "",
        "infrastructure_failure": False,
    }
    parent = Program(
        id="approval-template-parent",
        code=code,
        changes_description=config.prompt.initial_changes_description,
        language="json",
        metrics=metrics,
        metadata={"island": 0},
        artifacts_json=json.dumps(artifacts, sort_keys=True),
    )
    prompt = PromptSampler(config.prompt).build_prompt(
        current_program=parent.code,
        parent_program=parent.code,
        program_metrics=parent.metrics,
        previous_programs=[parent.to_dict()],
        top_programs=[parent.to_dict()],
        inspirations=[],
        language=config.language,
        evolution_round=1,
        diff_based_evolution=False,
        program_artifacts=artifacts,
        feature_dimensions=config.database.feature_dimensions,
        current_changes_description=None,
    )
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]


def _messages_at_endpoint(
    harness: str,
    *,
    endpoint_value: float,
) -> list[dict[str, str]]:
    if harness.startswith("openevolve_"):
        return _openevolve_messages(
            harness,
            endpoint_value=endpoint_value,
            record_character="a" if endpoint_value == 0.0 else "b",
        )
    return _native_messages(harness, endpoint_value=endpoint_value)


def _harness_plan(
    project_root: Path,
    harness: str,
) -> dict[str, Any]:
    relative_config = _EXPECTED_HARNESS_CONFIGS[harness]
    config_path = project_root / relative_config
    config = _config(config_path)
    openevolve = harness.startswith("openevolve_")
    profile = _resolved_profile(config, openevolve=openevolve)
    low_messages = _messages_at_endpoint(harness, endpoint_value=0.0)
    high_messages = _messages_at_endpoint(harness, endpoint_value=1.0)
    low_request = profile.chat_completion_request(low_messages)
    high_request = profile.chat_completion_request(high_messages)
    low_payload = _canonical_request(low_request)
    high_payload = _canonical_request(high_request)
    low_message_bytes = _message_content_bytes(low_messages)
    high_message_bytes = _message_content_bytes(high_messages)
    size_invariant = (
        len(low_payload) == len(high_payload)
        and low_message_bytes == high_message_bytes
    )
    if not openevolve and not size_invariant:
        raise ValueError(
            f"{harness} first-opportunity size depends on live evaluation values"
        )
    request_bytes_upper_bound = max(len(low_payload), len(high_payload))
    message_bytes_upper_bound = max(low_message_bytes, high_message_bytes)
    input_ceiling = (
        math.ceil(request_bytes_upper_bound / _INPUT_TOKEN_QUANTUM)
        * _INPUT_TOKEN_QUANTUM
    )
    if input_ceiling < request_bytes_upper_bound:
        raise AssertionError("provider input-token ceiling is not conservative")
    return {
        "harness": harness,
        "config_path": relative_config.as_posix(),
        "config_sha256": _sha256_file(config_path),
        "api_mode": API_MODE,
        "api_endpoint": OFFICIAL_OPENAI_API_BASE,
        "model": TARGET_MODEL,
        "generation_settings_sha256": generation_settings_sha256(low_request),
        "request_settings": {
            "reasoning_effort": profile.reasoning_effort,
            "max_completion_tokens": profile.max_completion_tokens,
            "seed": profile.seed,
            "timeout_seconds": profile.timeout_seconds,
            "retries": profile.retries,
            "retry_delay_seconds": profile.retry_delay_seconds,
            "temperature": None,
            "top_p": None,
        },
        "maximum_attempts": 1,
        "first_opportunity": {
            "constructor": (
                "vendored_openevolve_prompt_sampler"
                if openevolve
                else "native_controller_prompt_constructor"
            ),
            "approval_template_message_content_bytes": low_message_bytes,
            "approval_template_request_payload_bytes": len(low_payload),
            "request_payload_sha256": _sha256_bytes(low_payload),
            "request_payload_sha256_scope": (
                "canonical_zero_state_approval_template"
            ),
            "live_request_payload_sha256": None,
            "live_hash_unavailable_reason": (
                "the paid request follows live CUDA seed evaluation; approval "
                "does not fabricate its metric values or record ID"
            ),
            "live_message_content_bytes_exact": (
                low_message_bytes if size_invariant else None
            ),
            "live_message_content_bytes_upper_bound": message_bytes_upper_bound,
            "live_request_payload_bytes_exact": (
                len(low_payload) if size_invariant else None
            ),
            "live_request_payload_bytes_upper_bound": request_bytes_upper_bound,
            "size_bound_derivation": (
                "exact_across_bounded_seed_metric_endpoints"
                if size_invariant
                else "maximum_of_actual_prompt_constructor_outputs_at_bounded_"
                "seed_metric_endpoints"
            ),
            "size_invariance_check": {
                "endpoint_states": [0.0, 1.0],
                "same_message_content_bytes": (
                    low_message_bytes == high_message_bytes
                ),
                "same_canonical_request_bytes": (
                    len(low_payload) == len(high_payload)
                ),
            },
            "conservative_input_token_ceiling": input_ceiling,
            "input_token_ceiling_basis": (
                "at most one token per canonical UTF-8 request byte, rounded "
                f"up to {_INPUT_TOKEN_QUANTUM}"
            ),
        },
    }


def build_provider_canary_approval_plan(
    project_root: str | Path = ROOT,
    *,
    source_tree_sha256: str,
    cohort_id: str,
    candidate_resume_preflight_receipt_path: str,
    candidate_resume_preflight_receipt_sha256: str,
) -> dict[str, Any]:
    """Return a source-bound, four-attempt plan without initializing OpenAI."""

    root = Path(project_root).resolve()
    if root != ROOT.resolve():
        raise ValueError(
            "provider prompt constructors are source-bound to this project checkout"
        )
    if tuple(_EXPECTED_HARNESS_CONFIGS) != tuple(CANARY_ORDER):
        raise RuntimeError("provider canary plan order differs from Modal canary order")
    harnesses = [_harness_plan(root, harness) for harness in CANARY_ORDER]
    maximum_requests = sum(item["maximum_attempts"] for item in harnesses)
    total_input = sum(
        item["first_opportunity"]["conservative_input_token_ceiling"]
        for item in harnesses
    )
    total_completion = sum(
        item["request_settings"]["max_completion_tokens"] for item in harnesses
    )
    if maximum_requests != 4 or len(harnesses) != 4:
        raise RuntimeError("provider canary approval plan is not exactly four requests")
    if (
        PROVIDER_REQUEST_TIMEOUT_SECONDS
        + PROVIDER_ATTEMPT_FINALIZATION_RESERVE_SECONDS
        != CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS
        or CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS >= FUNCTION_TIMEOUT_SECONDS
    ):
        raise RuntimeError("provider canary timeout reserves are inconsistent")
    source = build_image_source_manifest(root)
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=source_tree_sha256,
        image_source_sha256=source.manifest_sha256,
        cohort_id=cohort_id,
    )
    if compute_source_tree_sha256(root) != identity.source_tree_sha256:
        raise ValueError("provider approval plan uses a stale source tree")
    if re.fullmatch(r"[0-9a-f]{64}", candidate_resume_preflight_receipt_sha256) is None:
        raise ValueError("candidate preflight receipt digest must be a SHA-256")
    preflight_path = PurePosixPath(candidate_resume_preflight_receipt_path)
    expected_preflight_parent = (
        modal_live_cohort_root(identity)
        / "components"
        / "candidate_resume_preflight_receipts"
        / "v2.0"
    )
    if (
        preflight_path.is_absolute()
        or preflight_path.parent != expected_preflight_parent
        or re.fullmatch(r"[0-9a-f]{64}\.json", preflight_path.name) is None
    ):
        raise ValueError(
            "candidate preflight receipt path is outside the selected cohort"
        )
    plan: dict[str, Any] = {
        "schema_name": "ProviderCanaryApprovalPlan",
        "schema_version": "1.2",
        "approval_plan_sha256_scope": _APPROVAL_PLAN_SHA256_SCOPE,
        "source_tree_sha256": identity.source_tree_sha256,
        "image_source_sha256": source.manifest_sha256,
        "cohort_id": identity.cohort_id,
        "dependency_lock_sha256": source.dependency_lock_sha256,
        "candidate_resume_preflight_receipt": {
            "path": preflight_path.as_posix(),
            "sha256": candidate_resume_preflight_receipt_sha256,
        },
        "provider": {
            "identity": "openai_official",
            "api_mode": API_MODE,
            "api_endpoint": OFFICIAL_OPENAI_API_BASE,
            "model": TARGET_MODEL,
        },
        "execution_deadlines": {
            "function_timeout_seconds": FUNCTION_TIMEOUT_SECONDS,
            "controller_subprocess_timeout_seconds": (
                CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS
            ),
            "provider_request_timeout_seconds": PROVIDER_REQUEST_TIMEOUT_SECONDS,
            "provider_attempt_finalization_reserve_seconds": (
                PROVIDER_ATTEMPT_FINALIZATION_RESERVE_SECONDS
            ),
        },
        "harness_order": list(CANARY_ORDER),
        "harnesses": harnesses,
        "totals": {
            "harness_count": 4,
            "maximum_requests": maximum_requests,
            "conservative_input_token_ceiling": total_input,
            "requested_completion_token_ceiling": total_completion,
        },
        "provider_calls_started": 0,
        "modal_calls_started": 0,
        "openai_clients_initialized": 0,
        "claim_scope": "cost_free_pre_request_approval_only",
    }
    plan["approval_plan_sha256"] = canonical_sha256(plan)
    return plan


def verify_provider_canary_approval_plan(plan: dict[str, Any]) -> str:
    if not isinstance(plan, dict):
        raise TypeError("provider canary approval plan must be an object")
    digest = plan.get("approval_plan_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("provider canary approval plan lacks its SHA-256")
    unsigned = dict(plan)
    del unsigned["approval_plan_sha256"]
    if unsigned.get("approval_plan_sha256_scope") != _APPROVAL_PLAN_SHA256_SCOPE:
        raise ValueError("provider canary approval plan hash scope changed")
    reconstructed = canonical_sha256(unsigned)
    if reconstructed != digest:
        raise ValueError("provider canary approval plan SHA-256 does not reconstruct")
    return digest


def create_provider_canary_approval_plan(
    output: str | Path,
    *,
    project_root: str | Path = ROOT,
    source_tree_sha256: str,
    cohort_id: str,
    candidate_resume_preflight_receipt_path: str,
    candidate_resume_preflight_receipt_sha256: str,
) -> dict[str, Any]:
    """Create and reopen one immutable provider approval plan."""

    destination = Path(output).expanduser()
    if destination.is_symlink():
        raise ValueError("provider canary approval output may not be a symlink")
    plan = build_provider_canary_approval_plan(
        project_root,
        source_tree_sha256=source_tree_sha256,
        cohort_id=cohort_id,
        candidate_resume_preflight_receipt_path=(
            candidate_resume_preflight_receipt_path
        ),
        candidate_resume_preflight_receipt_sha256=(
            candidate_resume_preflight_receipt_sha256
        ),
    )
    verify_provider_canary_approval_plan(plan)
    create_json_exclusive(destination, plan)
    # create_json_exclusive reopens the same regular-file descriptor, verifies
    # its exact bytes and parent identity, and rejects symlink substitution.
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the cost-free four-harness provider canary approval plan."
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="create the approval plan exactly once, then reopen and verify it",
    )
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument(
        "--candidate-resume-preflight-receipt-path",
        required=True,
    )
    parser.add_argument(
        "--candidate-resume-preflight-receipt-sha256",
        required=True,
    )
    arguments = parser.parse_args()
    plan = (
        create_provider_canary_approval_plan(
            arguments.output,
            project_root=arguments.project_root,
            source_tree_sha256=arguments.source_tree_sha256,
            cohort_id=arguments.cohort_id,
            candidate_resume_preflight_receipt_path=(
                arguments.candidate_resume_preflight_receipt_path
            ),
            candidate_resume_preflight_receipt_sha256=(
                arguments.candidate_resume_preflight_receipt_sha256
            ),
        )
        if arguments.output is not None
        else build_provider_canary_approval_plan(
            arguments.project_root,
            source_tree_sha256=arguments.source_tree_sha256,
            cohort_id=arguments.cohort_id,
            candidate_resume_preflight_receipt_path=(
                arguments.candidate_resume_preflight_receipt_path
            ),
            candidate_resume_preflight_receipt_sha256=(
                arguments.candidate_resume_preflight_receipt_sha256
            ),
        )
    )
    print(
        json.dumps(
            plan,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
