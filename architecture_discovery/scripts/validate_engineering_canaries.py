"""Provider-free static validation for four controller surface contracts.

This command deliberately separates three claims that are easy to conflate:

* controller-surface validation statically checks entrypoints, configuration,
  prompts, and a complete declarative Architecture IR response fixture;
* optional CUDA v2 or historical MPS v1 smoke-artifact validation checks
  internal consistency of an already completed trusted ten-step run without
  proving where it executed;
* optional downloaded Modal-canary validation checks one local, hash-bound
  four-harness bundle without contacting Modal or a model provider;
* scientific pilot readiness is never inferred by this command.

The deterministic fake response changes only non-executable graph metadata.  It
is size-bounded and checked through the same trusted IR validator used at the
candidate boundary.  No candidate model is constructed by the surface check,
no controller entrypoint is imported or executed, no provider SDK is
constructed, and no network request is made by this validator.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import stat
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_ir import (  # noqa: E402
    load_and_build_ir_candidate,
    validate_ir_candidate_json,
)
from architecture_ir.codec import MAX_IR_JSON_BYTES  # noqa: E402
from common.candidate_artifact import build_candidate_artifact  # noqa: E402
from common.descriptor_schema import (  # noqa: E402
    CATEGORY_CODES,
    SEMANTIC_METRIC_NAMES,
)
from common.gpt56_sol import (  # noqa: E402
    API_MODE,
    OFFICIAL_OPENAI_API_BASE,
    TARGET_MODEL,
)
from common.provider_attempts import (  # noqa: E402
    PROVIDER_ATTEMPT_LEDGER_FILENAME,
    PROVIDER_ATTEMPT_SCHEMA,
    ProviderAttemptRecord,
    generation_settings_sha256,
    load_provider_attempt_ledger,
    provider_attempt_totals,
)
from common.runtime_context import ExecutionContextV1  # noqa: E402
from common.task_adapter import DEFAULT_TASK  # noqa: E402
from common.trainer import (  # noqa: E402
    ResumeMismatchError,
    _dependency_lock_hash,
    _validate_resume,
    trusted_component_hashes,
    trusted_component_set_sha256,
)
from common.training_config import (  # noqa: E402
    SMOKE_TRAIN_CUDA_V2,
    SMOKE_TRAIN_V1,
    TrainingProfile,
    TrainingResult,
    TrainingSeedBundle,
    get_training_profile,
)
from containment.audit import CapabilityAudit  # noqa: E402
from modal_boundary import (  # noqa: E402
    APP_NAME,
    CANARY_ORDER,
    MAX_ARTIFACT_DOWNLOAD_FILE_BYTES,
    MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES,
    ArtifactIntegrityError,
    ModalLiveCohortIdentity,
    canonical_sha256,
    load_artifact_manifest,
    modal_live_cohort_root,
    validate_run_id,
    verify_artifact_manifest,
    volume_artifact_uri,
)
from study.serialization import create_json_exclusive  # noqa: E402

TRUSTED_CANDIDATE_RELATIVE_PATH = Path("common/initial_candidate.ir.json")
TRUSTED_LEGACY_PYTHON_CANDIDATE_RELATIVE_PATH = Path("common/initial_candidate.py")
# Deliberately stricter than the interpreter's own hard ceiling.  A fake
# provider response is only a small canary fixture, not a large search result.
MAX_FAKE_RESPONSE_BYTES = min(128 * 1024, MAX_IR_JSON_BYTES)
REQUIRED_CANARY_CLI_FLAGS = (
    "--engineering-pilot",
    "--iterations",
    "--seed",
    "--output-dir",
    "--training-profile",
    "--evaluation-profile",
    "--evaluation-cases",
    "--device",
)

_CONTROLLER_TRAINING_CONTRACTS = {
    ("full_train_v1", "1", "mps"): "historical_v1_mps",
    ("full_train_cuda_v2", "2", "cuda"): "active_v2_cuda",
}

_MODAL_CANARY_SUFFIXES = {
    "greedy_autoresearch": "greedy-ar",
    "semantic_autoresearch": "semantic-ar",
    "openevolve_generic": "openevolve-generic",
    "openevolve_semantic": "openevolve-semantic",
}
_MODAL_CANARY_SELECTOR_SCHEMA_NAME = "ModalProviderCanaryRunSelector"
_MODAL_CANARY_SELECTOR_SCHEMA_VERSION = "2.0"
_MODAL_CANARY_DOWNLOAD_ROOT = PurePosixPath(
    "outputs/development/modal_downloads"
)
_MODAL_CANARY_SELECTOR_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
        "selector_id",
        "harness_order",
        "runs",
    }
)
_MODAL_CANARY_SELECTOR_RUN_FIELDS = frozenset(
    {
        "harness",
        "run_id",
        "download_path",
        "artifact_manifest_path",
        "raw_artifact_manifest_sha256",
        "execution_context_path",
        "execution_context_sha256",
        "image_source_sha256",
        "modal_image_id",
    }
)
_NATIVE_CANARY_HARNESSES = frozenset({"greedy_autoresearch", "semantic_autoresearch"})
_MODAL_CANARY_GENERATOR_CONTRACT: dict[str, object] = {
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
}
_MODAL_CANARY_GENERATION_SETTINGS_SHA256 = generation_settings_sha256(
    {
        "model": TARGET_MODEL,
        "reasoning_effort": "high",
        "max_completion_tokens": 16_384,
        "seed": 1,
    }
)
_CREDENTIAL_FIELD_NAMES = frozenset(
    {
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "modal_token_id",
        "modal_token_secret",
    }
)

_CUDA_CANDIDATE_FILE_POLICY = {
    "best_checkpoint.pt": ("checkpoint", 100_000_000),
    "candidate_graph.json": ("json", 16_000_000),
    "latest_resume_checkpoint.pt": ("checkpoint", 100_000_000),
    "partial_resume_checkpoint.pt": ("checkpoint", 100_000_000),
    "runtime_validity.json": ("json", 2_000_000),
    "training_events.jsonl": ("jsonl", 10_000_000),
    "training_manifest.json": ("json", 2_000_000),
    "training_summary.json": ("json", 2_000_000),
}
_TRAINING_MANIFEST_V2_FIELDS = frozenset(
    {
        "created_at",
        "candidate_path",
        "candidate_source_hash",
        "candidate_artifact_hash",
        "candidate_format",
        "candidate_graph_hash",
        "immutable_candidate_relative_path",
        "candidate_initialization",
        "profile",
        "profile_hash",
        "seed_bundle",
        "seed_bundle_hash",
        "task_adapter_version",
        "task_adapter_hash",
        "requested_device",
        "selected_device",
        "allow_cpu_for_tests",
        "hardware_matched_scientific_run",
        "runtime",
        "dependency_lock_hash",
        "trusted_executable_component_hashes",
        "trusted_component_set_sha256",
        "controller_source_hash",
        "parameter_count_role",
        "development_only_checkpoint_selection",
        "scientific_limitations",
        "containment_audit",
        "containment_decision",
        "isolation_level",
        "reproducibility_note",
        "execution_context",
        "schema_name",
        "schema_version",
    }
)
_TRAINING_RUNTIME_V2_FIELDS = frozenset(
    {
        "platform",
        "machine",
        "processor",
        "python",
        "torch",
        "mps_built",
        "mps_available",
        "cuda_runtime",
        "cuda_available",
        "cuda_device_count",
        "deterministic_algorithms",
        "pytorch_enable_mps_fallback",
        "accelerator_memory_fraction",
        "cublas_workspace_config",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "accelerator_fingerprint",
        "declared_machine",
    }
)
_TRAINING_EVENT_V2_FIELDS = frozenset(
    {
        "timestamp",
        "optimizer_step",
        "examples_processed",
        "loss",
        "learning_rate",
        "gradient_norm",
        "validation_loss",
        "validation_exact_match_accuracy",
        "elapsed_seconds",
        "current_accelerator_allocated_bytes",
        "reserved_accelerator_allocated_bytes",
        "peak_accelerator_allocated_bytes",
        "accelerator_total_memory_bytes",
        "checkpoint_decision",
    }
)
_BEST_CHECKPOINT_V2_FIELDS = frozenset(
    {
        "checkpoint_kind",
        "model_state",
        "global_step",
        "examples_processed",
        "best_development_exact_match_accuracy",
        "best_development_loss",
        "candidate_source_hash",
        "profile_hash",
        "task_adapter_version",
        "task_adapter_hash",
        "seed_bundle",
        "seed_bundle_hash",
        "trusted_component_set_sha256",
        "dependency_lock_hash",
    }
)
_RESUME_CHECKPOINT_V2_FIELDS = frozenset(
    {
        "checkpoint_kind",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "global_step",
        "next_data_step",
        "examples_processed",
        "elapsed_seconds",
        "best_development_step",
        "best_development_exact_match_accuracy",
        "best_development_loss",
        "final_training_loss",
        "candidate_source_hash",
        "profile_hash",
        "task_adapter_version",
        "task_adapter_hash",
        "seed_bundle",
        "seed_bundle_hash",
        "trusted_component_set_sha256",
        "rng_state",
        "dependency_lock_hash",
    }
)
_HIGH_CONFIDENCE_CREDENTIAL = re.compile(
    r"(?:\ABearer\s+\S+|\Ask-[A-Za-z0-9_-]{12,}\Z)", re.IGNORECASE
)

_PRIVATE_CANARY_TOP_LEVEL_ROSTERS = {
    "greedy_autoresearch": frozenset(
        {
            "accepted_lineage",
            "architecture_hash_registry",
            "artifacts",
            "candidate_training",
            "incumbent.ir.json",
            "lineage.jsonl",
            "prompt_snapshot",
            PROVIDER_ATTEMPT_LEDGER_FILENAME,
            "run_manifest.json",
            "run_summary.json",
        }
    ),
    "semantic_autoresearch": frozenset(
        {
            "architecture_hash_registry",
            "artifacts",
            "candidate_training",
            "lineage.jsonl",
            "prompt_snapshot",
            PROVIDER_ATTEMPT_LEDGER_FILENAME,
            "run_manifest.json",
            "run_summary.json",
            "semantic_archive.json",
        }
    ),
    "openevolve_generic": frozenset(
        {
            "architecture_hash_registry",
            "best",
            "candidate_training",
            "checkpoints",
            "database",
            "evolution_trace.jsonl",
            "logs",
            "proposal_terminal_outcomes.jsonl",
            PROVIDER_ATTEMPT_LEDGER_FILENAME,
            "run_manifest.json",
            "run_result.json",
        }
    ),
    "openevolve_semantic": frozenset(
        {
            "architecture_hash_registry",
            "best",
            "candidate_training",
            "checkpoints",
            "database",
            "evolution_trace.jsonl",
            "logs",
            "proposal_terminal_outcomes.jsonl",
            PROVIDER_ATTEMPT_LEDGER_FILENAME,
            "run_manifest.json",
            "run_result.json",
        }
    ),
}
_PRIVATE_CANARY_TOP_LEVEL_FILES = {
    "greedy_autoresearch": frozenset(
        {
            "incumbent.ir.json",
            "lineage.jsonl",
            PROVIDER_ATTEMPT_LEDGER_FILENAME,
            "run_manifest.json",
            "run_summary.json",
        }
    ),
    "semantic_autoresearch": frozenset(
        {
            "lineage.jsonl",
            PROVIDER_ATTEMPT_LEDGER_FILENAME,
            "run_manifest.json",
            "run_summary.json",
            "semantic_archive.json",
        }
    ),
    "openevolve_generic": frozenset(
        {
            "evolution_trace.jsonl",
            "proposal_terminal_outcomes.jsonl",
            PROVIDER_ATTEMPT_LEDGER_FILENAME,
            "run_manifest.json",
            "run_result.json",
        }
    ),
    "openevolve_semantic": frozenset(
        {
            "evolution_trace.jsonl",
            "proposal_terminal_outcomes.jsonl",
            PROVIDER_ATTEMPT_LEDGER_FILENAME,
            "run_manifest.json",
            "run_result.json",
        }
    ),
}
_GREEDY_CONTROLLER_MANIFEST_FIELDS = frozenset(
    {
        "run_id",
        "condition",
        "seed",
        "candidate_budget",
        "mutation_budget",
        "maximum_provider_attempts",
        "candidate_training_budget",
        "initial_candidate_is_evaluated",
        "candidate_format",
        "max_ir_bytes",
        "run_mode",
        "exploratory_only",
        "selection_semantics",
        "greedy_retention",
        "generator",
        "initial_candidate_hash",
        "initial_architecture_hash",
        "architecture_hash_schema",
        "architecture_deduplication",
        "evaluator_hash",
        "trusted_executable_component_hashes",
        "trusted_component_set_sha256",
        "config_hash",
        "prompt_protocol",
        "training",
        "evaluation",
        "preflight",
        "evidence_scope",
        "authoritative_scientific_evidence",
        "provider_attempt_ledger",
        "provider_attempt_schema",
        "schema_name",
        "schema_version",
    }
)
_SEMANTIC_CONTROLLER_MANIFEST_FIELDS = frozenset(
    (_GREEDY_CONTROLLER_MANIFEST_FIELDS - {"greedy_retention"})
    | {"semantic_archive"}
)
_OPENEVOLVE_CONTROLLER_MANIFEST_FIELDS = frozenset(
    {
        "run_id",
        "condition",
        "seed",
        "candidate_budget",
        "mutation_budget",
        "proposal_opportunities",
        "maximum_provider_attempts",
        "provider_attempt_ledger",
        "provider_attempt_schema",
        "candidate_training_budget",
        "initial_program_is_evaluated",
        "engineering_pilot",
        "candidate_format",
        "proposal_format",
        "generated_python_execution",
        "containment_bypass",
        "generator",
        "initial_candidate_hash",
        "initial_architecture_hash",
        "architecture_hash_schema",
        "parent_relative_architecture_change_required",
        "architecture_deduplication",
        "proposal_terminal_ledger",
        "evaluator_hash",
        "trusted_executable_component_hashes",
        "trusted_component_set_sha256",
        "config_hash",
        "evidence_scope",
        "authoritative_scientific_evidence",
        "eligibility_threshold",
        "limitations",
        "training",
        "evaluation",
        "schema_name",
        "schema_version",
    }
)
_CONTROLLER_MANIFEST_FIELDS = {
    "greedy_autoresearch": _GREEDY_CONTROLLER_MANIFEST_FIELDS,
    "semantic_autoresearch": _SEMANTIC_CONTROLLER_MANIFEST_FIELDS,
    "openevolve_generic": _OPENEVOLVE_CONTROLLER_MANIFEST_FIELDS,
    "openevolve_semantic": _OPENEVOLVE_CONTROLLER_MANIFEST_FIELDS,
}
_CONTROLLER_SUMMARY_FIELDS = {
    "greedy_autoresearch": frozenset(
        {
            "run_id",
            "condition",
            "proposal_opportunities_requested",
            "proposal_opportunities_terminal",
            "lineage_path",
            "incumbent_path",
            "authoritative_scientific_evidence",
            "schema_name",
            "schema_version",
        }
    ),
    "semantic_autoresearch": frozenset(
        {
            "run_id",
            "condition",
            "proposal_opportunities_requested",
            "proposal_opportunities_terminal",
            "semantic_archive_cells",
            "lineage_path",
            "archive_path",
            "scientific_novelty_claim",
            "schema_name",
            "schema_version",
        }
    ),
    "openevolve_generic": frozenset(
        {
            "run_id",
            "condition",
            "completed",
            "eligible_best_program_found",
            "best_program_id",
            "proposal_opportunities_requested",
            "proposal_opportunities_completed",
            "proposal_terminal_iterations",
            "proposal_terminal_status_counts",
            "proposal_accounting_errors",
            "engineering_pilot",
            "authoritative_scientific_evidence",
            "failure_stage",
            "schema_name",
            "schema_version",
        }
    ),
}
_CONTROLLER_SUMMARY_FIELDS["openevolve_semantic"] = _CONTROLLER_SUMMARY_FIELDS[
    "openevolve_generic"
]
_CANDIDATE_RECORD_FIELDS = frozenset(
    {
        "run_id",
        "condition",
        "seed",
        "candidate_id",
        "parent_id",
        "lineage_record_id",
        "proposal_id",
        "parent_lineage_record_id",
        "inspiration_ids",
        "proposal_text",
        "mechanism_hypothesis",
        "prompt_hash",
        "response_hash",
        "code_hash",
        "diff",
        "proposal_timestamp",
        "completion_timestamp",
        "retention_decision",
        "archive_cells",
        "rollback_target",
        "future_parent_count",
        "input_tokens",
        "output_tokens",
    }
)
_CONTROLLER_SEARCH_LINEAGE_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "record_id",
        "evaluation_run_id",
        "condition_id",
        "evaluation_candidate_id",
        "execution_ok",
        "transformer_valid",
        "public_accuracy",
        "search_score",
        "eligible_for_parent",
        "failure_stage",
        "infrastructure_failure",
        "online_descriptor_codes",
    }
)
_NATIVE_LINEAGE_FIELDS = {
    "greedy_autoresearch": frozenset(
        _CANDIDATE_RECORD_FIELDS
        | _CONTROLLER_SEARCH_LINEAGE_FIELDS
        | {"proposal_opportunity", "candidate_role"}
    ),
    "semantic_autoresearch": frozenset(
        _CANDIDATE_RECORD_FIELDS
        | _CONTROLLER_SEARCH_LINEAGE_FIELDS
        | {"opportunity_index"}
    ),
}
_OPENEVOLVE_TRACE_FIELDS = frozenset(
    {
        "iteration",
        "timestamp",
        "parent_id",
        "child_id",
        "parent_metrics",
        "child_metrics",
        "parent_code",
        "child_code",
        "parent_changes_description",
        "prompt",
        "llm_response",
        "improvement_delta",
        "island_id",
        "generation",
        "artifacts",
        "metadata",
    }
)
_OPENEVOLVE_TRACE_METADATA_FIELDS = frozenset({"iteration_time", "changes"})
_OPENEVOLVE_TRACE_PROMPT_FIELDS = frozenset({"system", "user"})
_OPENEVOLVE_TRACE_ARTIFACT_FIELDS = frozenset(
    {
        "candidate_architecture_hash",
        "candidate_graph_hash",
        "failure_stage",
        "infrastructure_failure",
        "layer_a_record_id",
        "parent_architecture_hash",
    }
)
_OPENEVOLVE_CHECKPOINT_METADATA_FIELDS = frozenset(
    {
        "island_feature_maps",
        "islands",
        "archive",
        "best_program_id",
        "island_best_programs",
        "last_iteration",
        "current_island",
        "island_generations",
        "last_migration_generation",
        "feature_stats",
    }
)
_DOWNLOADED_CANARY_OUTER_ROSTER = frozenset(
    {
        "artifact_manifest.json",
        "controller",
        "execution_context.json",
        "image_source_manifest.json",
        "remote_action_result.json",
    }
)
_PRIVATE_CANARY_TEXT_SUFFIXES = frozenset(
    {".json", ".jsonl", ".log", ".md", ".sample", ".txt"}
)
_OPENEVOLVE_PROGRAM_FIELDS = frozenset(
    {
        "id",
        "code",
        "changes_description",
        "language",
        "parent_id",
        "generation",
        "timestamp",
        "iteration_found",
        "metrics",
        "complexity",
        "diversity",
        "metadata",
        "prompts",
        "artifacts_json",
        "artifact_dir",
        "embedding",
    }
)


@dataclass(frozen=True)
class HarnessSpec:
    harness_id: str
    display_name: str
    agent_directory: str
    parent_policy: str
    proposal_policy: str
    prompt_candidates: tuple[str, ...]
    config_condition: str | None = None
    delegated_controller_kind: str | None = None


HARNESSES = (
    HarnessSpec(
        harness_id="normal_autoresearch",
        display_name="Normal Autoresearch",
        agent_directory="greedy_autoresearch",
        parent_policy="single",
        proposal_policy="ordinary",
        prompt_candidates=("program.md", "system_prompt.md"),
        config_condition="greedy_autoresearch",
    ),
    HarnessSpec(
        harness_id="semantic_autoresearch",
        display_name="Semantic Autoresearch",
        agent_directory="semantic_autoresearch",
        parent_policy="single",
        proposal_policy="semantic_transition",
        prompt_candidates=("program.md", "system_prompt.md"),
        config_condition="semantic_autoresearch",
    ),
    HarnessSpec(
        harness_id="openevolve",
        display_name="OpenEvolve",
        agent_directory="openevolve_generic",
        parent_policy="portfolio",
        proposal_policy="ordinary",
        prompt_candidates=("system_prompt.md", "program.md"),
        delegated_controller_kind="generic",
    ),
    HarnessSpec(
        harness_id="semantic_openevolve",
        display_name="Semantic OpenEvolve",
        agent_directory="openevolve_semantic",
        parent_policy="portfolio",
        proposal_policy="semantic_archive",
        prompt_candidates=("system_prompt.md", "program.md"),
        delegated_controller_kind="semantic",
    ),
)


class DeterministicFakeProvider:
    """A local response fixture; it has no client, endpoint, or network method."""

    def __init__(self, response: str) -> None:
        if not isinstance(response, str):
            raise TypeError("fake-provider response must be text")
        if len(response.encode("utf-8")) > MAX_FAKE_RESPONSE_BYTES:
            raise ValueError("fake-provider response exceeds the canary byte limit")
        validation = validate_ir_candidate_json(response)
        if not validation.valid:
            raise ValueError(
                "fake-provider response is not valid Architecture IR: "
                + "; ".join(issue.message for issue in validation.issues)
            )
        self._response = response
        self.calls = 0

    def complete(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("fake-provider prompt cannot be empty")
        self.calls += 1
        return self._response


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_ir_response(trusted_ir: str, spec: HarnessSpec) -> str:
    """Return one complete valid IR document with a metadata-only mutation."""

    validation = validate_ir_candidate_json(trusted_ir)
    if not validation.valid or validation.graph is None:
        raise ValueError(
            "trusted seed is not valid Architecture IR: "
            + "; ".join(issue.message for issue in validation.issues)
        )
    payload = validation.graph.to_dict()
    payload["metadata"]["engineering_canary_fixture"] = spec.harness_id
    response = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if len(response.encode("utf-8")) > MAX_FAKE_RESPONSE_BYTES:
        raise ValueError("fixed Architecture IR response exceeds the canary byte limit")
    child_validation = validate_ir_candidate_json(response)
    if not child_validation.valid:
        raise ValueError(
            "fixed Architecture IR response failed trusted validation: "
            + "; ".join(issue.message for issue in child_validation.issues)
        )
    return response


def _graph_structure(graph: Any) -> dict[str, Any]:
    """Return executable graph fields, excluding non-executable metadata."""

    payload = graph.to_dict()
    payload.pop("metadata", None)
    return payload


def _fake_prompt(spec: HarnessSpec, source_hash: str) -> str:
    return "\n".join(
        (
            "Engineering static-surface fixture. Do not execute candidates.",
            f"harness={spec.harness_id}",
            f"parent_policy={spec.parent_policy}",
            f"proposal_policy={spec.proposal_policy}",
            f"trusted_parent_sha256={source_hash}",
            "Return exactly one complete Architecture IR JSON document.",
        )
    )


def _declared_cli_flags(tree: ast.AST) -> set[str]:
    """Collect literal long options without importing or executing a module."""

    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
            continue
        for argument in node.args:
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            ):
                flags.add(argument.value)
    return flags


def _delegates_to_controller(tree: ast.AST, expected_kind: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Name) or function.id != "run_controller":
            continue
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == expected_kind
        ):
            return True
    return False


def _controller_training_contract(training: dict[str, Any]) -> str:
    """Validate an active CUDA or historical MPS controller binding."""

    profile_name = training.get("profile")
    profile_version = training.get("profile_version")
    device = training.get("device")
    if not isinstance(profile_name, str):
        raise ValueError("configuration lacks a named training profile")
    if not isinstance(profile_version, (str, int)) or isinstance(profile_version, bool):
        raise ValueError("configuration lacks a training profile version")
    if not isinstance(device, str):
        raise ValueError("configuration lacks a named training device")
    binding = (profile_name, str(profile_version), device)
    try:
        contract = _CONTROLLER_TRAINING_CONTRACTS[binding]
    except KeyError as error:
        raise ValueError(
            "configuration training binding is neither active CUDA v2 nor "
            f"historical MPS v1: {binding!r}"
        ) from error
    profile = get_training_profile(profile_name)
    if profile.version != str(profile_version):
        raise ValueError("configured profile version differs from the frozen profile")
    if profile.device_requirement != device:
        raise ValueError(
            "configured device differs from the frozen profile requirement"
        )
    if training.get("allow_cpu_for_tests") is not False:
        raise ValueError("configured controller does not disable CPU training")
    return contract


def _validate_harness(
    project_root: Path,
    spec: HarnessSpec,
    *,
    trusted_ir: str,
    trusted_artifact_hash: str,
) -> dict[str, Any]:
    agent_root = project_root / "agents" / spec.agent_directory
    entrypoint = agent_root / "run.py"
    config_path = agent_root / "config.yaml"
    errors: list[str] = []
    static_cli_contract_ok = False
    training_contract: str | None = None

    if not entrypoint.is_file():
        errors.append(f"missing entrypoint: {entrypoint.relative_to(project_root)}")
    else:
        try:
            entrypoint_tree = ast.parse(entrypoint.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            errors.append(
                f"entrypoint is not valid Python: {type(error).__name__}: {error}"
            )
            entrypoint_tree = None
        if entrypoint_tree is not None:
            cli_flags = _declared_cli_flags(entrypoint_tree)
            delegation_ok = True
            if spec.delegated_controller_kind is not None:
                delegation_ok = _delegates_to_controller(
                    entrypoint_tree,
                    spec.delegated_controller_kind,
                )
                if not delegation_ok:
                    errors.append(
                        "entrypoint does not statically delegate to "
                        f"run_controller({spec.delegated_controller_kind!r})"
                    )
                shared_runner = project_root / "common" / "openevolve_runner.py"
                if not shared_runner.is_file():
                    errors.append("missing delegated common/openevolve_runner.py")
                else:
                    try:
                        shared_tree = ast.parse(
                            shared_runner.read_text(encoding="utf-8")
                        )
                    except (OSError, SyntaxError) as error:
                        errors.append(
                            "delegated runner is not valid Python: "
                            f"{type(error).__name__}: {error}"
                        )
                    else:
                        cli_flags.update(_declared_cli_flags(shared_tree))
            missing_flags = [
                flag for flag in REQUIRED_CANARY_CLI_FLAGS if flag not in cli_flags
            ]
            if missing_flags:
                errors.append(
                    "static CLI contract lacks required flags: "
                    + ", ".join(missing_flags)
                )
            static_cli_contract_ok = delegation_ok and not missing_flags

    config: dict[str, Any] | None = None
    if not config_path.is_file():
        errors.append(f"missing configuration: {config_path.relative_to(project_root)}")
    else:
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise TypeError("top-level YAML value must be a mapping")
            config = loaded
        except (OSError, TypeError, yaml.YAMLError) as error:
            errors.append(f"configuration is invalid: {type(error).__name__}: {error}")
    if config is not None:
        training = config.get("training")
        if not isinstance(training, dict):
            errors.append("configuration lacks a training mapping")
        else:
            try:
                training_contract = _controller_training_contract(training)
            except ValueError as error:
                errors.append(str(error))
        if (
            spec.config_condition is not None
            and config.get("condition") != spec.config_condition
        ):
            errors.append(
                "configured condition differs from the named harness: "
                f"{config.get('condition')!r} != {spec.config_condition!r}"
            )

    prompt_path = next(
        (
            agent_root / name
            for name in spec.prompt_candidates
            if (agent_root / name).is_file()
        ),
        None,
    )
    if prompt_path is None:
        choices = ", ".join(spec.prompt_candidates)
        errors.append(f"missing controller prompt (expected one of: {choices})")
    elif not prompt_path.read_text(encoding="utf-8").strip():
        errors.append("controller prompt is empty")

    fixed_response = _fixed_ir_response(trusted_ir, spec)
    fake = DeterministicFakeProvider(fixed_response)
    trusted_validation = validate_ir_candidate_json(trusted_ir)
    if not trusted_validation.valid or trusted_validation.graph is None:
        raise ValueError("trusted Architecture IR became invalid during validation")
    surface_ready = (
        entrypoint.is_file()
        and static_cli_contract_ok
        and config is not None
        and prompt_path is not None
        and not errors
    )
    if surface_ready:
        prompt = _fake_prompt(spec, trusted_artifact_hash)
        response = fake.complete(prompt)
        response_bytes = len(response.encode("utf-8"))
        if response_bytes > MAX_FAKE_RESPONSE_BYTES:
            errors.append(
                "fake Architecture IR response exceeded the canary byte limit"
            )
        child_validation = validate_ir_candidate_json(response)
        if not child_validation.valid or child_validation.graph is None:
            errors.append(
                "fixed response failed trusted Architecture IR validation: "
                + "; ".join(issue.message for issue in child_validation.issues)
            )
            executable_structure_unchanged = False
            graph_hash_changed = False
        else:
            executable_structure_unchanged = _graph_structure(
                child_validation.graph
            ) == _graph_structure(trusted_validation.graph)
            graph_hash_changed = (
                child_validation.graph_hash != trusted_validation.graph_hash
            )
        if not executable_structure_unchanged:
            errors.append("pre-reviewed mutation changed executable graph structure")
        if not graph_hash_changed:
            errors.append("pre-reviewed mutation did not change the graph document")
    else:
        response_bytes = 0
        child_validation = trusted_validation
        executable_structure_unchanged = True
        graph_hash_changed = False

    return {
        "harness_id": spec.harness_id,
        "display_name": spec.display_name,
        "agent_directory": spec.agent_directory,
        "expected_parent_policy": spec.parent_policy,
        "expected_proposal_policy": spec.proposal_policy,
        "entrypoint_present": entrypoint.is_file(),
        "entrypoint_executed": False,
        "static_cli_contract_passed": static_cli_contract_ok,
        "configuration_present": config_path.is_file(),
        "configuration_parsed": config is not None,
        "training_contract": training_contract,
        "prompt_present": prompt_path is not None,
        "local_fixture_calls": fake.calls,
        "real_provider_calls": 0,
        "fixed_response_format": "complete_architecture_ir_json",
        "fixed_response_bytes": response_bytes,
        "fixed_response_byte_limit": MAX_FAKE_RESPONSE_BYTES,
        "candidate_graph_hash_changed": graph_hash_changed,
        "candidate_executable_structure_unchanged": executable_structure_unchanged,
        "candidate_ir_valid": child_validation.valid,
        "candidate_graph_hash": child_validation.graph_hash,
        "candidate_executed": False,
        "training_started": False,
        "passed": not errors,
        "errors": errors,
    }


def validate_controller_surfaces(project_root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = root / TRUSTED_CANDIDATE_RELATIVE_PATH
    logical_candidate = TRUSTED_CANDIDATE_RELATIVE_PATH.as_posix()
    if not candidate.is_file():
        return {
            "passed": False,
            "trusted_candidate": logical_candidate,
            "real_provider_calls": 0,
            "local_fixture_calls": 0,
            "candidate_execution_runs": 0,
            "training_runs": 0,
            "entrypoint_execution_runs": 0,
            "harnesses": [],
            "errors": [f"trusted candidate is missing: {logical_candidate}"],
        }
    trusted_ir = candidate.read_text(encoding="utf-8")
    trusted_validation = validate_ir_candidate_json(trusted_ir)
    if not trusted_validation.valid:
        return {
            "passed": False,
            "trusted_candidate": logical_candidate,
            "real_provider_calls": 0,
            "local_fixture_calls": 0,
            "candidate_execution_runs": 0,
            "training_runs": 0,
            "entrypoint_execution_runs": 0,
            "harnesses": [],
            "errors": [
                "trusted candidate failed Architecture IR validation: "
                + "; ".join(issue.message for issue in trusted_validation.issues)
            ],
        }
    trusted_hash = _sha256_file(candidate)
    harness_reports = [
        _validate_harness(
            root,
            spec,
            trusted_ir=trusted_ir,
            trusted_artifact_hash=trusted_hash,
        )
        for spec in HARNESSES
    ]
    passed = len(harness_reports) == 4 and all(
        report["passed"] for report in harness_reports
    )
    return {
        "passed": passed,
        "trusted_candidate": logical_candidate,
        "trusted_candidate_sha256": trusted_hash,
        "trusted_candidate_graph_hash": trusted_validation.graph_hash,
        "candidate_format": "architecture_ir",
        "real_provider_calls": 0,
        "local_fixture_calls": sum(
            int(report["local_fixture_calls"]) for report in harness_reports
        ),
        "candidate_execution_runs": 0,
        "training_runs": 0,
        "entrypoint_execution_runs": 0,
        "harnesses": harness_reports,
        "errors": [error for report in harness_reports for error in report["errors"]],
    }


def _exact_bool(payload: dict[str, Any], field: str, expected: bool) -> None:
    value = payload.get(field)
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{field} must be exactly {expected}")


def _require_exact_fields(
    payload: dict[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    observed = frozenset(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{label} fields differ from the frozen canary schema "
            f"(missing={missing}, extra={extra})"
        )


def _safe_json_object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 2_000_000:
        raise ValueError(f"{path.name} exceeds the 2 MB evidence limit")
    payload = _strict_json_loads(
        path.read_text(encoding="utf-8"),
        label=path.name,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def _strict_json_loads(text: str, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value!r}")

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field, value in pairs:
            if field in payload:
                raise ValueError(f"{label} contains duplicate JSON field {field!r}")
            payload[field] = value
        return payload

    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=reject_constant,
    )


def _device_kind(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.strip().lower().split(":", maxsplit=1)[0]


def _nonnegative_integer(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _exact_integer(
    payload: dict[str, Any],
    field: str,
    *,
    minimum: int = 0,
) -> int:
    value = payload.get(field)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_number(
    payload: dict[str, Any],
    field: str,
    *,
    minimum: float | None = None,
) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} must be finite numeric data")
    return result


def _iso8601_timestamp(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return value


def _credential_value_paths(value: object, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for field, child in value.items():
            found.extend(
                _credential_value_paths(child, path=f"{path}.{field}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _credential_value_paths(child, path=f"{path}[{index}]")
            )
    elif isinstance(value, str) and _HIGH_CONFIDENCE_CREDENTIAL.search(value):
        found.append(path)
    return found


def _validate_json_security(payload: object, *, label: str) -> None:
    credential_fields = _credential_field_paths(payload)
    if credential_fields:
        raise ValueError(
            f"{label} contains credential-shaped fields: "
            + ", ".join(credential_fields[:5])
        )
    credential_values = _credential_value_paths(payload)
    if credential_values:
        raise ValueError(
            f"{label} contains credential-shaped values: "
            + ", ".join(credential_values[:5])
        )
    absolute_paths = _absolute_path_field_paths(payload)
    if absolute_paths:
        raise ValueError(
            f"{label} contains non-portable absolute path fields: "
            + ", ".join(absolute_paths[:5])
        )


def _validate_cuda_candidate_roster(output: Path) -> None:
    if output.is_symlink() or not output.is_dir():
        raise ValueError("CUDA candidate output must be a regular directory")
    entries = tuple(output.iterdir())
    observed = {entry.name for entry in entries}
    expected = set(_CUDA_CANDIDATE_FILE_POLICY)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "CUDA candidate artifact roster differs from the exact action policy "
            f"(missing={missing}, extra={extra})"
        )
    for entry in entries:
        kind, maximum_bytes = _CUDA_CANDIDATE_FILE_POLICY[entry.name]
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(
                f"CUDA candidate artifact must be a regular file: {entry.name}"
            )
        size = entry.stat().st_size
        if size < 1 or size > maximum_bytes:
            raise ValueError(
                f"CUDA candidate artifact size violates policy: {entry.name}"
            )
        if kind == "checkpoint" and entry.suffix != ".pt":
            raise ValueError(f"CUDA checkpoint has an invalid file type: {entry.name}")
        if kind == "json" and entry.suffix != ".json":
            raise ValueError(f"CUDA JSON artifact has an invalid type: {entry.name}")
        if kind == "jsonl" and entry.suffix != ".jsonl":
            raise ValueError(f"CUDA event artifact has an invalid type: {entry.name}")


def _validate_image_source_manifest(payload: dict[str, Any]) -> str:
    fields = {
        "schema_name",
        "schema_version",
        "recipe_version",
        "python_version",
        "uv_version",
        "modal_version",
        "dependency_lock_sha256",
        "files",
    }
    if set(payload) != fields:
        raise ValueError("image-source manifest fields differ from schema v1")
    if (
        payload["schema_name"] != "ModalImageSourceManifest"
        or payload["schema_version"] != "1.0"
    ):
        raise ValueError("image-source manifest schema identity is invalid")
    dependency = payload["dependency_lock_sha256"]
    if not _is_lower_sha256(dependency):
        raise ValueError("image-source dependency lock hash is invalid")
    files = payload["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("image-source manifest must bind at least one source file")
    previous = ""
    for index, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("image-source file record has an invalid exact schema")
        relative = record["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative <= previous
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or "\\" in relative
            or PureWindowsPath(relative).is_absolute()
        ):
            raise ValueError(f"image-source file {index} has a non-portable path")
        previous = relative
        if not _is_lower_sha256(record["sha256"]):
            raise ValueError(f"image-source file {index} hash is invalid")
        if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
            raise ValueError(f"image-source file {index} size is invalid")
    _validate_json_security(payload, label="image-source manifest")
    return dependency


def _validate_cuda_modal_binding(
    *,
    output: Path,
    manifest: dict[str, Any],
    dependency_lock_hash: str,
    require_modal_context: bool,
    expected_function: str,
    outer_context_name: str,
) -> None:
    context_payload = manifest.get("execution_context")
    if context_payload is None and not require_modal_context:
        return
    if not isinstance(context_payload, dict):
        raise ValueError("CUDA Modal smoke lacks an execution context")
    context = ExecutionContextV1.from_dict(context_payload)
    if not require_modal_context:
        return
    if context.execution_backend != "modal":
        raise ValueError("CUDA smoke execution context is not Modal")
    if context.function_name != expected_function:
        raise ValueError(
            f"CUDA smoke context is not bound to {expected_function}"
        )
    if any(
        value is None
        for value in (
            context.modal_app_id,
            context.modal_function_id,
            context.modal_call_id,
            context.modal_image_id,
        )
    ):
        raise ValueError("CUDA Modal smoke lacks complete Modal object IDs")
    if output.name != "seed_1" or output.parent.name != "candidate_smoke":
        raise ValueError("CUDA candidate output is not the frozen seed_1 action path")
    run_root = output.parent.parent
    if context.run_id != run_root.name:
        raise ValueError("CUDA training context run ID differs from its run directory")
    if context.artifact_uri != volume_artifact_uri(context.run_id):
        raise ValueError("CUDA training context artifact URI differs from its run ID")
    outer_context = _safe_json_object(run_root / outer_context_name)
    if outer_context != context_payload:
        raise ValueError("CUDA training context differs from the outer run context")
    image_payload = _safe_json_object(run_root / "image_source_manifest.json")
    image_dependency = _validate_image_source_manifest(image_payload)
    if image_dependency != dependency_lock_hash:
        raise ValueError("CUDA dependency lock differs from the image-source manifest")
    if canonical_sha256(image_payload) != context.image_source_sha256:
        raise ValueError("CUDA image-source digest differs from its execution context")


def _validate_seed_bundle(payload: object) -> TrainingSeedBundle:
    fields = {
        "model_initialization_seed",
        "training_data_seed",
        "development_set_seed",
        "dataloader_seed",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("training seed bundle has an invalid exact schema")
    for field in fields:
        if type(payload[field]) is not int or payload[field] < 0:
            raise ValueError(
                f"training seed bundle {field} must be a nonnegative integer"
            )
    return TrainingSeedBundle(**payload)


def _validate_cuda_manifest(
    manifest: dict[str, Any],
    *,
    output: Path,
    profile: TrainingProfile,
    candidate_hash: str,
    candidate_graph_hash: str,
    summary: dict[str, Any],
    require_modal_context: bool,
    expected_function: str = "candidate_smoke",
    outer_context_name: str = "execution_context.json",
) -> tuple[TrainingSeedBundle, str]:
    _validate_json_security(manifest, label="training manifest")
    expected_fields = set(_TRAINING_MANIFEST_V2_FIELDS)
    if not require_modal_context and "execution_context" not in manifest:
        expected_fields.remove("execution_context")
    if set(manifest) != expected_fields:
        raise ValueError("training manifest fields differ from the exact v2 schema")
    _iso8601_timestamp(manifest["created_at"], "training_manifest.created_at")
    expected_values = {
        "schema_name": "TrainingManifest",
        "schema_version": "2.0",
        "candidate_path": "candidate_graph.json",
        "candidate_source_hash": candidate_hash,
        "candidate_artifact_hash": candidate_hash,
        "candidate_format": "architecture_ir",
        "candidate_graph_hash": candidate_graph_hash,
        "immutable_candidate_relative_path": "candidate_graph.json",
        "candidate_initialization": "from_scratch",
        "profile": json.loads(json.dumps(profile.to_dict(), allow_nan=False)),
        "profile_hash": profile.profile_hash,
        "task_adapter_version": DEFAULT_TASK.version,
        "task_adapter_hash": DEFAULT_TASK.config_hash,
        "requested_device": "cuda",
        "selected_device": summary.get("device"),
        "parameter_count_role": "descriptive_metadata_only",
        "development_only_checkpoint_selection": profile.checkpoint_selection_rule,
        "scientific_limitations": [
            "Engineering only. Not valid for architecture ranking or "
            "scientific conclusions."
        ],
        "isolation_level": "engineering_only_or_scientific_gate_blocked",
    }
    for field, expected in expected_values.items():
        if manifest[field] != expected:
            raise ValueError(f"training manifest {field} differs from the run")
    _exact_bool(manifest, "allow_cpu_for_tests", False)
    _exact_bool(manifest, "hardware_matched_scientific_run", False)
    note = manifest["reproducibility_note"]
    if not isinstance(note, str) or not note.strip():
        raise ValueError("training manifest reproducibility note must be nonempty text")

    seeds = _validate_seed_bundle(manifest["seed_bundle"])
    if manifest["seed_bundle_hash"] != seeds.bundle_hash:
        raise ValueError("training manifest seed bundle hash is invalid")
    component_hashes = manifest["trusted_executable_component_hashes"]
    if (
        not isinstance(component_hashes, dict)
        or not component_hashes
        or any(
            not isinstance(name, str)
            or not name
            or not _is_lower_sha256(digest)
            for name, digest in component_hashes.items()
        )
    ):
        raise ValueError("training manifest trusted component map is invalid")
    if component_hashes != trusted_component_hashes():
        raise ValueError(
            "training manifest trusted component hashes differ from source"
        )
    component_set = trusted_component_set_sha256(component_hashes)
    if (
        manifest["trusted_component_set_sha256"] != component_set
        or manifest["controller_source_hash"] != component_set
    ):
        raise ValueError("training manifest trusted component-set binding is invalid")
    dependency_lock_hash = manifest["dependency_lock_hash"]
    if dependency_lock_hash != _dependency_lock_hash():
        raise ValueError("training manifest dependency lock differs from source")

    audit_payload = manifest["containment_audit"]
    if not isinstance(audit_payload, dict):
        raise ValueError("training manifest containment audit must be an object")
    audit = CapabilityAudit.from_dict(audit_payload)
    if audit.to_dict() != audit_payload:
        raise ValueError("training manifest containment audit is noncanonical")
    if audit.visible_credential_names:
        raise ValueError("candidate worker observed credential-like environment names")
    decision = manifest["containment_decision"]
    if not isinstance(decision, dict) or set(decision) != {
        "allowed",
        "scientific",
        "phase",
        "candidate_format",
        "blockers",
        "warnings",
        "audit_hash",
    }:
        raise ValueError("training manifest containment decision has invalid fields")
    _exact_bool(decision, "allowed", True)
    _exact_bool(decision, "scientific", False)
    if (
        decision["phase"] != "pre_execution"
        or decision["candidate_format"] != "architecture_ir"
        or decision["blockers"] != []
        or not isinstance(decision["warnings"], list)
        or not all(isinstance(item, str) and item for item in decision["warnings"])
        or decision["audit_hash"] != audit.audit_hash
    ):
        raise ValueError("training manifest containment decision is inconsistent")

    runtime = manifest["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != _TRAINING_RUNTIME_V2_FIELDS:
        raise ValueError("training runtime fields differ from the exact v2 schema")
    for field in ("platform", "machine", "processor", "python", "torch"):
        if not isinstance(runtime[field], str):
            raise ValueError(f"training runtime {field} must be text")
    for field, expected in {
        "mps_built": False,
        "mps_available": False,
        "cuda_available": True,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
    }.items():
        _exact_bool(runtime, field, expected)
    if _exact_integer(runtime, "cuda_device_count", minimum=1) != 1:
        raise ValueError("training runtime must expose exactly one CUDA device")
    if not isinstance(runtime["cuda_runtime"], str) or not runtime["cuda_runtime"]:
        raise ValueError("training runtime CUDA version is missing")
    if runtime["accelerator_memory_fraction"] != profile.accelerator_memory_fraction:
        raise ValueError("training runtime accelerator memory fraction differs")
    if runtime["cublas_workspace_config"] != profile.cublas_workspace_config:
        raise ValueError("training runtime CUBLAS workspace differs from the profile")
    if runtime["pytorch_enable_mps_fallback"] not in {"", "0"}:
        raise ValueError("training manifest requested MPS fallback")
    if not isinstance(runtime["declared_machine"], dict):
        raise ValueError("training runtime declared machine must be an object")
    fingerprint = _validate_cuda_fingerprint(
        summary.get("accelerator_fingerprint"),
        selected_device=str(summary.get("device")),
    )
    if runtime["accelerator_fingerprint"] != fingerprint:
        raise ValueError("CUDA runtime and summary fingerprints differ")
    _validate_cuda_modal_binding(
        output=output,
        manifest=manifest,
        dependency_lock_hash=dependency_lock_hash,
        require_modal_context=require_modal_context,
        expected_function=expected_function,
        outer_context_name=outer_context_name,
    )
    return seeds, dependency_lock_hash


def _validate_cuda_summary(
    summary: dict[str, Any],
    *,
    profile: TrainingProfile,
    candidate_hash: str,
    checkpoint_path: Path,
    event_path: Path,
) -> TrainingResult:
    _validate_json_security(summary, label="training summary")
    if set(summary) != set(TrainingResult.__dataclass_fields__):
        raise ValueError("training summary fields differ from the exact v2 schema")
    try:
        parsed = TrainingResult.from_dict(summary)
    except (TypeError, ValueError) as error:
        raise ValueError("training summary is not TrainingResult v2") from error
    if parsed.to_dict() != summary:
        raise ValueError("training summary contains coerced or noncanonical values")
    for field, expected in {
        "success": True,
        "unsupported_operation_fallback": False,
        "scientific": False,
        "hardware_matched": True,
        "cleanup_completed": True,
    }.items():
        _exact_bool(summary, field, expected)
    expected_text = {
        "failure_stage": "",
        "error": "",
        "profile_name": profile.name,
        "profile_version": profile.version,
        "profile_hash": profile.profile_hash,
        "candidate_source_hash": candidate_hash,
        "dtype": profile.dtype,
        "accelerator_kind": "cuda",
        "checkpoint_path": checkpoint_path.name,
        "event_log_path": event_path.name,
        "schema_name": "TrainingResult",
        "schema_version": "2.0",
    }
    for field, expected in expected_text.items():
        if summary[field] != expected:
            raise ValueError(f"training summary {field} differs from the run")
    selected_device = summary["device"]
    if not isinstance(selected_device, str) or _device_kind(selected_device) != "cuda":
        raise ValueError("training summary did not select CUDA")
    for field in (
        "initialization_seed",
        "data_seed",
        "development_seed",
        "dataloader_seed",
        "steps_completed",
        "examples_processed",
        "best_development_step",
        "peak_accelerator_allocated_bytes",
        "current_accelerator_allocated_bytes",
        "reserved_accelerator_allocated_bytes",
        "accelerator_total_memory_bytes",
        "parameter_count_metadata",
    ):
        _exact_integer(summary, field)
    if summary["steps_completed"] != profile.max_steps:
        raise ValueError("training summary steps differ from the smoke profile")
    if summary["examples_processed"] != profile.max_steps * profile.global_batch_size:
        raise ValueError("training summary examples differ from the smoke profile")
    if summary["best_development_step"] > profile.max_steps:
        raise ValueError("training summary best step exceeds the smoke profile")
    if summary["parameter_count_metadata"] < 1:
        raise ValueError("training summary parameter count must be positive")
    if summary["accelerator_total_memory_bytes"] < 1:
        raise ValueError("training summary lacks total accelerator memory")
    for field in (
        "best_development_exact_match_accuracy",
        "best_development_loss",
        "final_training_loss",
        "train_seconds",
    ):
        _finite_number(summary, field, minimum=0.0)
    accuracy = float(summary["best_development_exact_match_accuracy"])
    if accuracy > 1.0:
        raise ValueError("training summary development accuracy is outside [0, 1]")
    if not _is_lower_sha256(summary["checkpoint_sha256"]):
        raise ValueError("training summary checkpoint SHA-256 is invalid")
    if _sha256_file(checkpoint_path) != summary["checkpoint_sha256"]:
        raise ValueError("best checkpoint hash does not match the training summary")
    _validate_cuda_fingerprint(
        summary["accelerator_fingerprint"],
        selected_device=selected_device,
    )
    return parsed


def _load_training_events(
    path: Path,
    *,
    profile: TrainingProfile,
) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("training event log lacks a complete final record")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("training event log is not UTF-8") from error
    lines = text.splitlines()
    if len(lines) != profile.max_steps or any(not line for line in lines):
        raise ValueError("training event log does not contain the exact step roster")
    events: list[dict[str, Any]] = []
    previous_elapsed = -1.0
    for line_number, line in enumerate(lines, start=1):
        if len(line.encode("utf-8")) > 128 * 1024:
            raise ValueError(
                f"training event line {line_number} exceeds its size limit"
            )
        event = _strict_json_loads(
            line,
            label=f"training event line {line_number}",
        )
        if not isinstance(event, dict):
            raise ValueError(f"training event line {line_number} must be an object")
        _validate_json_security(event, label=f"training event line {line_number}")
        if set(event) != _TRAINING_EVENT_V2_FIELDS:
            raise ValueError(
                f"training event line {line_number} differs from the exact v2 schema"
            )
        _iso8601_timestamp(
            event["timestamp"],
            f"training_event[{line_number}].timestamp",
        )
        step = _exact_integer(event, "optimizer_step", minimum=1)
        if step != line_number:
            raise ValueError("smoke optimizer-step sequence is not contiguous")
        examples = _exact_integer(event, "examples_processed", minimum=1)
        if examples != step * profile.global_batch_size:
            raise ValueError("smoke event examples do not reconstruct from step")
        for field in ("loss", "learning_rate", "gradient_norm", "elapsed_seconds"):
            _finite_number(event, field, minimum=0.0)
        elapsed = float(event["elapsed_seconds"])
        if elapsed < previous_elapsed:
            raise ValueError("smoke event elapsed time decreases")
        previous_elapsed = elapsed
        is_validation = step % profile.validation_interval == 0
        for field in ("validation_loss", "validation_exact_match_accuracy"):
            value = event[field]
            if is_validation:
                _finite_number(event, field, minimum=0.0)
            elif value is not None:
                raise ValueError(
                    f"training event {field} must be null outside validation steps"
                )
        if is_validation and float(event["validation_exact_match_accuracy"]) > 1.0:
            raise ValueError("training event validation accuracy is outside [0, 1]")
        for field in (
            "current_accelerator_allocated_bytes",
            "reserved_accelerator_allocated_bytes",
            "peak_accelerator_allocated_bytes",
            "accelerator_total_memory_bytes",
        ):
            _exact_integer(event, field)
        if event["accelerator_total_memory_bytes"] < 1:
            raise ValueError("training event lacks total accelerator memory")
        decision = event["checkpoint_decision"]
        if is_validation:
            if decision not in {"best_development", "evaluated_not_best"}:
                raise ValueError("training validation event has an invalid decision")
        elif decision != "none":
            raise ValueError("training non-validation event has a checkpoint decision")
        events.append(event)
    return events


def _validate_tensor_mapping(value: object, *, field: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field} must be a nonempty tensor mapping")
    if any(
        not isinstance(name, str) or not isinstance(item, torch.Tensor)
        for name, item in value.items()
    ):
        raise ValueError(f"{field} must contain only named tensors")
    return value


def _validate_cuda_rng_state(value: object, *, field: str) -> str:
    if not isinstance(value, dict) or set(value) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_mps",
        "torch_cuda",
    }:
        raise ValueError(f"{field} has an invalid exact CUDA RNG schema")
    python_state = value["python"]
    if (
        not isinstance(python_state, tuple)
        or len(python_state) != 3
        or type(python_state[0]) is not int
        or python_state[0] != 3
        or not isinstance(python_state[1], tuple)
        or len(python_state[1]) != 625
        or any(type(item) is not int or item < 0 for item in python_state[1])
        or (
            python_state[2] is not None
            and (
                isinstance(python_state[2], bool)
                or not isinstance(python_state[2], (int, float))
                or not math.isfinite(float(python_state[2]))
            )
        )
    ):
        raise ValueError(f"{field}.python has an invalid random.getstate schema")
    numpy_state = value["numpy"]
    if not isinstance(numpy_state, dict) or set(numpy_state) != {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError(f"{field}.numpy has an invalid exact schema")
    keys = numpy_state["keys"]
    if (
        numpy_state["bit_generator"] != "MT19937"
        or not isinstance(keys, list)
        or len(keys) != 624
        or any(type(item) is not int or not 0 <= item < 2**32 for item in keys)
        or type(numpy_state["position"]) is not int
        or not 0 <= numpy_state["position"] <= 624
        or type(numpy_state["has_gauss"]) is not int
        or numpy_state["has_gauss"] not in {0, 1}
        or isinstance(numpy_state["cached_gaussian"], bool)
        or not isinstance(numpy_state["cached_gaussian"], (int, float))
        or not math.isfinite(float(numpy_state["cached_gaussian"]))
    ):
        raise ValueError(f"{field}.numpy contains invalid state values")

    def tensor_digest(item: object, tensor_field: str) -> dict[str, Any]:
        if (
            not isinstance(item, torch.Tensor)
            or item.device.type != "cpu"
            or item.dtype != torch.uint8
            or item.ndim != 1
            or item.numel() < 1
        ):
            raise ValueError(f"{tensor_field} must be a nonempty CPU uint8 vector")
        contiguous = item.detach().contiguous()
        return {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.numpy().tobytes()).hexdigest(),
        }

    cpu_record = tensor_digest(value["torch_cpu"], f"{field}.torch_cpu")
    mps_record = None
    if value["torch_mps"] is not None:
        mps_record = tensor_digest(value["torch_mps"], f"{field}.torch_mps")
    cuda_state = value["torch_cuda"]
    if not isinstance(cuda_state, list) or len(cuda_state) != 1:
        raise ValueError(f"{field}.torch_cuda must contain exactly one GPU state")
    cuda_record = tensor_digest(cuda_state[0], f"{field}.torch_cuda[0]")
    canonical = {
        "python": [python_state[0], list(python_state[1]), python_state[2]],
        "numpy": numpy_state,
        "torch_cpu": cpu_record,
        "torch_mps": mps_record,
        "torch_cuda": [cuda_record],
    }
    return canonical_sha256(canonical)


def _validate_cuda_checkpoints(
    *,
    output: Path,
    best: dict[str, Any],
    partial: dict[str, Any],
    latest: dict[str, Any],
    profile: TrainingProfile,
    candidate_hash: str,
    summary: dict[str, Any],
    seeds: TrainingSeedBundle,
    dependency_lock_hash: str,
) -> None:
    if set(best) != _BEST_CHECKPOINT_V2_FIELDS:
        raise ValueError("best checkpoint fields differ from the exact v2 schema")
    _validate_tensor_mapping(best["model_state"], field="best_checkpoint.model_state")
    for field in ("global_step", "examples_processed"):
        _exact_integer(best, field)
    for field in (
        "best_development_exact_match_accuracy",
        "best_development_loss",
    ):
        _finite_number(best, field, minimum=0.0)
    expected_best = {
        "checkpoint_kind": "best_evaluation_weights_v2",
        "global_step": summary["best_development_step"],
        "examples_processed": (
            summary["best_development_step"] * profile.global_batch_size
        ),
        "best_development_exact_match_accuracy": summary[
            "best_development_exact_match_accuracy"
        ],
        "best_development_loss": summary["best_development_loss"],
        "candidate_source_hash": candidate_hash,
        "profile_hash": profile.profile_hash,
        "task_adapter_version": DEFAULT_TASK.version,
        "task_adapter_hash": DEFAULT_TASK.config_hash,
        "seed_bundle": asdict(seeds),
        "seed_bundle_hash": seeds.bundle_hash,
        "trusted_component_set_sha256": trusted_component_set_sha256(),
        "dependency_lock_hash": dependency_lock_hash,
    }
    for field, expected in expected_best.items():
        if best[field] != expected:
            raise ValueError(f"best checkpoint {field} differs from the run")

    validation_arguments = {
        "candidate_hash": candidate_hash,
        "profile": profile,
        "task": DEFAULT_TASK,
        "seeds": seeds,
        "trusted_component_set_hash": trusted_component_set_sha256(),
        "dependency_lock_hash": dependency_lock_hash,
    }
    for label, checkpoint, expected_step in (
        ("partial resume checkpoint", partial, profile.checkpoint_interval),
        ("latest resume checkpoint", latest, profile.max_steps),
    ):
        if set(checkpoint) != _RESUME_CHECKPOINT_V2_FIELDS:
            raise ValueError(f"{label} fields differ from the exact v2 schema")
        try:
            _validate_resume(checkpoint, **validation_arguments)
        except ResumeMismatchError as error:
            raise ValueError(f"{label} failed its exact v2 identity") from error
        if checkpoint["global_step"] != expected_step:
            raise ValueError(f"{label} is not at its expected optimizer step")
        _validate_tensor_mapping(
            checkpoint["model_state"], field=f"{label}.model_state"
        )
        _validate_cuda_rng_state(checkpoint["rng_state"], field=f"{label}.rng_state")
    if _sha256_file(output / "partial_resume_checkpoint.pt") == _sha256_file(
        output / "latest_resume_checkpoint.pt"
    ):
        raise ValueError("partial and latest resume checkpoints are byte-identical")


def _validate_modal_canary_generator(generator: object) -> None:
    """Require the exact non-secret provider settings approved for canaries."""

    if not isinstance(generator, dict):
        raise ValueError("controller manifest lacks its generator contract")
    expected_fields = set(_MODAL_CANARY_GENERATOR_CONTRACT)
    observed_fields = set(generator)
    missing_fields = expected_fields - observed_fields
    unknown_fields = observed_fields - expected_fields
    if any(_is_credential_field(field) for field in unknown_fields):
        raise ValueError("controller manifest generator contains credential fields")
    if missing_fields:
        field = sorted(missing_fields)[0]
        raise ValueError(
            f"controller manifest generator {field} is missing from the "
            "frozen provider contract"
        )
    if unknown_fields:
        raise ValueError(
            "controller manifest generator fields differ from the frozen "
            "provider contract"
        )
    for field, expected in _MODAL_CANARY_GENERATOR_CONTRACT.items():
        observed = generator[field]
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"controller manifest generator {field} differs from the "
                "frozen provider contract"
            )


def _validate_modal_canary_provider_attempt(
    path: Path,
    *,
    harness: str,
    action_run_id: str,
    controller_run_id: str,
    modal_call_id: str,
) -> tuple[ProviderAttemptRecord, dict[str, int]]:
    """Require one successful, fully attributable, billable SDK attempt."""

    records = load_provider_attempt_ledger(path)
    if len(records) != 1:
        raise ValueError(
            "provider canary must contain exactly one actual API attempt"
        )
    record = records[0]
    expected = {
        "harness": harness,
        "action": "one_opportunity_engineering_canary",
        "controller_run_id": controller_run_id,
        "execution_backend": "modal",
        "action_run_id": action_run_id,
        "modal_call_id": modal_call_id,
        "attempt_ordinal": 1,
        "status": "success",
        "api_endpoint": OFFICIAL_OPENAI_API_BASE,
        "model": TARGET_MODEL,
        "generation_settings_sha256": (
            _MODAL_CANARY_GENERATION_SETTINGS_SHA256
        ),
        "usage_known": True,
        "error_class": None,
    }
    for field, expected_value in expected.items():
        if getattr(record, field) != expected_value:
            raise ValueError(
                f"provider attempt {field} differs from the canary contract"
            )
    if record.provider_response_id is None:
        raise ValueError("provider attempt lacks the official response ID")
    if record.provider_request_id is None:
        raise ValueError("provider attempt lacks the SDK-exposed request ID")
    totals = provider_attempt_totals(records)
    expected_counts = {
        "attempt_count": 1,
        "success_count": 1,
        "error_count": 0,
        "usage_known_count": 1,
    }
    for field, expected_value in expected_counts.items():
        if totals[field] != expected_value:
            raise ValueError(
                f"provider attempt aggregate {field} does not reconcile"
            )
    if totals["total_tokens"] != (
        totals["input_tokens"] + totals["output_tokens"]
    ):
        raise ValueError("provider attempt aggregate token totals do not reconcile")
    return record, totals


def _load_private_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{label} must be a nonempty, complete JSONL file")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line or len(line.encode("utf-8")) > 2_000_000:
            raise ValueError(f"{label} line {index} violates its byte policy")
        record = _strict_json_loads(line, label=f"{label} line {index}")
        if not isinstance(record, dict):
            raise ValueError(f"{label} line {index} must be a JSON object")
        _validate_json_security(record, label=f"{label} line {index}")
        records.append(record)
    return records


def _validate_private_canary_tree(controller: Path) -> tuple[int, int]:
    """Reject unsafe file kinds, unexpected extensions, and secret-like data."""

    metadata = controller.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("private canary controller root must be a regular directory")
    file_count = 0
    total_bytes = 0
    text_credential = re.compile(
        r"(?:Bearer[ \t]+[^\s]+|sk-[A-Za-z0-9_-]{12,})",
        re.IGNORECASE,
    )
    for path in sorted(controller.rglob("*")):
        relative = path.relative_to(controller)
        item = path.lstat()
        if stat.S_ISLNK(item.st_mode):
            raise ValueError("private canary staging may not contain symlinks")
        if stat.S_ISDIR(item.st_mode):
            continue
        if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
            raise ValueError(
                "private canary staging may contain only singly linked regular files"
            )
        if item.st_size < 1 or item.st_size > MAX_ARTIFACT_DOWNLOAD_FILE_BYTES:
            raise ValueError("private canary file violates the publication byte cap")
        total_bytes += item.st_size
        if total_bytes > MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES:
            raise ValueError("private canary staging exceeds the publication byte cap")
        file_count += 1

        suffix = path.suffix.lower()
        inside_git = len(relative.parts) >= 2 and relative.parts[0:2] == (
            "accepted_lineage",
            ".git",
        )
        inside_registry = relative.parts[0] == "architecture_hash_registry"
        if suffix == ".pt":
            if "candidate_training" not in relative.parts:
                raise ValueError("checkpoint file is outside candidate_training")
            continue
        if not suffix and (inside_git or inside_registry):
            continue
        if inside_git and suffix in {".idx", ".pack", ".rev"}:
            continue
        if suffix not in _PRIVATE_CANARY_TEXT_SUFFIXES:
            raise ValueError(
                f"private canary file has an unapproved type: {relative.as_posix()}"
            )
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"private canary text artifact is not UTF-8: {relative.as_posix()}"
            ) from error
        if text_credential.search(text):
            raise ValueError("private canary staging contains credential-shaped text")
        if suffix == ".json":
            payload = _strict_json_loads(text, label=relative.as_posix())
            _validate_json_security(payload, label=relative.as_posix())
        elif suffix == ".jsonl":
            _load_private_jsonl(path, label=relative.as_posix())
    if file_count < 1:
        raise ValueError("private canary staging contains no files")
    return file_count, total_bytes


def _validate_private_cuda_candidate(
    output: Path,
    *,
    execution_context: ExecutionContextV1,
) -> dict[str, str]:
    """Validate one complete CUDA candidate without requiring a final run manifest."""

    _validate_cuda_candidate_roster(output)
    graph_path = output / "candidate_graph.json"
    checkpoint_path = output / "best_checkpoint.pt"
    event_path = output / "training_events.jsonl"
    graph_text = graph_path.read_text(encoding="utf-8")
    graph_validation = validate_ir_candidate_json(graph_text)
    if not graph_validation.valid or graph_validation.graph is None:
        raise ValueError("private canary candidate graph is not valid Architecture IR")
    candidate_hash = _sha256_file(graph_path)
    graph_hash = graph_validation.graph_hash
    architecture_hash = graph_validation.graph.architecture_hash
    summary = _safe_json_object(output / "training_summary.json")
    manifest = _safe_json_object(output / "training_manifest.json")
    runtime_validity = _safe_json_object(output / "runtime_validity.json")
    _validate_json_security(runtime_validity, label="runtime validity artifact")
    seeds, dependency_lock_hash = _validate_cuda_v2_records(
        output=output,
        summary=summary,
        manifest=manifest,
        profile=SMOKE_TRAIN_CUDA_V2,
        candidate_hash=candidate_hash,
        candidate_graph_hash=graph_hash,
        checkpoint_path=checkpoint_path,
        event_path=event_path,
        require_modal_context=False,
    )
    if manifest.get("execution_context") != execution_context.to_dict():
        raise ValueError(
            "private canary candidate context differs from the outer Modal call"
        )
    _load_training_events(event_path, profile=SMOKE_TRAIN_CUDA_V2)

    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    partial = torch.load(
        output / "partial_resume_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    latest = torch.load(
        output / "latest_resume_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not all(isinstance(item, dict) for item in (best, partial, latest)):
        raise ValueError("private canary checkpoints must be mappings")
    _validate_cuda_checkpoints(
        output=output,
        best=best,
        partial=partial,
        latest=latest,
        profile=SMOKE_TRAIN_CUDA_V2,
        candidate_hash=candidate_hash,
        summary=summary,
        seeds=seeds,
        dependency_lock_hash=dependency_lock_hash,
    )
    interpreted = load_and_build_ir_candidate(
        graph_path,
        int(summary["initialization_seed"]),
    )
    initial_state = interpreted.model.state_dict()
    trained_state = best.get("model_state")
    if not isinstance(trained_state, dict) or set(trained_state) != set(initial_state):
        raise ValueError("private canary checkpoint architecture is inconsistent")
    parameters_changed = False
    for name, initial in initial_state.items():
        trained = trained_state[name]
        if (
            not isinstance(trained, torch.Tensor)
            or trained.shape != initial.shape
            or trained.dtype != initial.dtype
        ):
            raise ValueError("private canary checkpoint tensor contract is invalid")
        parameters_changed = parameters_changed or not torch.equal(initial, trained)
    if not parameters_changed:
        raise ValueError("private canary checkpoint is a fresh initialization")
    return {
        "candidate_sha256": candidate_hash,
        "graph_sha256": graph_hash,
        "architecture_sha256": architecture_hash,
    }


def _validate_private_architecture_registry(
    directory: Path,
    *,
    architecture_hashes: set[str],
) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("private canary architecture registry is unsafe")
    entries = tuple(directory.iterdir())
    if {entry.name for entry in entries} != architecture_hashes:
        raise ValueError("private canary architecture registry does not reconcile")
    for entry in entries:
        if (
            entry.is_symlink()
            or not entry.is_file()
            or entry.read_text(encoding="ascii") != f"{entry.name}\n"
        ):
            raise ValueError("private canary architecture claim is invalid")


def _validate_private_prompt_snapshot(
    controller: Path,
    *,
    harness: str,
    controller_manifest: dict[str, Any],
) -> None:
    program_name = (
        "greedy_autoresearch_program.md"
        if harness == "greedy_autoresearch"
        else "semantic_autoresearch_program.md"
    )
    expected = {
        "architecture_ir_contract.md",
        "combined_system_prompt.md",
        program_name,
        "shared_system.md",
        "shared_task.md",
    }
    snapshot = controller / "prompt_snapshot"
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("native prompt snapshot is unsafe")
    if {entry.name for entry in snapshot.iterdir()} != expected:
        raise ValueError("native prompt snapshot has an unexpected roster")
    components = [
        "shared_system",
        "shared_task",
        "architecture_ir_contract",
        program_name.removesuffix(".md"),
    ]
    combined = "\n\n".join(
        (snapshot / f"{name}.md").read_text(encoding="utf-8")
        for name in components
    )
    if (snapshot / "combined_system_prompt.md").read_text(
        encoding="utf-8"
    ) != combined:
        raise ValueError("native combined prompt snapshot does not reconstruct")
    protocol = controller_manifest.get("prompt_protocol")
    if not isinstance(protocol, dict) or set(protocol) != {
        "components",
        "combined_system_prompt_sha256",
        "message_hash",
        "snapshot_directory",
    }:
        raise ValueError("native prompt protocol manifest is invalid")
    if (
        protocol["combined_system_prompt_sha256"]
        != hashlib.sha256(combined.encode("utf-8")).hexdigest()
        or protocol["message_hash"] != "sha256_canonical_json_v1"
        or protocol["snapshot_directory"] != "prompt_snapshot"
    ):
        raise ValueError("native prompt protocol does not bind its snapshot")
    component_records = protocol["components"]
    if not isinstance(component_records, list) or len(component_records) != 4:
        raise ValueError("native prompt protocol component roster is invalid")
    by_name = {
        item.get("name"): item
        for item in component_records
        if isinstance(item, dict)
    }
    if set(by_name) != set(components):
        raise ValueError("native prompt protocol component names are invalid")
    for name in components:
        record = by_name[name]
        if set(record) != {"name", "source_path", "sha256"}:
            raise ValueError("native prompt protocol component schema is invalid")
        if record["sha256"] != _sha256_file(snapshot / f"{name}.md"):
            raise ValueError("native prompt protocol component hash differs")


def _validate_private_native_artifacts(
    controller: Path,
    *,
    harness: str,
    controller_run_id: str,
    candidate_hashes: set[str],
    provider_attempt: ProviderAttemptRecord,
) -> None:
    artifacts = controller / "artifacts"
    expected_fixed = {
        "0001.messages.json",
        "0001.prompt.md",
        "0001.response.txt",
    }
    if harness == "semantic_autoresearch":
        expected_fixed.add("0000_seed.ir.json")
    entries = tuple(artifacts.iterdir())
    ir_candidates = [
        item
        for item in entries
        if re.fullmatch(r"0001_[0-9a-f]{12}\.ir\.json", item.name)
    ]
    if len(ir_candidates) != 1 or {item.name for item in entries} != (
        expected_fixed | {ir_candidates[0].name}
    ):
        raise ValueError("native proposal artifact roster is not exactly one opportunity")
    child_path = ir_candidates[0]
    child_validation = validate_ir_candidate_json(
        child_path.read_text(encoding="utf-8")
    )
    child_hash = _sha256_file(child_path)
    if (
        not child_validation.valid
        or child_validation.graph is None
        or child_hash not in candidate_hashes
        or child_path.name != f"0001_{child_hash[:12]}.ir.json"
    ):
        raise ValueError("native proposal IR is not bound to candidate training")
    if harness == "semantic_autoresearch":
        seed_hash = _sha256_file(artifacts / "0000_seed.ir.json")
        if seed_hash not in candidate_hashes:
            raise ValueError("semantic seed IR is not bound to candidate training")

    # Messages are a JSON array, so use the strict loader directly.
    messages_payload = _strict_json_loads(
        (artifacts / "0001.messages.json").read_text(encoding="utf-8"),
        label="native provider messages",
    )
    if (
        not isinstance(messages_payload, list)
        or len(messages_payload) != 2
        or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] != expected_role
            or not isinstance(message["content"], str)
            or not message["content"]
            for message, expected_role in zip(
                messages_payload, ("system", "user"), strict=True
            )
        )
    ):
        raise ValueError("native provider message payload is invalid")
    _validate_json_security(messages_payload, label="native provider messages")
    if (artifacts / "0001.prompt.md").read_text(
        encoding="utf-8"
    ) != messages_payload[-1]["content"]:
        raise ValueError("native prompt artifact differs from the provider request")
    if not (artifacts / "0001.response.txt").read_text(encoding="utf-8").strip():
        raise ValueError("native provider response artifact is empty")

    lineage = _load_private_jsonl(controller / "lineage.jsonl", label="native lineage")
    if len(lineage) != 2:
        raise ValueError("native lineage must contain seed plus one proposal")
    opportunity_field = (
        "proposal_opportunity"
        if harness == "greedy_autoresearch"
        else "opportunity_index"
    )
    if [record.get(opportunity_field) for record in lineage] != [0, 1]:
        raise ValueError("native lineage opportunity sequence is invalid")
    observed_hashes: set[str] = set()
    for record in lineage:
        _require_exact_fields(
            record,
            _NATIVE_LINEAGE_FIELDS[harness],
            label=f"{harness} native lineage record",
        )
        candidate_id = record.get("candidate_id")
        if candidate_id not in candidate_hashes:
            raise ValueError("native lineage candidate is not trained")
        observed_hashes.add(candidate_id)
        if (
            record.get("run_id") != controller_run_id
            or record.get("condition") != harness
            or record.get("code_hash") != candidate_id
            or record.get("evaluation_candidate_id") != f"candidate-{candidate_id}"
        ):
            raise ValueError("native lineage identity binding is invalid")
        for field in (
            "execution_ok",
            "transformer_valid",
            "eligible_for_parent",
        ):
            _exact_bool(record, field, True)
        _exact_bool(record, "infrastructure_failure", False)
        if record.get("failure_stage") != "":
            raise ValueError("native lineage contains an evaluation failure")
    if observed_hashes != candidate_hashes:
        raise ValueError("native lineage does not account for both trained candidates")
    proposal = lineage[1]
    if (
        proposal.get("input_tokens") != provider_attempt.input_tokens
        or proposal.get("output_tokens") != provider_attempt.output_tokens
    ):
        raise ValueError("native lineage usage differs from the provider ledger")


def _validate_private_native_terminal_state(
    controller: Path,
    *,
    harness: str,
    candidate_hashes: set[str],
    summary: dict[str, Any],
) -> None:
    if harness == "greedy_autoresearch":
        incumbent = controller / "incumbent.ir.json"
        incumbent_validation = validate_ir_candidate_json(
            incumbent.read_text(encoding="utf-8")
        )
        accepted = controller / "accepted_lineage"
        if (
            not incumbent_validation.valid
            or _sha256_file(incumbent) not in candidate_hashes
            or accepted.is_symlink()
            or not accepted.is_dir()
            or {item.name for item in accepted.iterdir()}
            != {".git", "candidate.ir.json"}
            or not (accepted / ".git").is_dir()
            or not any((accepted / ".git").rglob("*"))
            or _sha256_file(accepted / "candidate.ir.json") != _sha256_file(incumbent)
        ):
            raise ValueError("greedy accepted-lineage state is invalid")
        return

    archive = _safe_json_object(controller / "semantic_archive.json")
    expected_axes = [SEMANTIC_METRIC_NAMES[axis] for axis in CATEGORY_CODES]
    if set(archive) != {
        "schema_name",
        "schema_version",
        "axes",
        "coverage_cells",
        "novelty_role",
        "scientific_novelty_claim",
        "cells",
    }:
        raise ValueError("semantic archive fields differ from schema v2")
    if (
        archive["schema_name"] != "semantic_autoresearch_archive"
        or archive["schema_version"] != "2.0"
        or archive["axes"] != expected_axes
        or archive["novelty_role"] != "exploratory_coverage_tiebreak_only"
    ):
        raise ValueError("semantic archive identity is invalid")
    _exact_bool(archive, "scientific_novelty_claim", False)
    cells = archive["cells"]
    coverage = _nonnegative_integer(archive, "coverage_cells")
    if (
        not isinstance(cells, list)
        or not cells
        or coverage != len(cells)
        or summary.get("semantic_archive_cells") != coverage
    ):
        raise ValueError("semantic archive coverage does not reconcile")
    seen_candidates: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {
            "cell",
            "signature",
            "candidate_id",
            "lineage_record_id",
            "source_path",
            "search_score",
            "public_accuracy",
            "discovered_opportunity",
            "parent_uses",
        }:
            raise ValueError("semantic archive cell schema is invalid")
        candidate_id = cell["candidate_id"]
        if candidate_id not in candidate_hashes or candidate_id in seen_candidates:
            raise ValueError("semantic archive candidate identity is invalid")
        seen_candidates.add(candidate_id)
        source = cell["source_path"]
        if (
            not isinstance(source, str)
            or not source.startswith("artifacts/")
            or PurePosixPath(source).is_absolute()
            or ".." in PurePosixPath(source).parts
        ):
            raise ValueError("semantic archive source path is invalid")
        source_path = controller / PurePosixPath(source)
        if not source_path.is_file() or _sha256_file(source_path) != candidate_id:
            raise ValueError("semantic archive source does not bind its candidate")
        signature = cell["signature"]
        if (
            not isinstance(signature, list)
            or len(signature) != len(expected_axes)
            or any(type(value) is not int for value in signature)
        ):
            raise ValueError("semantic archive signature is invalid")
        _exact_integer(cell, "discovered_opportunity")
        _exact_integer(cell, "parent_uses")
        _finite_number(cell, "search_score")
        _finite_number(cell, "public_accuracy")


def _validate_openevolve_program(
    path: Path,
    *,
    require_prompt: bool,
    require_artifacts: bool,
) -> tuple[str, str, dict[str, Any]]:
    program = _safe_json_object(path)
    if set(program) != _OPENEVOLVE_PROGRAM_FIELDS:
        raise ValueError("OpenEvolve program fields differ from the frozen schema")
    if (
        not isinstance(program["id"], str)
        or not program["id"]
        or program["language"] != "json"
        or not isinstance(program["code"], str)
    ):
        raise ValueError("OpenEvolve program identity is invalid")
    validation = validate_ir_candidate_json(program["code"])
    if not validation.valid or validation.graph is None:
        raise ValueError("OpenEvolve stored an invalid Architecture IR program")
    metrics = program["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("OpenEvolve program metrics are invalid")
    for field in ("execution_ok", "transformer_valid", "eligible_for_parent"):
        if metrics.get(field) != 1.0:
            raise ValueError("OpenEvolve program lacks a successful evaluation")
    if require_prompt:
        prompts = program["prompts"]
        if not isinstance(prompts, dict) or not prompts:
            raise ValueError("OpenEvolve checkpoint lacks the actual provider prompt")
    artifacts_json = program["artifacts_json"]
    if artifacts_json is None and not require_artifacts:
        return (
            hashlib.sha256(program["code"].encode("utf-8")).hexdigest(),
            validation.graph.architecture_hash,
            program,
        )
    if not isinstance(artifacts_json, str) or not artifacts_json:
        raise ValueError("OpenEvolve program lacks evaluator artifact bindings")
    artifacts = _strict_json_loads(
        artifacts_json,
        label="OpenEvolve program evaluator artifacts",
    )
    _validate_json_security(artifacts, label="OpenEvolve evaluator artifacts")
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("candidate_graph_hash") != validation.graph_hash
        or artifacts.get("candidate_architecture_hash")
        != validation.graph.architecture_hash
        or artifacts.get("failure_stage") != ""
        or artifacts.get("infrastructure_failure") is not False
    ):
        raise ValueError("OpenEvolve evaluator artifacts do not bind the program")
    return (
        hashlib.sha256(program["code"].encode("utf-8")).hexdigest(),
        validation.graph.architecture_hash,
        program,
    )


def _validate_private_openevolve_artifacts(
    controller: Path,
    *,
    harness: str,
    candidate_hashes: set[str],
) -> None:
    best = controller / "best"
    if {item.name for item in best.iterdir()} != {
        "best_program.json",
        "best_program_info.json",
    }:
        raise ValueError("OpenEvolve best-program roster is invalid")
    best_validation = validate_ir_candidate_json(
        (best / "best_program.json").read_text(encoding="utf-8")
    )
    if (
        not best_validation.valid
        or hashlib.sha256(
            (best / "best_program.json").read_bytes()
        ).hexdigest()
        not in candidate_hashes
    ):
        raise ValueError("OpenEvolve best program is not a trained candidate")
    best_info = _safe_json_object(best / "best_program_info.json")
    if set(best_info) != {
        "id",
        "generation",
        "iteration",
        "timestamp",
        "parent_id",
        "metrics",
        "language",
        "saved_at",
    }:
        raise ValueError("OpenEvolve best-program info schema is invalid")

    database_programs = controller / "database" / "programs"
    if (controller / "database").is_symlink() or not database_programs.is_dir():
        raise ValueError("OpenEvolve database roster is unsafe")
    database_entries = tuple(database_programs.iterdir())
    if len(database_entries) != 2 or any(item.suffix != ".json" for item in database_entries):
        raise ValueError("OpenEvolve database must contain exactly two programs")
    program_hashes: set[str] = set()
    program_ids: set[str] = set()
    for path in database_entries:
        candidate_hash, _architecture_hash, program = _validate_openevolve_program(
            path,
            require_prompt=False,
            require_artifacts=False,
        )
        if path.name != f"{program['id']}.json":
            raise ValueError("OpenEvolve database program filename is invalid")
        program_hashes.add(candidate_hash)
        program_ids.add(program["id"])
    if program_hashes != candidate_hashes or len(program_ids) != 2:
        raise ValueError("OpenEvolve database does not bind both trained candidates")

    checkpoints = controller / "checkpoints"
    if {item.name for item in checkpoints.iterdir()} != {"checkpoint_1"}:
        raise ValueError("OpenEvolve checkpoint roster is not one opportunity")
    checkpoint = checkpoints / "checkpoint_1"
    if {item.name for item in checkpoint.iterdir()} != {
        "best_program.json",
        "best_program_info.json",
        "metadata.json",
        "programs",
    }:
        raise ValueError("OpenEvolve checkpoint fields differ from the frozen roster")
    checkpoint_metadata = _safe_json_object(checkpoint / "metadata.json")
    _require_exact_fields(
        checkpoint_metadata,
        _OPENEVOLVE_CHECKPOINT_METADATA_FIELDS,
        label="OpenEvolve checkpoint metadata",
    )
    checkpoint_programs = tuple((checkpoint / "programs").iterdir())
    if len(checkpoint_programs) != 2:
        raise ValueError("OpenEvolve checkpoint must contain exactly two programs")
    checkpoint_hashes: set[str] = set()
    checkpoint_ids: set[str] = set()
    for path in checkpoint_programs:
        candidate_hash, _architecture_hash, program = _validate_openevolve_program(
            path,
            require_prompt=False,
            require_artifacts=True,
        )
        checkpoint_hashes.add(candidate_hash)
        checkpoint_ids.add(program["id"])
    if checkpoint_hashes != candidate_hashes or checkpoint_ids != program_ids:
        raise ValueError("OpenEvolve checkpoint programs differ from its database")
    if not any(
        isinstance(_safe_json_object(path).get("prompts"), dict)
        and _safe_json_object(path).get("prompts")
        for path in checkpoint_programs
    ):
        raise ValueError("OpenEvolve checkpoint lacks the actual paid prompt")

    terminal = _load_private_jsonl(
        controller / "proposal_terminal_outcomes.jsonl",
        label="OpenEvolve terminal outcomes",
    )
    if terminal != [
        {"candidate_produced": True, "iteration": 1, "status": "candidate"}
    ]:
        raise ValueError("OpenEvolve terminal ledger is not one successful opportunity")
    trace = _load_private_jsonl(
        controller / "evolution_trace.jsonl",
        label="OpenEvolve evolution trace",
    )
    if len(trace) != 1:
        raise ValueError("OpenEvolve evolution trace is not one opportunity")
    trace_record = trace[0]
    _require_exact_fields(
        trace_record,
        _OPENEVOLVE_TRACE_FIELDS,
        label="OpenEvolve evolution trace",
    )
    trace_metadata = trace_record["metadata"]
    if not isinstance(trace_metadata, dict):
        raise ValueError("OpenEvolve evolution trace metadata must be an object")
    _require_exact_fields(
        trace_metadata,
        _OPENEVOLVE_TRACE_METADATA_FIELDS,
        label="OpenEvolve evolution trace metadata",
    )
    trace_prompt = trace_record["prompt"]
    if not isinstance(trace_prompt, dict):
        raise ValueError("OpenEvolve evolution trace prompt must be an object")
    _require_exact_fields(
        trace_prompt,
        _OPENEVOLVE_TRACE_PROMPT_FIELDS,
        label="OpenEvolve evolution trace prompt",
    )
    trace_artifacts = trace_record["artifacts"]
    if not isinstance(trace_artifacts, dict):
        raise ValueError("OpenEvolve evolution trace artifacts must be an object")
    _require_exact_fields(
        trace_artifacts,
        _OPENEVOLVE_TRACE_ARTIFACT_FIELDS,
        label="OpenEvolve evolution trace artifacts",
    )
    if (
        trace_record.get("iteration") != 1
        or trace_record.get("parent_id") not in program_ids
        or trace_record.get("child_id") not in program_ids
        or trace_record.get("parent_id") == trace_record.get("child_id")
        or not isinstance(trace_record.get("prompt"), dict)
        or not trace_record.get("prompt")
        or not isinstance(trace_record.get("llm_response"), str)
        or not trace_record.get("llm_response")
    ):
        raise ValueError("OpenEvolve evolution trace lacks provider/evaluation binding")
    logs = tuple((controller / "logs").iterdir())
    if len(logs) != 1 or not re.fullmatch(
        r"openevolve_[0-9]{8}_[0-9]{6}\.log", logs[0].name
    ):
        raise ValueError("OpenEvolve log roster is invalid")


def validate_private_canary_staging(
    controller_directory: str | Path,
    *,
    harness: str,
    execution_context: ExecutionContextV1,
) -> dict[str, Any]:
    """Validate a private provider canary before any artifact publication.

    This gate intentionally consumes only the off-Volume ``controller/`` tree.
    It requires no artifact manifest, makes no provider or Modal call, and does
    not mutate the staging tree.
    """

    if harness not in CANARY_ORDER:
        raise ValueError("private canary names an unknown harness")
    if not isinstance(execution_context, ExecutionContextV1):
        raise TypeError("private canary execution context must be ExecutionContextV1")
    if (
        execution_context.execution_backend != "modal"
        or execution_context.function_name != f"canary_{harness}"
        or execution_context.modal_call_id is None
        or execution_context.modal_image_id is None
    ):
        raise ValueError("private canary execution context is incomplete")
    controller = Path(controller_directory)
    if not controller.is_absolute():
        controller = controller.resolve()
    file_count, total_bytes = _validate_private_canary_tree(controller)
    observed_top_level = {entry.name for entry in controller.iterdir()}
    if observed_top_level != _PRIVATE_CANARY_TOP_LEVEL_ROSTERS[harness]:
        missing = sorted(_PRIVATE_CANARY_TOP_LEVEL_ROSTERS[harness] - observed_top_level)
        extra = sorted(observed_top_level - _PRIVATE_CANARY_TOP_LEVEL_ROSTERS[harness])
        raise ValueError(
            "private canary top-level roster differs from the harness contract "
            f"(missing={missing}, extra={extra})"
        )

    manifest = _safe_json_object(controller / "run_manifest.json")
    summary_name = (
        "run_summary.json" if harness in _NATIVE_CANARY_HARNESSES else "run_result.json"
    )
    summary = _safe_json_object(controller / summary_name)
    _require_exact_fields(
        manifest,
        _CONTROLLER_MANIFEST_FIELDS[harness],
        label=f"{harness} ControllerRunManifest",
    )
    _require_exact_fields(
        summary,
        _CONTROLLER_SUMMARY_FIELDS[harness],
        label=f"{harness} controller summary",
    )
    if (
        manifest.get("schema_name") != "ControllerRunManifest"
        or manifest.get("schema_version") != "2.0"
        or manifest.get("condition") != harness
    ):
        raise ValueError("private canary controller manifest identity is invalid")
    summary_schema = (
        "ControllerRunSummary"
        if harness in _NATIVE_CANARY_HARNESSES
        else "ControllerRunResult"
    )
    if (
        summary.get("schema_name") != summary_schema
        or summary.get("schema_version") != "2.0"
        or summary.get("condition") != harness
    ):
        raise ValueError("private canary controller summary identity is invalid")
    controller_run_id = manifest.get("run_id")
    if not isinstance(controller_run_id, str) or summary.get("run_id") != controller_run_id:
        raise ValueError("private canary controller run IDs do not reconcile")
    for field, expected in {
        "candidate_budget": 2,
        "mutation_budget": 1,
        "candidate_training_budget": 2,
        "maximum_provider_attempts": 1,
    }.items():
        if _nonnegative_integer(manifest, field) != expected:
            raise ValueError(f"private canary manifest {field} is invalid")
    _exact_bool(manifest, "authoritative_scientific_evidence", False)
    _validate_modal_canary_generator(manifest.get("generator"))
    if (
        manifest.get("provider_attempt_ledger")
        != PROVIDER_ATTEMPT_LEDGER_FILENAME
        or manifest.get("provider_attempt_schema") != PROVIDER_ATTEMPT_SCHEMA
    ):
        raise ValueError("private canary provider ledger contract is invalid")
    provider_attempt, provider_totals = _validate_modal_canary_provider_attempt(
        controller / PROVIDER_ATTEMPT_LEDGER_FILENAME,
        harness=harness,
        action_run_id=execution_context.run_id,
        controller_run_id=controller_run_id,
        modal_call_id=execution_context.modal_call_id,
    )
    training = manifest.get("training")
    if not isinstance(training, dict):
        raise ValueError("private canary lacks its training contract")
    for field, expected in {
        "profile": SMOKE_TRAIN_CUDA_V2.name,
        "profile_version": SMOKE_TRAIN_CUDA_V2.version,
        "profile_hash": SMOKE_TRAIN_CUDA_V2.profile_hash,
        "device": "cuda",
        "allow_cpu_for_tests": False,
    }.items():
        observed = str(training.get(field)) if field == "profile_version" else training.get(field)
        if observed != expected:
            raise ValueError(f"private canary training {field} is invalid")
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("private canary lacks its evaluation contract")
    if (
        evaluation.get("profile") != "smoke_eval_v1"
        or _nonnegative_integer(evaluation, "case_count") != 24
    ):
        raise ValueError("private canary evaluation contract is invalid")
    _exact_bool(evaluation, "scientific", False)

    if harness in _NATIVE_CANARY_HARNESSES:
        if manifest.get("run_mode") != "engineering_pilot":
            raise ValueError("native private canary is not an engineering pilot")
        _exact_bool(manifest, "exploratory_only", True)
        if (
            _nonnegative_integer(summary, "proposal_opportunities_requested") != 1
            or _nonnegative_integer(summary, "proposal_opportunities_terminal") != 1
        ):
            raise ValueError("native private canary opportunity accounting is invalid")
    else:
        _exact_bool(manifest, "engineering_pilot", True)
        if _nonnegative_integer(manifest, "proposal_opportunities") != 1:
            raise ValueError("OpenEvolve private canary opportunity budget is invalid")
        for field, expected in {
            "completed": True,
            "engineering_pilot": True,
            "authoritative_scientific_evidence": False,
        }.items():
            _exact_bool(summary, field, expected)
        if (
            _nonnegative_integer(summary, "proposal_opportunities_requested") != 1
            or _nonnegative_integer(summary, "proposal_opportunities_completed") != 1
            or summary.get("proposal_accounting_errors") != []
            or summary.get("failure_stage") != ""
        ):
            raise ValueError("OpenEvolve private canary opportunity accounting is invalid")

    training_root = controller / "candidate_training"
    candidate_directories = tuple(sorted(training_root.iterdir()))
    if len(candidate_directories) != 2 or any(
        item.is_symlink() or not item.is_dir() for item in candidate_directories
    ):
        raise ValueError("private canary must contain exactly two trained candidates")
    if harness in _NATIVE_CANARY_HARNESSES:
        names = [item.name for item in candidate_directories]
        if not re.fullmatch(r"0000_[0-9a-f]{12}", names[0]) or not re.fullmatch(
            r"0001_[0-9a-f]{12}", names[1]
        ):
            raise ValueError("native candidate-training directory names are invalid")
    elif any(
        not re.fullmatch(r"[0-9a-f]{12}_[0-9a-f]{8}", item.name)
        for item in candidate_directories
    ):
        raise ValueError("OpenEvolve candidate-training directory names are invalid")
    candidate_reports = [
        _validate_private_cuda_candidate(
            item,
            execution_context=execution_context,
        )
        for item in candidate_directories
    ]
    candidate_hashes = {item["candidate_sha256"] for item in candidate_reports}
    graph_hashes = {item["graph_sha256"] for item in candidate_reports}
    architecture_hashes = {item["architecture_sha256"] for item in candidate_reports}
    if not all(len(values) == 2 for values in (candidate_hashes, graph_hashes, architecture_hashes)):
        raise ValueError("private canary did not train two distinct architectures")
    if manifest.get("initial_candidate_hash") not in candidate_hashes:
        raise ValueError("private canary initial candidate was not trained")
    if manifest.get("initial_architecture_hash") not in architecture_hashes:
        raise ValueError("private canary initial architecture was not registered")
    _validate_private_architecture_registry(
        controller / "architecture_hash_registry",
        architecture_hashes=architecture_hashes,
    )
    if harness in _NATIVE_CANARY_HARNESSES:
        _validate_private_prompt_snapshot(
            controller,
            harness=harness,
            controller_manifest=manifest,
        )
        _validate_private_native_artifacts(
            controller,
            harness=harness,
            controller_run_id=controller_run_id,
            candidate_hashes=candidate_hashes,
            provider_attempt=provider_attempt,
        )
        _validate_private_native_terminal_state(
            controller,
            harness=harness,
            candidate_hashes=candidate_hashes,
            summary=summary,
        )
    else:
        _validate_private_openevolve_artifacts(
            controller,
            harness=harness,
            candidate_hashes=candidate_hashes,
        )
    return {
        "schema_name": "PrivateProviderCanaryStagingValidation",
        "schema_version": "1.0",
        "valid": True,
        "harness": harness,
        "controller_run_id": controller_run_id,
        "candidate_count": 2,
        "provider_attempt_count": provider_totals["attempt_count"],
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _validate_cuda_fingerprint(
    fingerprint: object,
    *,
    selected_device: str,
) -> dict[str, Any]:
    if not isinstance(fingerprint, dict):
        raise ValueError("CUDA smoke lacks accelerator fingerprint evidence")
    required_fields = {
        "requested_device",
        "selected_device",
        "accelerator_kind",
        "gpu_name",
        "gpu_count",
        "compute_capability",
        "cuda_runtime",
        "cuda_driver",
        "torch_version",
        "host_platform",
    }
    if set(fingerprint) != required_fields:
        raise ValueError("CUDA accelerator fingerprint fields differ from schema v2")
    if _device_kind(fingerprint.get("requested_device")) != "cuda":
        raise ValueError("CUDA fingerprint did not record a CUDA request")
    if fingerprint.get("selected_device") != selected_device:
        raise ValueError("CUDA fingerprint selected device differs from the summary")
    if fingerprint.get("accelerator_kind") != "cuda":
        raise ValueError("CUDA fingerprint accelerator kind is invalid")
    for field in (
        "gpu_name",
        "compute_capability",
        "cuda_runtime",
        "torch_version",
        "host_platform",
    ):
        value = fingerprint.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"CUDA fingerprint {field} must be nonempty text")
    if _nonnegative_integer(fingerprint, "gpu_count") < 1:
        raise ValueError("CUDA fingerprint must expose at least one GPU")
    driver = fingerprint.get("cuda_driver")
    if driver is not None and (not isinstance(driver, str) or not driver.strip()):
        raise ValueError("CUDA fingerprint driver must be nonempty text or null")
    return fingerprint


def _validate_cuda_v2_records(
    *,
    output: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    profile: TrainingProfile,
    candidate_hash: str,
    candidate_graph_hash: str,
    checkpoint_path: Path,
    event_path: Path,
    require_modal_context: bool,
) -> tuple[TrainingSeedBundle, str]:
    _validate_cuda_summary(
        summary,
        profile=profile,
        candidate_hash=candidate_hash,
        checkpoint_path=checkpoint_path,
        event_path=event_path,
    )
    return _validate_cuda_manifest(
        manifest,
        output=output,
        profile=profile,
        candidate_hash=candidate_hash,
        candidate_graph_hash=candidate_graph_hash,
        summary=summary,
        require_modal_context=require_modal_context,
    )


def _validate_existing_smoke(
    training_output_dir: str | Path | None,
    *,
    project_root: str | Path = ROOT,
    profile: TrainingProfile,
    accelerator_kind: str,
    require_modal_context: bool,
) -> dict[str, Any]:
    """Check smoke artifacts for consistency without proving execution origin."""

    accelerator_label = accelerator_kind.upper()
    legacy_mps = profile.version == "1" and accelerator_kind == "mps"
    if training_output_dir is None:
        return {
            "provided": False,
            "valid": False,
            "artifact_self_consistent": False,
            "execution_origin_attested": False,
            "training_started_by_validator": False,
            "profile": profile.name,
            "accelerator_kind": accelerator_kind,
            "errors": [f"no existing {accelerator_label} smoke output was provided"],
        }

    output = Path(training_output_dir).resolve()
    root = Path(project_root).resolve()
    summary_path = output / "training_summary.json"
    manifest_path = output / "training_manifest.json"
    graph_path = output / "candidate_graph.json"
    source_path = output / "candidate_source.py"
    checkpoint_path = output / "best_checkpoint.pt"
    event_path = output / "training_events.jsonl"
    errors: list[str] = []
    if not legacy_mps:
        try:
            _validate_cuda_candidate_roster(output)
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"{type(error).__name__}: {error}")
    required = (
        summary_path,
        manifest_path,
        checkpoint_path,
        event_path,
    )
    for path in required:
        if not path.is_file():
            errors.append(f"missing smoke artifact: {path.name}")
        elif path.is_symlink():
            errors.append(f"smoke artifact may not be a symlink: {path.name}")
    if errors:
        return {
            "provided": True,
            "output_dir": str(output),
            "valid": False,
            "artifact_self_consistent": False,
            "execution_origin_attested": False,
            "training_started_by_validator": False,
            "profile": profile.name,
            "accelerator_kind": accelerator_kind,
            "errors": errors,
        }

    try:
        if checkpoint_path.stat().st_size > 100_000_000:
            raise ValueError("best checkpoint exceeds the 100 MB smoke-evidence limit")
        if event_path.stat().st_size > 10_000_000:
            raise ValueError("training event log exceeds the 10 MB evidence limit")
        summary = _safe_json_object(summary_path)
        manifest = _safe_json_object(manifest_path)
        if not legacy_mps:
            runtime_validity = _safe_json_object(output / "runtime_validity.json")
            _validate_json_security(
                runtime_validity,
                label="runtime validity artifact",
            )
        candidate_format = manifest.get(
            "candidate_format",
            "arbitrary_python" if legacy_mps else None,
        )
        if candidate_format not in {"architecture_ir", "arbitrary_python"}:
            raise ValueError("training manifest has an unsupported candidate format")
        if not legacy_mps and candidate_format != "architecture_ir":
            raise ValueError("CUDA smoke candidate format is not Architecture IR")
        candidate_path = (
            graph_path if candidate_format == "architecture_ir" else source_path
        )
        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"missing smoke artifact: {candidate_path.name}"
            )
        if candidate_path.is_symlink():
            raise ValueError(
                f"smoke artifact may not be a symlink: {candidate_path.name}"
            )
        for field, expected in {
            "success": True,
            "scientific": False,
            "hardware_matched": True,
            "unsupported_operation_fallback": False,
            "cleanup_completed": True,
        }.items():
            _exact_bool(summary, field, expected)
        stored_artifact_hash = _sha256_file(candidate_path)
        if candidate_format == "architecture_ir":
            trusted_candidate = root / TRUSTED_CANDIDATE_RELATIVE_PATH
            trusted_ir_validation = validate_ir_candidate_json(
                trusted_candidate.read_text(encoding="utf-8")
            )
            if (
                not trusted_ir_validation.valid
                or trusted_ir_validation.graph is None
            ):
                raise ValueError(
                    "trusted initial candidate is not valid Architecture IR"
                )
            expected_graph_hash = trusted_ir_validation.graph_hash
            stored_ir_validation = validate_ir_candidate_json(
                candidate_path.read_text(encoding="utf-8")
            )
            if not stored_ir_validation.valid:
                raise ValueError("stored candidate graph is not valid Architecture IR")
            if stored_ir_validation.graph_hash != expected_graph_hash:
                raise ValueError(
                    "stored candidate graph differs from the trusted initial candidate"
                )
        else:
            trusted_candidate = (
                root / TRUSTED_LEGACY_PYTHON_CANDIDATE_RELATIVE_PATH
            )
            if not trusted_candidate.is_file() or trusted_candidate.is_symlink():
                raise ValueError("trusted historical Python candidate is unavailable")
            if _sha256_file(trusted_candidate) != stored_artifact_hash:
                raise ValueError(
                    "stored candidate source differs from the trusted initial candidate"
                )
            expected_graph_hash = None
        if not legacy_mps:
            _validate_cuda_summary(
                summary,
                profile=profile,
                candidate_hash=stored_artifact_hash,
                checkpoint_path=checkpoint_path,
                event_path=event_path,
            )
        expected_summary = {
            "profile_name": profile.name,
            "profile_version": profile.version,
            "profile_hash": profile.profile_hash,
            # TrainingResult retains this legacy field name for schema
            # compatibility; the manifest below records the precise IR terms.
            # Native controllers canonicalize the trusted seed before training,
            # while the standalone trainer preserves its original formatting.
            # In both cases identity must bind to the immutable stored artifact;
            # semantic equality to the checked-in seed is verified by graph hash.
            "candidate_source_hash": stored_artifact_hash,
            "dtype": "float32",
            "steps_completed": profile.max_steps,
            "examples_processed": (profile.max_steps * profile.global_batch_size),
        }
        seeds: TrainingSeedBundle | None = None
        dependency_lock_hash: str | None = None
        if legacy_mps:
            expected_summary["device"] = "mps"
        else:
            expected_summary.update(
                {
                    "schema_name": "TrainingResult",
                    "schema_version": "2.0",
                    "accelerator_kind": accelerator_kind,
                }
            )
        for field, expected in expected_summary.items():
            if summary.get(field) != expected:
                raise ValueError(
                    f"training summary {field} does not match {profile.name}"
                )
        if _device_kind(summary.get("device")) != accelerator_kind:
            raise ValueError(
                "training summary device does not match the expected accelerator"
            )
        for field in (
            "best_development_loss",
            "final_training_loss",
            "train_seconds",
        ):
            value = summary.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"training summary {field} must be finite numeric data"
                )

        for field, expected in {
            "allow_cpu_for_tests": False,
            "hardware_matched_scientific_run": False,
        }.items():
            _exact_bool(manifest, field, expected)
        expected_manifest: dict[str, Any] = {
            "candidate_source_hash": expected_summary["candidate_source_hash"],
            "profile_hash": profile.profile_hash,
            "parameter_count_role": "descriptive_metadata_only",
            "isolation_level": "engineering_only_or_scientific_gate_blocked",
        }
        if candidate_format == "architecture_ir":
            expected_manifest.update(
                {
                    "candidate_artifact_hash": stored_artifact_hash,
                    "candidate_format": "architecture_ir",
                    "candidate_graph_hash": expected_graph_hash,
                }
            )
        else:
            # The retained pre-IR v1 records predate these explicit fields. If
            # a later v1 record includes them, they must still bind correctly.
            optional_python_bindings = {
                "candidate_artifact_hash": stored_artifact_hash,
                "candidate_format": "arbitrary_python",
                "candidate_graph_hash": None,
            }
            for field, expected in optional_python_bindings.items():
                if field in manifest and manifest[field] != expected:
                    raise ValueError(
                        f"training manifest {field} does not match the smoke run"
                    )
        if not legacy_mps:
            expected_manifest.update(
                {
                    "schema_name": "TrainingManifest",
                    "schema_version": "2.0",
                    "candidate_path": "candidate_graph.json",
                }
            )
        for field, expected in expected_manifest.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"training manifest {field} does not match the smoke run"
                )
        if "profile" in manifest:
            expected_profile = json.loads(
                json.dumps(profile.to_dict(), allow_nan=False)
            )
            if manifest["profile"] != expected_profile:
                raise ValueError(
                    "training manifest profile differs from the frozen profile"
                )
        runtime = manifest.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("training manifest lacks runtime evidence")
        if legacy_mps:
            _exact_bool(runtime, "mps_built", True)
            _exact_bool(runtime, "mps_available", True)
            if manifest.get("requested_device") != "mps":
                raise ValueError("historical manifest did not request MPS")
            if manifest.get("selected_device") != "mps":
                raise ValueError("historical manifest did not select MPS")
        else:
            if expected_graph_hash is None:
                raise ValueError("CUDA smoke lacks a candidate graph hash")
            seeds, dependency_lock_hash = _validate_cuda_v2_records(
                output=output,
                summary=summary,
                manifest=manifest,
                profile=profile,
                candidate_hash=stored_artifact_hash,
                candidate_graph_hash=expected_graph_hash,
                checkpoint_path=checkpoint_path,
                event_path=event_path,
                require_modal_context=require_modal_context,
            )
        if runtime.get("pytorch_enable_mps_fallback") not in {"", "0"}:
            raise ValueError("training manifest requested MPS fallback")
        decision = manifest.get("containment_decision")
        if not isinstance(decision, dict):
            raise ValueError("training manifest lacks containment decision")
        _exact_bool(decision, "allowed", True)
        _exact_bool(decision, "scientific", False)
        audit = manifest.get("containment_audit")
        if not isinstance(audit, dict):
            raise ValueError("training manifest lacks containment audit")
        if audit.get("visible_credential_names") not in ([], ()):
            raise ValueError(
                "candidate worker observed credential-like environment names"
            )

        if _sha256_file(candidate_path) != stored_artifact_hash:
            raise ValueError("stored candidate artifact changed during validation")
        if _sha256_file(checkpoint_path) != summary.get("checkpoint_sha256"):
            raise ValueError("best checkpoint hash does not match the training summary")
        if Path(str(summary.get("checkpoint_path", ""))).name != checkpoint_path.name:
            raise ValueError("training summary names an unexpected checkpoint")
        if Path(str(summary.get("event_log_path", ""))).name != event_path.name:
            raise ValueError("training summary names an unexpected event log")

        if legacy_mps:
            events = [
                _strict_json_loads(line, label="historical training event")
                for line in event_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(events) != profile.max_steps:
                raise ValueError(
                    "smoke event log does not contain exactly ten optimizer steps"
                )
            if [event.get("optimizer_step") for event in events] != list(
                range(1, profile.max_steps + 1)
            ):
                raise ValueError("smoke optimizer-step sequence is not contiguous")
            if any(
                isinstance(event.get("loss"), bool)
                or not isinstance(event.get("loss"), (int, float))
                or not math.isfinite(event["loss"])
                for event in events
            ):
                raise ValueError("smoke event log contains a non-finite loss")
        else:
            _load_training_events(event_path, profile=profile)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model_state"), dict
        ):
            raise ValueError("best checkpoint lacks a model_state mapping")
        if not legacy_mps:
            if seeds is None or dependency_lock_hash is None:
                raise AssertionError("CUDA manifest validation did not return identity")
            partial_checkpoint = torch.load(
                output / "partial_resume_checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
            latest_checkpoint = torch.load(
                output / "latest_resume_checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(partial_checkpoint, dict) or not isinstance(
                latest_checkpoint, dict
            ):
                raise ValueError("CUDA resume checkpoints must be mappings")
            _validate_cuda_checkpoints(
                output=output,
                best=checkpoint,
                partial=partial_checkpoint,
                latest=latest_checkpoint,
                profile=profile,
                candidate_hash=stored_artifact_hash,
                summary=summary,
                seeds=seeds,
                dependency_lock_hash=dependency_lock_hash,
            )
        else:
            for field, expected in {
                "checkpoint_kind": "best_evaluation_weights_v1",
                "candidate_source_hash": stored_artifact_hash,
                "profile_hash": profile.profile_hash,
                "global_step": profile.max_steps,
                "examples_processed": (
                    profile.max_steps * profile.global_batch_size
                ),
            }.items():
                if field in checkpoint and checkpoint[field] != expected:
                    raise ValueError(
                        f"historical checkpoint {field} does not match the run binding"
                    )
        if candidate_format == "architecture_ir":
            interpreted = load_and_build_ir_candidate(
                candidate_path,
                int(summary["initialization_seed"]),
            )
            initial_model = interpreted.model
        else:
            # Never import the artifact copy. Its byte identity is first bound
            # to the checked-in trusted seed, and only that trusted file runs.
            initial_model = build_candidate_artifact(
                trusted_candidate,
                seed=int(summary["initialization_seed"]),
            ).model
        initial_state = initial_model.state_dict()
        trained_state = checkpoint["model_state"]
        if set(initial_state) != set(trained_state):
            raise ValueError(
                "checkpoint model state does not match the trusted architecture"
            )
        for name, initial_value in initial_state.items():
            trained_value = trained_state[name]
            if not isinstance(trained_value, torch.Tensor):
                raise ValueError(f"checkpoint state {name} is not a tensor")
            if trained_value.shape != initial_value.shape:
                raise ValueError(f"checkpoint state {name} has the wrong shape")
            if trained_value.dtype != initial_value.dtype:
                raise ValueError(f"checkpoint state {name} has the wrong dtype")
        parameters_changed = any(
            not torch.equal(initial_state[name], trained_state[name])
            for name in initial_state
        )
        if not parameters_changed:
            raise ValueError("smoke checkpoint is identical to fresh initialization")
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        errors.append(f"{type(error).__name__}: {error}")
        parameters_changed = False

    valid = not errors
    return {
        "provided": True,
        "output_dir": str(output),
        "valid": valid,
        "artifact_self_consistent": valid,
        "execution_origin_attested": False,
        "training_started_by_validator": False,
        "profile": profile.name,
        "accelerator_kind": accelerator_kind,
        "scientific": False,
        "claim_scope": "self_authored_artifact_consistency_only",
        "parameters_changed": parameters_changed,
        "errors": errors,
    }


def validate_existing_mps_smoke(
    training_output_dir: str | Path | None,
    *,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Read historical smoke_train_v1 evidence without rewriting it."""

    return _validate_existing_smoke(
        training_output_dir,
        project_root=project_root,
        profile=SMOKE_TRAIN_V1,
        accelerator_kind="mps",
        require_modal_context=False,
    )


def validate_existing_cuda_smoke(
    training_output_dir: str | Path | None,
    *,
    project_root: str | Path = ROOT,
    require_modal_context: bool = True,
) -> dict[str, Any]:
    """Read active smoke_train_cuda_v2 evidence without rewriting it."""

    return _validate_existing_smoke(
        training_output_dir,
        project_root=project_root,
        profile=SMOKE_TRAIN_CUDA_V2,
        accelerator_kind="cuda",
        require_modal_context=require_modal_context,
    )


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_credential_field(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    return normalized in _CREDENTIAL_FIELD_NAMES or normalized.endswith(
        ("_api_key", "_password", "_secret", "_access_token", "_refresh_token")
    )


def _credential_field_paths(value: object, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if _is_credential_field(key_text):
                found.append(child_path)
            found.extend(_credential_field_paths(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_credential_field_paths(child, path=f"{path}[{index}]"))
    return found


def _is_path_field(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    return normalized in {
        "candidate",
        "path",
        "directory",
        "dir",
        "root",
        "ledger",
        "location",
    } or normalized.endswith(
        (
            "_path",
            "_directory",
            "_dir",
            "_root",
            "_ledger",
            "_location",
        )
    )


def _absolute_path_field_paths(value: object, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if (
                _is_path_field(key_text)
                and isinstance(child, str)
                and (
                    Path(child).is_absolute()
                    or PureWindowsPath(child).is_absolute()
                    or bool(PureWindowsPath(child).drive)
                    or child.lower().startswith("file://")
                )
            ):
                found.append(child_path)
            found.extend(_absolute_path_field_paths(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _absolute_path_field_paths(child, path=f"{path}[{index}]")
            )
    return found


def _validate_no_credential_fields(
    run_directory: Path,
    artifact_paths: set[str],
) -> None:
    for relative_path in sorted(artifact_paths):
        relative = Path(relative_path)
        for component in relative.parts:
            logical_name = Path(component).stem
            if _is_credential_field(logical_name):
                raise ValueError(
                    f"canary artifact has a credential-bearing path: {relative_path}"
                )
        path = run_directory / relative
        if relative.suffix == ".json":
            payloads = [json.loads(path.read_text(encoding="utf-8"))]
        elif relative.suffix == ".jsonl":
            payloads = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            continue
        for payload in payloads:
            forbidden = _credential_field_paths(payload)
            if forbidden:
                raise ValueError(
                    "canary artifact contains credential fields in "
                    f"{relative_path}: {', '.join(forbidden[:5])}"
                )
            absolute_paths = _absolute_path_field_paths(payload)
            if absolute_paths:
                raise ValueError(
                    "canary artifact contains executor-absolute path fields in "
                    f"{relative_path}: {', '.join(absolute_paths[:5])}"
                )


def _discover_modal_canary_runs(
    download_root_or_prefix: str | Path,
) -> tuple[Path, str | None, dict[str, Path]]:
    """Resolve either a four-run download directory or a non-existent prefix path."""

    raw = Path(download_root_or_prefix).expanduser()
    if raw.is_symlink():
        raise ValueError("Modal canary download root/prefix may not be a symlink")
    if raw.exists():
        if not raw.is_dir():
            raise ValueError("Modal canary download root must be a directory")
        download_root = raw.resolve()
        entries = tuple(download_root.iterdir())
        if len(entries) != len(CANARY_ORDER):
            raise ValueError(
                "Modal canary download root must contain exactly four runs"
            )
        if any(entry.is_symlink() or not entry.is_dir() for entry in entries):
            raise ValueError(
                "Modal canary download root contains a non-directory or symlink"
            )
        prefixes: set[str] = set()
        runs: dict[str, Path] = {}
        for entry in entries:
            matches = [
                harness
                for harness, suffix in _MODAL_CANARY_SUFFIXES.items()
                if entry.name.endswith(f"-{suffix}")
            ]
            if len(matches) != 1:
                raise ValueError(f"unexpected Modal canary run suffix: {entry.name}")
            harness = matches[0]
            suffix = _MODAL_CANARY_SUFFIXES[harness]
            prefix = entry.name[: -len(suffix) - 1]
            if not prefix or harness in runs:
                raise ValueError(
                    "Modal canary run roster is duplicated or lacks a prefix"
                )
            prefixes.add(prefix)
            runs[harness] = entry.resolve()
        if set(runs) != set(CANARY_ORDER):
            raise ValueError(
                "Modal canary runs do not form one complete frozen roster"
            )
        # A directory bundle may deliberately combine successful attempts after
        # one paid harness was retried.  Per-run manifests and execution
        # contexts remain authoritative, while the bundle checks below still
        # require four unique calls and one shared image/source identity.
        common_prefix = next(iter(prefixes)) if len(prefixes) == 1 else None
        return download_root, common_prefix, runs

    download_root = raw.parent.resolve()
    prefix = raw.name
    if not prefix or not download_root.is_dir() or download_root.is_symlink():
        raise ValueError("Modal canary prefix parent is missing or unsafe")
    expected = {
        harness: download_root / f"{prefix}-{suffix}"
        for harness, suffix in _MODAL_CANARY_SUFFIXES.items()
    }
    related = {
        entry.name
        for entry in download_root.iterdir()
        if entry.name.startswith(f"{prefix}-")
    }
    expected_names = {path.name for path in expected.values()}
    if related != expected_names:
        raise ValueError(
            "Modal canary prefix does not resolve to exactly four frozen runs"
        )
    if any(path.is_symlink() or not path.is_dir() for path in expected.values()):
        raise ValueError("Modal canary prefix resolves to a non-directory or symlink")
    return (
        download_root,
        prefix,
        {harness: path.resolve() for harness, path in expected.items()},
    )


def _selector_project_root(project_root: str | Path) -> Path:
    raw = Path(project_root).expanduser()
    if raw.is_symlink():
        raise ValueError("selector project root may not be a symlink")
    root = raw.resolve()
    if not root.is_dir():
        raise ValueError("selector project root must be a directory")
    return root


def _reject_project_path_symlinks(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("selector path escapes the project root") from error
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("selector paths may not contain symlinks")


def _selector_run_directory(root: Path, run_id: str) -> tuple[str, Path]:
    selected_run_id = validate_run_id(run_id)
    logical = (_MODAL_CANARY_DOWNLOAD_ROOT / selected_run_id).as_posix()
    directory = root.joinpath(*PurePosixPath(logical).parts)
    _reject_project_path_symlinks(root, directory)
    if not directory.is_dir():
        raise ValueError(f"selected Modal canary run is missing: {logical}")
    if directory.resolve() != directory:
        raise ValueError("selected Modal canary run path is not canonical")
    return logical, directory


def _selector_identity_from_payload(
    payload: dict[str, Any],
) -> ModalLiveCohortIdentity:
    try:
        return ModalLiveCohortIdentity(
            source_tree_sha256=payload["source_tree_sha256"],
            image_source_sha256=payload["image_source_sha256"],
            cohort_id=payload["cohort_id"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Modal canary selector cohort identity is invalid") from error


def _selector_output_path(
    root: Path,
    identity: ModalLiveCohortIdentity,
    selector_id: str,
) -> Path:
    selected_id = validate_run_id(selector_id)
    relative = (
        modal_live_cohort_root(identity)
        / "provider_canary_selection"
        / selected_id
        / "canary_run_selector.json"
    )
    output = root.joinpath(*relative.parts)
    _reject_project_path_symlinks(root, output)
    return output


def _load_modal_canary_selector(
    selector_path: str | Path,
    *,
    project_root: str | Path = ROOT,
    expected_identity: ModalLiveCohortIdentity | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    root = _selector_project_root(project_root)
    raw_selector = Path(selector_path).expanduser()
    path = raw_selector if raw_selector.is_absolute() else root / raw_selector
    _reject_project_path_symlinks(root, path)
    if not path.is_file():
        raise ValueError("Modal canary selector must be a regular file")
    payload = _safe_json_object(path)
    _require_exact_fields(
        payload,
        _MODAL_CANARY_SELECTOR_FIELDS,
        label="Modal canary selector",
    )
    if (
        payload.get("schema_name") != _MODAL_CANARY_SELECTOR_SCHEMA_NAME
        or payload.get("schema_version") != _MODAL_CANARY_SELECTOR_SCHEMA_VERSION
    ):
        raise ValueError("Modal canary selector has the wrong schema identity")
    selector_id = payload.get("selector_id")
    if not isinstance(selector_id, str):
        raise ValueError("Modal canary selector ID must be text")
    validate_run_id(selector_id)
    identity = _selector_identity_from_payload(payload)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Modal canary selector differs from the selected cohort")
    if path.resolve() != _selector_output_path(root, identity, selector_id):
        raise ValueError("Modal canary selector is not at its canonical path")
    if payload.get("harness_order") != list(CANARY_ORDER):
        raise ValueError("Modal canary selector harness order changed")
    runs = payload.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(CANARY_ORDER):
        raise ValueError("Modal canary selector run mapping is not exact")

    directories: dict[str, Path] = {}
    observed_values: dict[str, set[str]] = {
        "run_id": set(),
        "download_path": set(),
        "raw_artifact_manifest_sha256": set(),
        "execution_context_sha256": set(),
    }
    image_source_hashes: set[str] = set()
    modal_image_ids: set[str] = set()
    for harness in CANARY_ORDER:
        entry = runs.get(harness)
        if not isinstance(entry, dict):
            raise ValueError(f"Modal canary selector entry is missing: {harness}")
        _require_exact_fields(
            entry,
            _MODAL_CANARY_SELECTOR_RUN_FIELDS,
            label=f"Modal canary selector {harness} entry",
        )
        if entry.get("harness") != harness:
            raise ValueError("Modal canary selector contains a harness substitution")
        run_id = entry.get("run_id")
        if not isinstance(run_id, str):
            raise ValueError("Modal canary selector run ID must be text")
        logical, run_directory = _selector_run_directory(root, run_id)
        expected_manifest_path = f"{logical}/artifact_manifest.json"
        expected_context_path = f"{logical}/execution_context.json"
        if (
            entry.get("download_path") != logical
            or entry.get("artifact_manifest_path") != expected_manifest_path
            or entry.get("execution_context_path") != expected_context_path
        ):
            raise ValueError("Modal canary selector contains a noncanonical path")
        manifest_path = run_directory / "artifact_manifest.json"
        context_path = run_directory / "execution_context.json"
        for evidence_path in (manifest_path, context_path):
            _reject_project_path_symlinks(root, evidence_path)
            if not evidence_path.is_file():
                raise ValueError("Modal canary selector evidence file is missing")
        for field, evidence_path in (
            ("raw_artifact_manifest_sha256", manifest_path),
            ("execution_context_sha256", context_path),
        ):
            digest = entry.get(field)
            if not _is_lower_sha256(digest) or digest != _sha256_file(evidence_path):
                raise ValueError(f"Modal canary selector {field} does not match")
        artifact_manifest = load_artifact_manifest(manifest_path)
        context = ExecutionContextV1.from_dict(_safe_json_object(context_path))
        if (
            artifact_manifest.run_id != run_id
            or context.run_id != run_id
            or context.function_name != f"canary_{harness}"
            or artifact_manifest.image_source_sha256
            != entry.get("image_source_sha256")
            or context.image_source_sha256 != entry.get("image_source_sha256")
            or context.modal_image_id != entry.get("modal_image_id")
        ):
            raise ValueError("Modal canary selector run/source/image binding differs")
        for field in observed_values:
            value = entry.get(field)
            if not isinstance(value, str):
                raise ValueError(f"Modal canary selector {field} must be text")
            observed_values[field].add(value)
        image_source_hashes.add(str(entry["image_source_sha256"]))
        modal_image_ids.add(str(entry["modal_image_id"]))
        directories[harness] = run_directory

    if any(len(values) != len(CANARY_ORDER) for values in observed_values.values()):
        raise ValueError("Modal canary selector contains duplicate run evidence")
    if len(image_source_hashes) != 1 or len(modal_image_ids) != 1:
        raise ValueError("Modal canary selector mixes source or image identities")
    if image_source_hashes != {identity.image_source_sha256}:
        raise ValueError("Modal canary selector image differs from its cohort identity")
    return payload, directories


def load_modal_canary_selector(
    selector_path: str | Path,
    *,
    project_root: str | Path = ROOT,
    expected_identity: ModalLiveCohortIdentity | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Read and validate one immutable accepted-canary selector binding."""

    return _load_modal_canary_selector(
        selector_path,
        project_root=project_root,
        expected_identity=expected_identity,
    )


def create_modal_canary_selector(
    *,
    selector_id: str,
    run_ids: dict[str, str],
    identity: ModalLiveCohortIdentity,
    project_root: str | Path = ROOT,
) -> tuple[Path, dict[str, Any]]:
    """Create one immutable accepted-run selector for aggregate/recovery mixes."""

    root = _selector_project_root(project_root)
    if tuple(run_ids) != tuple(CANARY_ORDER):
        raise ValueError(
            "selector run IDs must cover CANARY_ORDER exactly and in order"
        )
    if not isinstance(identity, ModalLiveCohortIdentity):
        raise TypeError("selector identity must be ModalLiveCohortIdentity")
    selected_id = validate_run_id(selector_id)
    entries: dict[str, dict[str, str]] = {}
    run_reports: list[dict[str, Any]] = []
    for harness in CANARY_ORDER:
        run_id = run_ids[harness]
        logical, run_directory = _selector_run_directory(root, run_id)
        report = _validate_modal_canary_run(run_directory, harness=harness)
        run_reports.append(report)
        manifest_path = run_directory / "artifact_manifest.json"
        context_path = run_directory / "execution_context.json"
        entries[harness] = {
            "harness": harness,
            "run_id": run_id,
            "download_path": logical,
            "artifact_manifest_path": f"{logical}/artifact_manifest.json",
            "raw_artifact_manifest_sha256": _sha256_file(manifest_path),
            "execution_context_path": f"{logical}/execution_context.json",
            "execution_context_sha256": _sha256_file(context_path),
            "image_source_sha256": report["image_source_sha256"],
            "modal_image_id": report["modal_image_id"],
        }
    if len({report["run_id"] for report in run_reports}) != len(CANARY_ORDER):
        raise ValueError("selected Modal canary run IDs are not unique")
    if len({report["modal_call_id"] for report in run_reports}) != len(CANARY_ORDER):
        raise ValueError("selected Modal canary call IDs are not unique")
    if len({report["image_source_sha256"] for report in run_reports}) != 1:
        raise ValueError("selected Modal canaries do not share one source identity")
    if {report["image_source_sha256"] for report in run_reports} != {
        identity.image_source_sha256
    }:
        raise ValueError("selected Modal canaries differ from the cohort image")
    if len({report["modal_image_id"] for report in run_reports}) != 1:
        raise ValueError("selected Modal canaries do not share one image identity")
    payload: dict[str, Any] = {
        "schema_name": _MODAL_CANARY_SELECTOR_SCHEMA_NAME,
        "schema_version": _MODAL_CANARY_SELECTOR_SCHEMA_VERSION,
        "source_tree_sha256": identity.source_tree_sha256,
        "image_source_sha256": identity.image_source_sha256,
        "cohort_id": identity.cohort_id,
        "selector_id": selected_id,
        "harness_order": list(CANARY_ORDER),
        "runs": entries,
    }
    output = _selector_output_path(root, identity, selected_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_project_path_symlinks(root, output)
    create_json_exclusive(output, payload)
    load_modal_canary_selector(
        output,
        project_root=root,
        expected_identity=identity,
    )
    return output, payload


def _parse_modal_canary_run_selections(values: list[str]) -> dict[str, str]:
    run_ids: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str) or value.count("=") != 1:
            raise ValueError("--modal-canary-run must be HARNESS=RUN_ID")
        harness, run_id = value.split("=", 1)
        if harness not in CANARY_ORDER or harness in run_ids:
            raise ValueError("--modal-canary-run harness is unknown or duplicated")
        run_ids[harness] = validate_run_id(run_id)
    if set(run_ids) != set(CANARY_ORDER):
        raise ValueError("--modal-canary-run must name each CANARY_ORDER harness once")
    return {harness: run_ids[harness] for harness in CANARY_ORDER}


def _validate_modal_canary_run(
    run_directory: Path,
    *,
    harness: str,
) -> dict[str, Any]:
    expected_run_id = run_directory.name
    observed_outer_roster = {entry.name for entry in run_directory.iterdir()}
    if observed_outer_roster != _DOWNLOADED_CANARY_OUTER_ROSTER:
        missing = sorted(_DOWNLOADED_CANARY_OUTER_ROSTER - observed_outer_roster)
        extra = sorted(observed_outer_roster - _DOWNLOADED_CANARY_OUTER_ROSTER)
        raise ValueError(
            "downloaded canary outer roster differs from the frozen contract "
            f"(missing={missing}, extra={extra})"
        )
    for entry in run_directory.iterdir():
        if entry.is_symlink():
            raise ValueError("downloaded canary outer roster contains a symlink")
        if entry.name == "controller":
            if not entry.is_dir():
                raise ValueError("downloaded canary controller is not a directory")
        elif not entry.is_file():
            raise ValueError("downloaded canary outer evidence is not a regular file")
    manifest_path = run_directory / "artifact_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("downloaded canary lacks a safe artifact_manifest.json")
    artifact_manifest = load_artifact_manifest(manifest_path)
    if artifact_manifest.run_id != expected_run_id:
        raise ValueError("artifact manifest run ID differs from its directory")
    verification = verify_artifact_manifest(run_directory, artifact_manifest)
    artifact_paths = {item.relative_path for item in artifact_manifest.files}
    actual_downloaded_paths = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_downloaded_paths != artifact_paths:
        raise ValueError("downloaded canary contains an unmanifested artifact file")
    summary_relative_path = (
        "controller/run_summary.json"
        if harness in _NATIVE_CANARY_HARNESSES
        else "controller/run_result.json"
    )
    required_paths = {
        "execution_context.json",
        "image_source_manifest.json",
        "remote_action_result.json",
        "controller/run_manifest.json",
        f"controller/{PROVIDER_ATTEMPT_LEDGER_FILENAME}",
        summary_relative_path,
    }
    missing = required_paths - artifact_paths
    if missing:
        raise ValueError(
            "downloaded canary lacks required artifacts: " + ", ".join(sorted(missing))
        )
    controller_directory = run_directory / "controller"
    if controller_directory.is_symlink() or not controller_directory.is_dir():
        raise ValueError("downloaded canary controller is missing or unsafe")
    observed_controller_roster = {
        entry.name for entry in controller_directory.iterdir()
    }
    if observed_controller_roster != _PRIVATE_CANARY_TOP_LEVEL_ROSTERS[harness]:
        missing = sorted(
            _PRIVATE_CANARY_TOP_LEVEL_ROSTERS[harness] - observed_controller_roster
        )
        extra = sorted(
            observed_controller_roster - _PRIVATE_CANARY_TOP_LEVEL_ROSTERS[harness]
        )
        raise ValueError(
            "downloaded canary controller roster differs from the frozen contract "
            f"(missing={missing}, extra={extra})"
        )
    expected_controller_files = _PRIVATE_CANARY_TOP_LEVEL_FILES[harness]
    for entry in controller_directory.iterdir():
        if entry.is_symlink():
            raise ValueError("downloaded canary controller roster contains a symlink")
        if entry.name in expected_controller_files:
            if not entry.is_file():
                raise ValueError("downloaded canary controller file has the wrong type")
        elif not entry.is_dir():
            raise ValueError("downloaded canary controller directory has the wrong type")
    alternate_summary = (
        "controller/run_result.json"
        if harness in _NATIVE_CANARY_HARNESSES
        else "controller/run_summary.json"
    )
    if alternate_summary in artifact_paths:
        raise ValueError("downloaded canary contains a substituted controller summary")

    image_source = _safe_json_object(run_directory / "image_source_manifest.json")
    if image_source.get("schema_name") != "ModalImageSourceManifest":
        raise ValueError("canary image source has the wrong schema")
    if image_source.get("schema_version") != "1.0":
        raise ValueError("canary image source has the wrong schema version")
    image_source_sha256 = canonical_sha256(image_source)
    if image_source_sha256 != artifact_manifest.image_source_sha256:
        raise ValueError("artifact manifest is not bound to its image source manifest")

    context_payload = _safe_json_object(run_directory / "execution_context.json")
    context = ExecutionContextV1.from_dict(context_payload)
    expected_function = f"canary_{harness}"
    if context.execution_backend != "modal":
        raise ValueError("canary execution context is not Modal")
    if context.run_id != expected_run_id:
        raise ValueError("canary execution context run ID differs from its directory")
    if context.app_name != APP_NAME or context.function_name != expected_function:
        raise ValueError("canary execution context names the wrong harness function")
    if context.modal_call_id is None or context.modal_image_id is None:
        raise ValueError("canary execution context lacks a Modal call or image ID")
    if context.image_source_sha256 != image_source_sha256:
        raise ValueError("canary execution context is not bound to its image source")
    if context.artifact_uri != volume_artifact_uri(expected_run_id):
        raise ValueError("canary execution context is not bound to its artifact URI")

    remote_result = _safe_json_object(run_directory / "remote_action_result.json")
    expected_remote_fields = {
        "success",
        "mode",
        "harness",
        "returncode",
        "stdout_sha256",
        "stdout_size_bytes",
        "stderr_sha256",
        "stderr_size_bytes",
    }
    if set(remote_result) != expected_remote_fields:
        raise ValueError("remote action result fields differ from the canary contract")
    _exact_bool(remote_result, "success", True)
    if remote_result.get("mode") != "one_opportunity_engineering_canary":
        raise ValueError("remote action result has the wrong canary mode")
    if remote_result.get("harness") != harness:
        raise ValueError("remote action result names a substituted harness")
    if _nonnegative_integer(remote_result, "returncode") != 0:
        raise ValueError("remote canary subprocess was not successful")
    for field in ("stdout_sha256", "stderr_sha256"):
        if not _is_lower_sha256(remote_result.get(field)):
            raise ValueError(f"remote action result {field} is invalid")
    for field in ("stdout_size_bytes", "stderr_size_bytes"):
        _nonnegative_integer(remote_result, field)

    _validate_no_credential_fields(run_directory, artifact_paths)
    controller_manifest = _safe_json_object(
        run_directory / "controller" / "run_manifest.json"
    )
    controller_summary = _safe_json_object(run_directory / summary_relative_path)
    _require_exact_fields(
        controller_manifest,
        _CONTROLLER_MANIFEST_FIELDS[harness],
        label=f"downloaded {harness} ControllerRunManifest",
    )
    _require_exact_fields(
        controller_summary,
        _CONTROLLER_SUMMARY_FIELDS[harness],
        label=f"downloaded {harness} controller summary",
    )
    if harness in _NATIVE_CANARY_HARNESSES:
        downloaded_lineage = _load_private_jsonl(
            controller_directory / "lineage.jsonl",
            label=f"downloaded {harness} native lineage",
        )
        if len(downloaded_lineage) != 2:
            raise ValueError("downloaded native canary lineage is not seed plus proposal")
        for record in downloaded_lineage:
            _require_exact_fields(
                record,
                _NATIVE_LINEAGE_FIELDS[harness],
                label=f"downloaded {harness} native lineage record",
            )
    else:
        downloaded_trace = _load_private_jsonl(
            controller_directory / "evolution_trace.jsonl",
            label=f"downloaded {harness} OpenEvolve trace",
        )
        if len(downloaded_trace) != 1:
            raise ValueError("downloaded OpenEvolve trace is not one opportunity")
        downloaded_trace_record = downloaded_trace[0]
        _require_exact_fields(
            downloaded_trace_record,
            _OPENEVOLVE_TRACE_FIELDS,
            label=f"downloaded {harness} OpenEvolve trace",
        )
        for field, expected_fields in (
            ("metadata", _OPENEVOLVE_TRACE_METADATA_FIELDS),
            ("prompt", _OPENEVOLVE_TRACE_PROMPT_FIELDS),
            ("artifacts", _OPENEVOLVE_TRACE_ARTIFACT_FIELDS),
        ):
            nested = downloaded_trace_record[field]
            if not isinstance(nested, dict):
                raise ValueError(f"downloaded OpenEvolve trace {field} is not an object")
            _require_exact_fields(
                nested,
                expected_fields,
                label=f"downloaded {harness} OpenEvolve trace {field}",
            )
        downloaded_checkpoint_metadata_path = (
            controller_directory / "checkpoints" / "checkpoint_1" / "metadata.json"
        )
        if (
            downloaded_checkpoint_metadata_path.is_symlink()
            or not downloaded_checkpoint_metadata_path.is_file()
        ):
            raise ValueError("downloaded OpenEvolve checkpoint metadata is missing or unsafe")
        downloaded_checkpoint_metadata = _safe_json_object(
            downloaded_checkpoint_metadata_path
        )
        _require_exact_fields(
            downloaded_checkpoint_metadata,
            _OPENEVOLVE_CHECKPOINT_METADATA_FIELDS,
            label=f"downloaded {harness} OpenEvolve checkpoint metadata",
        )
    if controller_manifest.get("schema_name") != "ControllerRunManifest":
        raise ValueError("controller manifest has the wrong schema name")
    if controller_manifest.get("schema_version") != "2.0":
        raise ValueError("controller manifest has the wrong schema version")
    expected_summary_schema = (
        "ControllerRunSummary"
        if harness in _NATIVE_CANARY_HARNESSES
        else "ControllerRunResult"
    )
    if controller_summary.get("schema_name") != expected_summary_schema:
        raise ValueError("controller summary has the wrong schema name")
    if controller_summary.get("schema_version") != "2.0":
        raise ValueError("controller summary has the wrong schema version")
    expected_condition = harness
    if controller_manifest.get("condition") != expected_condition:
        raise ValueError("controller manifest names a substituted harness")
    controller_run_id = controller_manifest.get("run_id")
    if not isinstance(controller_run_id, str) or not controller_run_id:
        raise ValueError("controller manifest lacks its run ID")
    if controller_summary.get("run_id") != controller_run_id:
        raise ValueError("controller manifest and summary run IDs differ")
    if controller_summary.get("condition") != expected_condition:
        raise ValueError("controller summary names a substituted harness")
    for field, expected in {
        "candidate_budget": 2,
        "mutation_budget": 1,
        "candidate_training_budget": 2,
    }.items():
        if _nonnegative_integer(controller_manifest, field) != expected:
            raise ValueError(f"controller manifest {field} is not one opportunity")
    _exact_bool(
        controller_manifest,
        "authoritative_scientific_evidence",
        False,
    )
    _validate_modal_canary_generator(controller_manifest.get("generator"))
    if controller_manifest.get("provider_attempt_ledger") != (
        PROVIDER_ATTEMPT_LEDGER_FILENAME
    ):
        raise ValueError("controller manifest names the wrong provider ledger")
    if controller_manifest.get("provider_attempt_schema") != PROVIDER_ATTEMPT_SCHEMA:
        raise ValueError("controller manifest names the wrong provider schema")
    if _nonnegative_integer(controller_manifest, "maximum_provider_attempts") != 1:
        raise ValueError("controller manifest permits more than one provider attempt")
    provider_attempt, provider_totals = _validate_modal_canary_provider_attempt(
        run_directory / "controller" / PROVIDER_ATTEMPT_LEDGER_FILENAME,
        harness=harness,
        action_run_id=expected_run_id,
        controller_run_id=controller_run_id,
        modal_call_id=context.modal_call_id,
    )
    training = controller_manifest.get("training")
    if not isinstance(training, dict):
        raise ValueError("controller manifest lacks its training contract")
    for field, expected in {
        "profile": SMOKE_TRAIN_CUDA_V2.name,
        "profile_version": SMOKE_TRAIN_CUDA_V2.version,
        "profile_hash": SMOKE_TRAIN_CUDA_V2.profile_hash,
        "device": "cuda",
        "allow_cpu_for_tests": False,
    }.items():
        observed = training.get(field)
        if field == "profile_version":
            observed = str(observed)
        if observed != expected:
            raise ValueError(f"controller manifest training {field} is invalid")
    evaluation = controller_manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("controller manifest lacks its evaluation contract")
    if evaluation.get("profile") != "smoke_eval_v1":
        raise ValueError("controller canary did not use smoke_eval_v1")
    if _nonnegative_integer(evaluation, "case_count") != 24:
        raise ValueError("controller canary did not use 24 smoke cases")
    _exact_bool(evaluation, "scientific", False)

    if harness in _NATIVE_CANARY_HARNESSES:
        if controller_manifest.get("run_mode") != "engineering_pilot":
            raise ValueError("native controller was not an engineering pilot")
        _exact_bool(controller_manifest, "exploratory_only", True)
        if (
            _nonnegative_integer(controller_summary, "proposal_opportunities_requested")
            != 1
        ):
            raise ValueError("native controller requested more than one opportunity")
        if (
            _nonnegative_integer(controller_summary, "proposal_opportunities_terminal")
            != 1
        ):
            raise ValueError(
                "native controller did not terminally account for one opportunity"
            )
    else:
        _exact_bool(controller_manifest, "engineering_pilot", True)
        if _nonnegative_integer(controller_manifest, "proposal_opportunities") != 1:
            raise ValueError(
                "OpenEvolve controller requested more than one opportunity"
            )
        _exact_bool(controller_summary, "completed", True)
        _exact_bool(controller_summary, "engineering_pilot", True)
        _exact_bool(
            controller_summary,
            "authoritative_scientific_evidence",
            False,
        )
        if (
            _nonnegative_integer(controller_summary, "proposal_opportunities_requested")
            != 1
        ):
            raise ValueError("OpenEvolve summary requested more than one opportunity")
        if (
            _nonnegative_integer(controller_summary, "proposal_opportunities_completed")
            != 1
        ):
            raise ValueError("OpenEvolve summary did not complete one opportunity")
        if controller_summary.get("proposal_accounting_errors") != []:
            raise ValueError("OpenEvolve summary contains proposal accounting errors")
        if controller_summary.get("failure_stage") != "":
            raise ValueError("OpenEvolve summary contains a failure stage")

    return {
        "harness": harness,
        "run_id": expected_run_id,
        "controller_run_id": controller_run_id,
        "artifact_manifest_sha256": verification["manifest_sha256"],
        "artifact_file_count": verification["file_count"],
        "modal_call_id": context.modal_call_id,
        "modal_image_id": context.modal_image_id,
        "image_source_sha256": image_source_sha256,
        "artifact_uri": context.artifact_uri,
        "profile": SMOKE_TRAIN_CUDA_V2.name,
        "proposal_opportunities": 1,
        "provider_attempt_count": provider_totals["attempt_count"],
        "provider_response_id": provider_attempt.provider_response_id,
        "provider_request_id": provider_attempt.provider_request_id,
        "provider_input_tokens": provider_totals["input_tokens"],
        "provider_output_tokens": provider_totals["output_tokens"],
        "provider_total_tokens": provider_totals["total_tokens"],
        "valid": True,
        "errors": [],
    }


def validate_downloaded_modal_canaries(
    modal_canary_download_root: str | Path | None,
    *,
    modal_canary_selector: str | Path | None = None,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Validate four downloaded canaries locally without any remote calls."""

    if modal_canary_download_root is not None and modal_canary_selector is not None:
        return {
            "provided": True,
            "valid": False,
            "all_four_canaries_validated": False,
            "remote_calls_started_by_validator": 0,
            "provider_calls_started_by_validator": 0,
            "training_runs_started_by_validator": 0,
            "runs": [],
            "errors": ["provide a Modal canary root/prefix or selector, not both"],
        }
    if modal_canary_download_root is None and modal_canary_selector is None:
        return {
            "provided": False,
            "valid": False,
            "all_four_canaries_validated": False,
            "remote_calls_started_by_validator": 0,
            "provider_calls_started_by_validator": 0,
            "training_runs_started_by_validator": 0,
            "runs": [],
            "errors": [
                "no downloaded Modal canary root/prefix or selector was provided"
            ],
        }
    selector_payload: dict[str, Any] | None = None
    selector_file: Path | None = None
    try:
        if modal_canary_selector is not None:
            root = _selector_project_root(project_root)
            selector_file = Path(modal_canary_selector).expanduser()
            if not selector_file.is_absolute():
                selector_file = root / selector_file
            selector_payload, run_directories = load_modal_canary_selector(
                selector_file,
                project_root=root,
            )
            download_root = root.joinpath(*_MODAL_CANARY_DOWNLOAD_ROOT.parts)
            prefix = None
        else:
            if modal_canary_download_root is None:  # pragma: no cover - guarded above
                raise RuntimeError("unreachable Modal canary selection state")
            download_root, prefix, run_directories = _discover_modal_canary_runs(
                modal_canary_download_root
            )
    except (
        ArtifactIntegrityError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        supplied = (
            modal_canary_selector
            if modal_canary_selector is not None
            else modal_canary_download_root
        )
        return {
            "provided": True,
            "download_root": str(Path(supplied).expanduser()),
            "valid": False,
            "all_four_canaries_validated": False,
            "remote_calls_started_by_validator": 0,
            "provider_calls_started_by_validator": 0,
            "training_runs_started_by_validator": 0,
            "runs": [],
            "errors": [f"{type(error).__name__}: {error}"],
        }

    run_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for harness in CANARY_ORDER:
        try:
            report = _validate_modal_canary_run(
                run_directories[harness],
                harness=harness,
            )
        except (
            ArtifactIntegrityError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            message = f"{harness}: {type(error).__name__}: {error}"
            report = {
                "harness": harness,
                "run_id": run_directories[harness].name,
                "valid": False,
                "errors": [message],
            }
            errors.append(message)
        run_reports.append(report)

    successful = [report for report in run_reports if report["valid"]]
    if len(successful) == len(CANARY_ORDER):
        bundle_checks = {
            "Modal call IDs": {report["modal_call_id"] for report in successful},
            "controller run IDs": {
                report["controller_run_id"] for report in successful
            },
            "provider response IDs": {
                report["provider_response_id"] for report in successful
            },
            "provider request IDs": {
                report["provider_request_id"] for report in successful
            },
        }
        for label, values in bundle_checks.items():
            if len(values) != len(CANARY_ORDER):
                errors.append(f"bundle does not contain four unique {label}")
        for label, values in {
            "image source digests": {
                report["image_source_sha256"] for report in successful
            },
            "Modal image IDs": {report["modal_image_id"] for report in successful},
        }.items():
            if len(values) != 1:
                errors.append(f"bundle does not share one frozen {label}")

    valid = len(successful) == len(CANARY_ORDER) and not errors
    provider_input_tokens = sum(
        int(report.get("provider_input_tokens", 0)) for report in successful
    )
    provider_output_tokens = sum(
        int(report.get("provider_output_tokens", 0)) for report in successful
    )
    provider_total_tokens = sum(
        int(report.get("provider_total_tokens", 0)) for report in successful
    )
    if provider_total_tokens != provider_input_tokens + provider_output_tokens:
        errors.append("bundle provider token totals do not reconcile")
        valid = False
    return {
        "provided": True,
        "download_root": str(download_root),
        "run_id_prefix": prefix,
        "selection_mode": (
            "exact_create_only_selector"
            if selector_payload is not None
            else "directory_or_prefix_discovery"
        ),
        "selector_path": str(selector_file) if selector_file is not None else None,
        "selector_sha256": (
            _sha256_file(selector_file) if selector_file is not None else None
        ),
        "recovery_bundle": selector_payload is not None or prefix is None,
        "valid": valid,
        "all_four_canaries_validated": valid,
        "remote_calls_started_by_validator": 0,
        "provider_calls_started_by_validator": 0,
        "training_runs_started_by_validator": 0,
        "provider_attempts_observed": sum(
            int(report.get("provider_attempt_count", 0))
            for report in successful
        ),
        "provider_input_tokens": provider_input_tokens,
        "provider_output_tokens": provider_output_tokens,
        "provider_total_tokens": provider_total_tokens,
        "runs": run_reports,
        "errors": errors,
    }


def build_report(
    *,
    project_root: str | Path = ROOT,
    mps_smoke_output: str | Path | None = None,
    cuda_smoke_output: str | Path | None = None,
    modal_canary_download_root: str | Path | None = None,
    modal_canary_selector: str | Path | None = None,
) -> dict[str, Any]:
    surfaces = validate_controller_surfaces(project_root)
    mps = validate_existing_mps_smoke(
        mps_smoke_output,
        project_root=project_root,
    )
    cuda = validate_existing_cuda_smoke(
        cuda_smoke_output,
        project_root=project_root,
    )
    modal_canaries = validate_downloaded_modal_canaries(
        modal_canary_download_root,
        modal_canary_selector=modal_canary_selector,
        project_root=project_root,
    )
    accelerator_smoke_valid = (
        cuda["artifact_self_consistent"]
        if cuda["provided"]
        else mps["artifact_self_consistent"]
    )
    return {
        "schema_name": "FourHarnessStaticSurfaceReport",
        "schema_version": "3.0",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "static_controller_surfaces_and_optional_smoke_artifact_consistency",
        "status": "static_controller_surfaces_passed"
        if surfaces["passed"]
        else "blocked",
        "static_controller_surfaces_passed": surfaces["passed"],
        "accelerator_smoke_artifacts_self_consistent": accelerator_smoke_valid,
        "cuda_smoke_artifacts_self_consistent": cuda["artifact_self_consistent"],
        "cuda_execution_origin_attested": False,
        "modal_canaries_validated": modal_canaries["all_four_canaries_validated"],
        "mps_smoke_artifacts_self_consistent": mps["artifact_self_consistent"],
        "mps_execution_origin_attested": False,
        "provider_calls": 0,
        "local_fixture_calls": surfaces["local_fixture_calls"],
        "entrypoint_execution_runs": 0,
        "candidate_execution_runs": 0,
        "training_runs": 0,
        "scientific": False,
        "scientific_pilot_ready": False,
        "autonomous_generated_candidate_execution_ready": False,
        "static_controller_surfaces": surfaces,
        "existing_cuda_smoke_artifacts": cuda,
        "downloaded_modal_canaries": modal_canaries,
        "existing_mps_smoke_artifacts": mps,
        "limitations": [
            "The fake provider is local and deterministic; no provider "
            "connectivity was tested.",
            "Entrypoints are parsed statically and never imported or executed.",
            "The fixed response is not injected into a live controller.",
            "The complete child IR was validated but never constructed or executed.",
            "Smoke artifacts are self-authored and cannot independently attest "
            "execution origin.",
            "Smoke artifact consistency, when supplied, covers only the trusted "
            "checked-in seed.",
            "This report cannot authorize live generated-candidate execution or "
            "a scientific pilot.",
            "Use scripts/audit_scientific_readiness.py for the separate "
            "fail-closed scientific audit.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Statically validate four controller surfaces without executing them."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--mps-smoke-output",
        type=Path,
        help=("optional historical smoke_train_v1 output to check without rewriting"),
    )
    parser.add_argument("--require-mps-smoke", action="store_true")
    parser.add_argument(
        "--cuda-smoke-output",
        type=Path,
        help="optional Modal smoke_train_cuda_v2 output to check for consistency",
    )
    parser.add_argument("--require-cuda-smoke", action="store_true")
    parser.add_argument(
        "--modal-canary-download-root",
        type=Path,
        help=(
            "local directory containing exactly four downloaded Modal canary runs, "
            "or a non-existent path whose name is their shared run-ID prefix"
        ),
    )
    parser.add_argument(
        "--modal-canary-selector",
        type=Path,
        help=(
            "create-only ModalProviderCanaryRunSelector/2.0 selecting exact "
            "aggregate/recovery downloads"
        ),
    )
    parser.add_argument(
        "--create-modal-canary-selector",
        metavar="SELECTOR_ID",
        help=(
            "create the canonical cohort-scoped selector and exit; requires "
            "the three cohort identity flags and four --modal-canary-run values"
        ),
    )
    parser.add_argument("--source-tree-sha256", default="")
    parser.add_argument("--image-source-sha256", default="")
    parser.add_argument("--cohort-id", default="")
    parser.add_argument(
        "--modal-canary-run",
        action="append",
        default=[],
        metavar="HARNESS=RUN_ID",
        help="one accepted downloaded run for selector creation",
    )
    parser.add_argument("--require-modal-canaries", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.create_modal_canary_selector is not None:
        if (
            arguments.modal_canary_selector is not None
            or arguments.modal_canary_download_root is not None
            or arguments.mps_smoke_output is not None
            or arguments.cuda_smoke_output is not None
            or arguments.require_mps_smoke
            or arguments.require_cuda_smoke
            or arguments.require_modal_canaries
            or arguments.output is not None
        ):
            parser.error("selector creation cannot be combined with report options")
        try:
            run_ids = _parse_modal_canary_run_selections(
                arguments.modal_canary_run
            )
            identity = ModalLiveCohortIdentity(
                source_tree_sha256=arguments.source_tree_sha256,
                image_source_sha256=arguments.image_source_sha256,
                cohort_id=arguments.cohort_id,
            )
            _selector_path, selector = create_modal_canary_selector(
                selector_id=arguments.create_modal_canary_selector,
                run_ids=run_ids,
                identity=identity,
                project_root=arguments.project_root,
            )
        except (
            ArtifactIntegrityError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            parser.error(str(error))
        print(json.dumps(selector, indent=2, sort_keys=True))
        return 0
    if arguments.modal_canary_run:
        parser.error("--modal-canary-run requires --create-modal-canary-selector")
    if any(
        (
            arguments.source_tree_sha256,
            arguments.image_source_sha256,
            arguments.cohort_id,
        )
    ):
        parser.error("cohort identity flags require --create-modal-canary-selector")
    if (
        arguments.modal_canary_selector is not None
        and arguments.modal_canary_download_root is not None
    ):
        parser.error("provide --modal-canary-selector or --modal-canary-download-root")
    report = build_report(
        project_root=arguments.project_root,
        mps_smoke_output=arguments.mps_smoke_output,
        cuda_smoke_output=arguments.cuda_smoke_output,
        modal_canary_download_root=arguments.modal_canary_download_root,
        modal_canary_selector=arguments.modal_canary_selector,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    passed = report["static_controller_surfaces_passed"]
    if arguments.require_mps_smoke:
        passed = passed and report["mps_smoke_artifacts_self_consistent"]
    if arguments.require_cuda_smoke:
        passed = passed and report["cuda_smoke_artifacts_self_consistent"]
    if arguments.require_modal_canaries:
        passed = passed and report["modal_canaries_validated"]
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
