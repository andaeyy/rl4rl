"""Fail-closed, provider-free audit for paid-pilot scientific readiness."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.evaluation_profiles import EVALUATION_PROFILES
from common.training_config import FULL_TRAIN_CUDA_V2, FULL_TRAIN_V1
from containment.audit import audit_runtime
from evaluation.dependency_audit import assert_controller_dependencies_clean
from mechanism.plans import load_frozen_mechanism_plan
from novelty.corpus import verify_frozen_corpus
from novelty.dependency_audit import audit_science_boundary
from replication.policy import load_frozen_replication_policy
from analysis.plan import load_frozen_analysis_plan
from research_ledger.protocol import load_frozen_protocol
from study.contracts import ConditionSpec, StudySpec
from study.randomization import RandomizationPlan
from study.serialization import read_json
from scripts.record_accelerator_validation import (
    validate_accelerator_validation_evidence,
)
from scripts.record_local_engineering_evidence import (
    LOCAL_ENGINEERING_RECEIPT_CONTRACTS,
    current_local_engineering_receipt_path,
    validate_local_engineering_freeze_receipt,
    validate_local_engineering_receipt,
)
from scripts.record_modal_readiness import (
    MODAL_READINESS_RECEIPT_CONTRACTS,
    validate_modal_readiness_gate_record,
)


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    evidence: str
    blockers: tuple[str, ...] = ()


def _null_paths(value: Any, prefix: str = "") -> list[str]:
    if value is None:
        return [prefix or "<root>"]
    if isinstance(value, dict):
        paths: list[str] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_null_paths(item, child))
        return paths
    if isinstance(value, list):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_null_paths(item, f"{prefix}[{index}]"))
        return paths
    return []


_DECISION_SECTIONS = (
    "treatment",
    "budgets",
    "evaluation",
    "containment",
    "novelty",
    "mechanism_and_replication",
    "statistics",
    "scope_and_release",
)
_PLACEHOLDER_VALUES = {
    "tbd",
    "todo",
    "unknown",
    "unresolved",
    "decision_required",
    "pi_required",
    "placeholder",
}


def _unresolved_value_paths(value: Any, prefix: str) -> list[str]:
    if value is None:
        return [prefix]
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized or normalized in _PLACEHOLDER_VALUES:
            return [prefix]
        return []
    if isinstance(value, dict):
        if not value:
            return [prefix]
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(_unresolved_value_paths(item, f"{prefix}.{key}"))
        return paths
    if isinstance(value, list):
        if not value:
            return [prefix]
        paths: list[str] = []
        for index, item in enumerate(value):
            paths.extend(_unresolved_value_paths(item, f"{prefix}[{index}]"))
        return paths
    return []


def _valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _decision_issues(decisions: Any) -> list[str]:
    if not isinstance(decisions, dict):
        return ["decision ledger must be a mapping"]
    common_root = {
        "schema_name",
        "schema_version",
        "status",
        "note",
        "approval",
        "launch_authorization",
        *_DECISION_SECTIONS,
    }
    issues: list[str] = []
    version = decisions.get("schema_version")
    expected_root = (
        common_root | {"schema_migration"}
        if version == "2"
        else common_root
    )
    if set(decisions) != expected_root:
        issues.append(
            f"decision ledger root fields differ from the frozen v{version} schema"
        )
    if decisions.get("schema_name") != "scientific_decision_ledger":
        issues.append("decision ledger schema_name is invalid")
    if version not in {"1", "2"}:
        issues.append("decision ledger schema_version is invalid")
    if version == "2":
        migration = decisions.get("schema_migration")
        expected_migration = {
            "from_version": "1",
            "active_replacements": {
                "containment.mps_resource_limits": (
                    "containment.accelerator_resource_limits"
                ),
                "containment.mps_evidence_custody_rule": (
                    "containment.accelerator_evidence_custody_rule"
                ),
            },
            "compatibility": (
                "historical version-1 ledgers remain readable and hash-stable"
            ),
            "equivalence_claim": (
                "Modal CUDA is a new execution condition, not an MPS-equivalent condition"
            ),
        }
        if migration != expected_migration:
            issues.append("schema_migration does not match the frozen v1-to-v2 migration")
    if decisions.get("status") != "approved":
        issues.append("decision ledger status must be 'approved'")
    for section in _DECISION_SECTIONS:
        if section not in decisions:
            issues.append(f"missing decision section: {section}")
            continue
        issues.extend(
            f"unresolved: {path}"
            for path in _unresolved_value_paths(decisions[section], section)
        )

    approval = decisions.get("approval")
    if not isinstance(approval, dict) or set(approval) != {
        "principal_investigator",
        "approved_at_utc",
        "decision_attestation",
    }:
        issues.append("approval must match the frozen approval schema")
    else:
        for field in ("principal_investigator", "decision_attestation"):
            value = approval.get(field)
            if not isinstance(value, str) or not value.strip():
                issues.append(f"approval.{field} must be non-empty text")
        if not _valid_utc_timestamp(approval.get("approved_at_utc")):
            issues.append("approval.approved_at_utc must be an explicit UTC timestamp")

    launch = decisions.get("launch_authorization")
    if not isinstance(launch, dict) or set(launch) != {
        "pilot_authorized",
        "main_authorized",
        "authorized_by",
        "authorized_at_utc",
    }:
        issues.append("launch_authorization must match the frozen schema")
    else:
        for field in ("pilot_authorized", "main_authorized"):
            if type(launch.get(field)) is not bool:
                issues.append(f"launch_authorization.{field} must be boolean")

    def exact_int(path: str, *, minimum: int) -> None:
        section, field = path.split(".", 1)
        value = decisions.get(section, {}).get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < minimum
        ):
            issues.append(f"{path} must be an integer >= {minimum}")

    exact_int("treatment.portfolio_size_k", minimum=2)
    exact_int("budgets.generator_token_ceiling", minimum=1)
    exact_int("budgets.proposal_opportunity_ceiling", minimum=1)
    exact_int("budgets.descendant_training_ceiling", minimum=1)
    exact_int("budgets.repair_limit_per_opportunity", minimum=0)
    exact_int("budgets.provider_attempt_limit_per_opportunity", minimum=1)
    exact_int("evaluation.layer_a_case_count", minimum=10_000)
    exact_int("evaluation.layer_b_case_count", minimum=10_000)
    exact_int("evaluation.layer_c_case_count", minimum=10_000)
    exact_int("novelty.reviewer_count", minimum=3)

    schedule = decisions.get("treatment", {}).get("transition_schedule")
    if schedule is not None and (
        not isinstance(schedule, list)
        or not schedule
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in schedule
        )
        or len(schedule) != len(set(schedule))
    ):
        issues.append("treatment.transition_schedule must be unique positive integers")
    seeds = decisions.get("mechanism_and_replication", {}).get(
        "replication_seed_list"
    )
    if seeds is not None and (
        not isinstance(seeds, list)
        or not seeds
        or len({json.dumps(item, sort_keys=True) for item in seeds}) != len(seeds)
    ):
        issues.append("replication_seed_list must be a non-empty unique list")
    for field in (
        "smallest_effect_of_interest",
        "target_power_or_precision",
        "type_i_error",
    ):
        value = decisions.get("statistics", {}).get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            issues.append(f"statistics.{field} must be a finite positive number")
    for field in ("target_power_or_precision", "type_i_error"):
        value = decisions.get("statistics", {}).get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 1:
            issues.append(f"statistics.{field} must be less than one")
    return issues


def _gate(name: str, check: Callable[[], str]) -> GateResult:
    try:
        evidence = check()
    except Exception as error:
        return GateResult(
            gate=name,
            passed=False,
            evidence="",
            blockers=(f"{type(error).__name__}: {error}",),
        )
    return GateResult(gate=name, passed=True, evidence=evidence)


def _file_gate(name: str, path: Path, description: str) -> GateResult:
    if not path.is_file():
        return GateResult(name, False, "", (f"missing {description}: {path}",))
    return GateResult(name, True, str(path))


def _sha256_canonical(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_readiness(
    *,
    root: Path = ROOT,
    corpus_manifest: Path | None = None,
    corpus_seal: Path | None = None,
    analysis_plan: Path | None = None,
    mechanism_plan: Path | None = None,
    replication_policy: Path | None = None,
    research_protocol: Path | None = None,
    mps_evidence: Path | None = None,
    accelerator_evidence: Path | None = None,
    study_spec: Path | None = None,
) -> dict[str, Any]:
    results: list[GateResult] = []

    def common_engine() -> str:
        conditions = ConditionSpec.primary()
        if len(conditions) != 4:
            raise ValueError("primary engine does not expose exactly C0-C3")
        expected = {
            "C0": ("single", "ordinary"),
            "C1": ("single", "scheduled_transition"),
            "C2": ("portfolio", "ordinary"),
            "C3": ("portfolio", "scheduled_transition"),
        }
        observed = {
            item.condition_id.value: (
                item.parent_policy.value,
                item.proposal_policy.value,
            )
            for item in conditions
        }
        if observed != expected:
            raise ValueError("C0-C3 treatment mapping differs from the contract")
        for path in (
            root / "study" / "engine.py",
            root / "study" / "randomization.py",
            root / "study" / "scheduling.py",
            root / "study" / "runtime_adapters.py",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        return "common engine, randomization, scheduler, and runtime adapters present"

    results.append(_gate("common_c0_c3_engine", common_engine))

    def evaluation_firewall() -> str:
        entries = tuple((root / "agents").glob("*/run.py")) + (
            root / "common" / "openevolve_runner.py",
            root / "study" / "runtime_adapters.py",
        )
        assert_controller_dependencies_clean(entries, project_root=root)
        issues = audit_science_boundary(root)
        if issues:
            raise ValueError(
                "; ".join(
                    f"{issue.path}:{issue.line} {issue.rule}"
                    for issue in issues
                )
            )
        return "controller dependency graph is free of sealed and post-search science"

    results.append(_gate("evaluation_and_novelty_firewall", evaluation_firewall))

    def profiles() -> str:
        for profile in EVALUATION_PROFILES.values():
            profile.validate_definition()
        for name in (
            "scientific_layer_a_v1",
            "scientific_layer_b_v1",
            "scientific_layer_c_v1",
        ):
            profile = EVALUATION_PROFILES[name]
            if profile.default_case_count is not None or profile.minimum_case_count < 10_000:
                raise ValueError(f"{name} permits an implicit or smoke-sized count")
        return "scientific A/B/C profiles require explicit counts of at least 10,000"

    results.append(_gate("scientific_evaluation_profiles", profiles))

    def scientific_no_search() -> str:
        provider_adapter = root / "baselines" / "provider.py"
        scientific_runner = root / "scripts" / "study_scientific_run.py"
        if not provider_adapter.is_file():
            raise FileNotFoundError(
                "feedback-free real-provider no-search adapter is missing"
            )
        if "NoSearchProposalGenerator" not in scientific_runner.read_text(
            encoding="utf-8"
        ):
            raise ValueError(
                "scientific scheduler does not include the no-search assignment"
            )
        return "real no-search adapter is scheduled with the matched downstream path"

    results.append(
        _gate("scientific_no_search_execution_integrated", scientific_no_search)
    )

    def sealed_post_search() -> str:
        post_search = root / "scripts" / "study_post_search.py"
        if not post_search.is_file():
            raise FileNotFoundError(
                "trusted Layer B post-search orchestration entrypoint is missing"
            )
        source = post_search.read_text(encoding="utf-8")
        for token in (
            "LayerBQualificationRunner",
            "freeze_completed_run",
            "ImmutableStudyEventSink",
        ):
            if token not in source:
                raise ValueError(f"post-search pipeline lacks {token}")
        return "Layer B qualification is connected to frozen runs and immutable events"

    results.append(
        _gate("sealed_post_search_execution_integrated", sealed_post_search)
    )

    decisions_path = root / "scientific_decisions.yaml"
    decisions = yaml.safe_load(decisions_path.read_text(encoding="utf-8"))
    decision_issues = _decision_issues(decisions)
    decisions_hash = _sha256_canonical(decisions)
    results.append(
        GateResult(
            "principal_investigator_decisions",
            not decision_issues,
            f"{decisions_path} sha256={decisions_hash}",
            tuple(decision_issues),
        )
    )

    manifest = yaml.safe_load((root / "experiment_manifest.yaml").read_text())
    evidence_ledger = yaml.safe_load(
        (root / "readiness_evidence.yaml").read_text(encoding="utf-8")
    )
    manifest_hash = _sha256_canonical(manifest)
    active_accelerator_manifest = bool(
        isinstance(manifest, dict) and manifest.get("schema_version") == "3"
    )
    active_training_profile = (
        FULL_TRAIN_CUDA_V2 if active_accelerator_manifest else FULL_TRAIN_V1
    )

    def manifest_security_invariants() -> str:
        if not isinstance(manifest, dict):
            raise ValueError("experiment manifest must be a mapping")
        if manifest.get("schema_name") != "architecture_discovery_experiment_manifest":
            raise ValueError("experiment manifest schema_name is invalid")
        if manifest.get("schema_version") not in {"2", "3"}:
            raise ValueError("experiment manifest schema_version is invalid")
        expected_values: dict[str, Any] = {
            "study.parameter_count_role": "descriptive_metadata_only",
            "evaluation.parameter_count_in_fitness": False,
            "evaluation.internet_access_during_runs": False,
            "evaluation.layer_a.controller_visible": True,
            "evaluation.layer_b.controller_visible": False,
            "evaluation.layer_b.sealed": True,
            "evaluation.layer_c.controller_visible": False,
            "evaluation.layer_c.sealed": True,
            "evaluation.layer_c.one_shot_release": True,
            "training.profile": active_training_profile.name,
            "training.profile_version": active_training_profile.version,
            "training.profile_hash": active_training_profile.profile_hash,
            "training.scientific": True,
            "training.device": active_training_profile.device_requirement,
            "training.cpu_fallback": False,
            "training.parallel_candidate_training": False,
            "engineering_smoke.scientific": False,
            "secondary_controls.no_search.adaptive_feedback": False,
            "shared_generation.target_model": "gpt-5.6-sol",
        }
        if active_accelerator_manifest:
            expected_values.update(
                {
                    "machine.execution_backend": "modal",
                    "machine.preferred_device": "cuda",
                    "machine.scientific_fallback_device": "none",
                    "runtime.python": "3.12",
                    "runtime.setup_selected_device": "cuda",
                    "runtime.pytorch_enable_mps_fallback": False,
                    "runtime.cublas_workspace_config": ":4096:8",
                    "training.deterministic_algorithms": True,
                    "training.cudnn_deterministic": True,
                    "training.cudnn_benchmark": False,
                    "training.allow_tf32": False,
                    "training.cublas_workspace_config": ":4096:8",
                    "remote_execution.provider": "modal",
                    "remote_execution.profile": "scalingintelligence",
                    "remote_execution.app_name": "rl4rl-architecture-discovery",
                    "remote_execution.app_module": "modal_app.py",
                    "remote_execution.mode": "ephemeral_modal_run",
                    "remote_execution.deployed_app": False,
                    "remote_execution.detached_calls": False,
                    "remote_execution.python": "3.12",
                    "remote_execution.initial_gpu": "T4",
                    "remote_execution.function_cpu_request_cores": 2.0,
                    "remote_execution.function_cpu_soft_limit_cores": 2.0,
                    "remote_execution.function_cpu_limit_kind": (
                        "soft_throttle_threshold"
                    ),
                    "remote_execution.function_memory_request_mib": 8192,
                    "remote_execution.function_memory_limit_mib": 8192,
                    "remote_execution.function_memory_limit_kind": "hard",
                    "remote_execution.function_platform_compute_cost_ceiling_enforced": (
                        False
                    ),
                    "remote_execution.runtime_functions_preemptible": True,
                    "remote_execution.platform_preemption_restart_possible": True,
                    "remote_execution.logical_call_count_is_not_container_attempt_ceiling": (
                        True
                    ),
                    "remote_execution.function_region": None,
                    "remote_execution.image_build_cpu_request_cores": 2.0,
                    "remote_execution.image_build_cpu_soft_limit_cores": None,
                    "remote_execution.image_build_memory_request_mib": 8192,
                    "remote_execution.image_build_memory_limit_mib": None,
                    "remote_execution.image_build_region": None,
                    "remote_execution.image_build_subprocess_thread_limit": 2,
                    "remote_execution.image_build_resource_limits_exposed": (
                        False
                    ),
                    "remote_execution.image_build_platform_compute_cost_ceiling_enforced": (
                        False
                    ),
                    "remote_execution.modal_price_basis_schema": (
                        "ModalPriceBasis/1.0"
                    ),
                    "remote_execution.modal_price_basis_max_age_hours": 48,
                    "remote_execution.modal_cost_gate_scope": (
                        "local_pre_popen_request_rate_and_one_gib_month_storage_"
                        "estimate_not_platform_billing_cap"
                    ),
                    "remote_execution.provider_canary_aggregate_outcome_schema": (
                        "ProviderCanaryAggregateOutcomeReceipt/1.1"
                    ),
                    "remote_execution.modal_local_host_anchor_schema": (
                        "ModalLocalHostAnchor/1.0"
                    ),
                    "remote_execution.modal_remote_run_reservation_schema": (
                        "ModalRemoteRunReservation/1.2"
                    ),
                    "remote_execution.modal_action_intent_schema": (
                        "ModalActionIntent/1.6"
                    ),
                    "remote_execution.modal_local_process_start_schema": (
                        "ModalLocalProcessStart/1.1"
                    ),
                    "remote_execution.modal_action_attempt_receipt_schema": (
                        "ModalActionAttemptReceipt/3.6"
                    ),
                    "remote_execution.modal_action_recovery_request_schema": (
                        "ModalActionRecoveryRequest/1.0"
                    ),
                    "remote_execution.modal_action_recovery_intent_schema": (
                        "ModalActionRecoveryIntent/1.0"
                    ),
                    "remote_execution.modal_action_recovery_host_containment_schema": (
                        "ModalActionRecoveryHostContainment/1.0"
                    ),
                    "remote_execution.modal_action_recovery_resolution_schema": (
                        "ModalActionRecoveryResolution/1.0"
                    ),
                    "remote_execution.modal_prior_cohort_quarantine_accounting_schema": (
                        "ModalPriorCohortQuarantineAccounting/1.1"
                    ),
                    "remote_execution.modal_action_lock_path": (
                        "outputs/readiness/.modal_action.lock"
                    ),
                    "remote_execution.modal_action_lock_scope": (
                        "launcher_recovery_snapshot_capture_prior_accounting_lineage_"
                        "roster_resource_cleanup_migration_bundle_and_global_"
                        "journal_scan"
                    ),
                    "remote_execution.modal_global_action_journal_scanner_implemented": (
                        True
                    ),
                    "remote_execution.modal_global_action_journal_prelaunch_gate_wired": (
                        True
                    ),
                    "remote_execution.modal_action_orphan_recovery_status": (
                        "operational_exact_v1_cli_scanner_validated"
                    ),
                    "remote_execution.max_containers": 1,
                    "remote_execution.min_containers": 0,
                    "remote_execution.retries": 0,
                    "remote_execution.function_timeout_seconds": 300,
                    "remote_execution.parallel_candidate_training": False,
                    "remote_execution.provider_canaries_sequential": True,
                    "remote_execution.artifact_volume": (
                        "rl4rl-architecture-artifacts"
                    ),
                    "remote_execution.artifact_mount": "/mnt/discovery",
                    "artifacts_and_reconstruction.active_budget_schema": (
                        "BudgetLedger/2.0"
                    ),
                    "artifacts_and_reconstruction.active_training_result_schema": (
                        "TrainingResult/2.0"
                    ),
                    "artifacts_and_reconstruction.active_compute_field": (
                        "accelerator_seconds"
                    ),
                    "artifacts_and_reconstruction.active_compute_kind_field": (
                        "accelerator_kind"
                    ),
                    "historical_mps_compatibility.status": (
                        "retained_read_only_compatible"
                    ),
                    "historical_mps_compatibility.cross_device_equivalence_claim": (
                        "none"
                    ),
                    "historical_mps_compatibility.full_profile.name": (
                        FULL_TRAIN_V1.name
                    ),
                    "historical_mps_compatibility.full_profile.version": (
                        FULL_TRAIN_V1.version
                    ),
                    "historical_mps_compatibility.full_profile.hash": (
                        FULL_TRAIN_V1.profile_hash
                    ),
                }
            )

        def lookup(path: str) -> Any:
            value: Any = manifest
            for part in path.split("."):
                if not isinstance(value, dict) or part not in value:
                    raise ValueError(f"manifest field is missing: {path}")
                value = value[part]
            return value

        mismatches: dict[str, dict[str, Any]] = {}
        for path, expected in expected_values.items():
            observed = lookup(path)
            if type(expected) is bool and type(observed) is not bool:
                mismatches[path] = {"expected": expected, "observed": observed}
            elif (
                isinstance(expected, int)
                and not isinstance(expected, bool)
                and (
                    not isinstance(observed, int)
                    or isinstance(observed, bool)
                )
            ):
                mismatches[path] = {"expected": expected, "observed": observed}
            elif observed != expected:
                mismatches[path] = {"expected": expected, "observed": observed}
        for path in (
            "study.pilot_ready",
            "study.pilot_validated",
            "study.main_study_ready",
        ):
            observed = lookup(path)
            if type(observed) is not bool:
                mismatches[path] = {"expected": "boolean", "observed": observed}
        if lookup("study.launch_status") not in {
            "blocked",
            "pilot_authorized",
            "main_authorized",
        }:
            mismatches["study.launch_status"] = {
                "expected": "blocked|pilot_authorized|main_authorized",
                "observed": lookup("study.launch_status"),
            }
        if mismatches:
            raise ValueError(f"manifest security invariant mismatch: {mismatches}")
        return f"experiment_manifest.yaml sha256={manifest_hash}"

    results.append(_gate("manifest_security_invariants", manifest_security_invariants))

    if active_accelerator_manifest:
        def modal_cuda_configuration() -> str:
            remote = manifest.get("remote_execution")
            if not isinstance(remote, dict):
                raise ValueError("active manifest lacks remote_execution")
            if remote.get("provider") != "modal":
                raise ValueError("active remote execution provider is not Modal")
            if manifest.get("training", {}).get("device") != "cuda":
                raise ValueError("active scientific training device is not CUDA")
            if (
                remote.get("max_containers"),
                remote.get("min_containers"),
                remote.get("retries"),
                remote.get("function_timeout_seconds"),
            ) != (1, 0, 0, 300):
                raise ValueError("Modal execution bounds differ from the frozen limits")
            if remote.get("parallel_candidate_training") is not False:
                raise ValueError("Modal candidate training is not sequential")
            for relative in (
                "modal_app.py",
                "modal_boundary.py",
                "common/runtime_context.py",
                "scripts/record_accelerator_validation.py",
            ):
                if not (root / relative).is_file():
                    raise FileNotFoundError(root / relative)
            return (
                "Modal/CUDA execution is configured for one T4, one container, "
                "zero retries, and a 300-second timeout"
            )

        results.append(
            _gate("modal_cuda_execution_configured", modal_cuda_configuration)
        )
        for modal_gate_name in MODAL_READINESS_RECEIPT_CONTRACTS:
            results.append(
                _gate(
                    modal_gate_name,
                    lambda gate_name=modal_gate_name: (
                        validate_modal_readiness_gate_record(
                            evidence_ledger,
                            gate_name,
                            root=root,
                        )
                    ),
                )
            )

    def launch_authorization(*, phase: str) -> str:
        launch = decisions.get("launch_authorization", {})
        authorization_field = f"{phase}_authorized"
        if type(launch.get(authorization_field)) is not bool:
            raise ValueError(f"{authorization_field} must be boolean")
        if launch.get(authorization_field) is not True:
            raise ValueError(f"PI has not authorized the {phase} launch")
        if not isinstance(launch.get("authorized_by"), str) or not launch[
            "authorized_by"
        ].strip():
            raise ValueError("launch authorization lacks an authorizing identity")
        if not _valid_utc_timestamp(launch.get("authorized_at_utc")):
            raise ValueError("launch authorization lacks a UTC timestamp")
        required_status = "main_authorized" if phase == "main" else {
            "pilot_authorized",
            "main_authorized",
        }
        manifest_status = manifest.get("study", {}).get("launch_status")
        evidence_status = evidence_ledger.get("status")
        if phase == "main":
            allowed_statuses = {required_status}
        else:
            allowed_statuses = required_status
        if manifest_status not in allowed_statuses or evidence_status not in allowed_statuses:
            raise ValueError(
                f"manifest/readiness kill switches do not authorize {phase}"
            )
        readiness_field = "main_study_ready" if phase == "main" else "pilot_ready"
        if manifest.get("study", {}).get(readiness_field) is not True:
            raise ValueError(f"manifest {readiness_field} switch is not true")
        return f"authorized_by={launch['authorized_by']} at={launch['authorized_at_utc']}"

    results.append(
        _gate(
            "pilot_launch_authorization",
            lambda: launch_authorization(phase="pilot"),
        )
    )
    results.append(
        _gate(
            "main_launch_authorization",
            lambda: launch_authorization(phase="main"),
        )
    )

    executable_spec_path = study_spec or root / "study" / "scientific_study.json"

    def load_executable_spec() -> StudySpec:
        spec = StudySpec.from_dict(
            json.loads(executable_spec_path.read_text(encoding="utf-8"))
        )
        if not spec.scientific:
            raise ValueError("executable StudySpec is marked toy/non-scientific")
        return spec

    def frozen_executable_contract() -> str:
        spec = load_executable_spec()
        comparisons = {
            "accelerator_kind": ("cuda", spec.budget.accelerator_kind),
            "portfolio_size_k": (
                decisions["treatment"]["portfolio_size_k"],
                spec.portfolio_size,
            ),
            "transition_schedule": (
                decisions["treatment"]["transition_schedule"],
                list(spec.transition_opportunities),
            ),
            "proposal_opportunity_ceiling": (
                decisions["budgets"]["proposal_opportunity_ceiling"],
                spec.budget.proposal_opportunities,
            ),
            "provider_attempt_limit_per_opportunity": (
                decisions["budgets"]["provider_attempt_limit_per_opportunity"],
                spec.budget.provider_attempts_per_opportunity,
            ),
            "repair_limit_per_opportunity": (
                decisions["budgets"]["repair_limit_per_opportunity"],
                spec.budget.repair_attempts_per_opportunity,
            ),
            "manifest_portfolio_size_k": (
                manifest["primary_causal_design"]["portfolio_size_k"],
                spec.portfolio_size,
            ),
            "manifest_transition_schedule": (
                manifest["primary_causal_design"]["transition_schedule"],
                list(spec.transition_opportunities),
            ),
            "manifest_hash": (manifest_hash, spec.common_config_hash),
        }
        mismatches = {
            name: {"frozen": frozen, "study_spec": observed}
            for name, (frozen, observed) in comparisons.items()
            if frozen != observed
        }
        if mismatches:
            raise ValueError(f"executable StudySpec mismatch: {mismatches}")
        return f"{executable_spec_path} spec_hash={spec.spec_hash}"

    results.append(
        _gate("frozen_executable_study_contract", frozen_executable_contract)
    )

    layer_issues: list[str] = []
    layer_source_ids: list[str] = []
    layer_source_hashes: list[str] = []
    for layer in ("layer_a", "layer_b", "layer_c"):
        config = manifest["evaluation"][layer]
        for field in ("case_count", "source_id", "source_hash"):
            if config.get(field) is None:
                layer_issues.append(f"{layer}.{field} is unresolved")
        case_count = config.get("case_count")
        profile_name = config.get("profile")
        profile = EVALUATION_PROFILES.get(profile_name)
        if profile is None:
            layer_issues.append(f"{layer}.profile is unknown: {profile_name!r}")
        elif case_count is not None:
            if not isinstance(case_count, int) or isinstance(case_count, bool):
                layer_issues.append(f"{layer}.case_count must be an integer")
            else:
                try:
                    profile.resolve_case_count(case_count)
                except ValueError as error:
                    layer_issues.append(f"{layer}.case_count invalid: {error}")
        source_id = config.get("source_id")
        source_hash = config.get("source_hash")
        if source_id is not None:
            if not isinstance(source_id, str) or not source_id:
                layer_issues.append(f"{layer}.source_id must be non-empty text")
            else:
                layer_source_ids.append(source_id)
        if source_hash is not None:
            if not isinstance(source_hash, str):
                layer_issues.append(f"{layer}.source_hash must be text")
                continue
            layer_source_hashes.append(source_hash)
            if not _sha256_digest(source_hash):
                layer_issues.append(f"{layer}.source_hash is not a lowercase SHA-256")
    if len(layer_source_ids) == 3 and len(set(layer_source_ids)) != 3:
        layer_issues.append("Layer A/B/C source IDs are not pairwise disjoint")
    if len(layer_source_hashes) == 3 and len(set(layer_source_hashes)) != 3:
        layer_issues.append("Layer A/B/C source hashes are not pairwise disjoint")
    results.append(
        GateResult(
            "disjoint_frozen_layer_sources",
            not layer_issues,
            "experiment_manifest.yaml:evaluation",
            tuple(layer_issues),
        )
    )

    containment = audit_runtime()
    results.append(
        GateResult(
            "strong_candidate_containment",
            containment.strong_containment_proven,
            containment.audit_hash,
            ()
            if containment.strong_containment_proven
            else ("required OS boundary controls are not proven",),
        )
    )
    if not active_accelerator_manifest:
        mps_ready = bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )
        results.append(
            GateResult(
                "mps_available_no_fallback",
                mps_ready and not containment.mps_fallback_requested,
                f"built={containment.mps_built}, available={containment.mps_available}",
                ()
                if mps_ready and not containment.mps_fallback_requested
                else ("MPS is unavailable or fallback is requested",),
            )
        )
    results.append(
        GateResult(
            "trusted_ir_interpreter",
            (root / "architecture_ir" / "interpreter.py").is_file(),
            "architecture_ir/interpreter.py",
            ()
            if (root / "architecture_ir" / "interpreter.py").is_file()
            else ("typed IR exists but no trusted evaluator-owned interpreter exists",),
        )
    )

    def runtime_validity_integration() -> str:
        interpreter = root / "architecture_ir" / "interpreter.py"
        evaluator = root / "common" / "evaluator.py"
        record_schema = root / "evaluation" / "records.py"
        if not interpreter.is_file():
            raise FileNotFoundError(interpreter)
        evaluator_source = evaluator.read_text(encoding="utf-8")
        record_source = record_schema.read_text(encoding="utf-8")
        if "probe_runtime_validity" not in evaluator_source:
            raise ValueError("Layer A evaluator does not run the trusted runtime probe")
        if "runtime_validity" not in record_source:
            raise ValueError("evaluation records do not retain runtime-validity evidence")
        return "trusted IR runtime/metamorphic evidence is retained by evaluation"

    results.append(
        _gate("runtime_transformer_validity_integrated", runtime_validity_integration)
    )

    corpus_manifest = corpus_manifest or root / "novelty" / "reference_corpus.json"
    corpus_seal = corpus_seal or root / "novelty" / "reference_corpus.seal.json"
    verification = verify_frozen_corpus(
        manifest_path=corpus_manifest,
        seal_path=corpus_seal,
        require_scientific_ready=True,
    )
    results.append(
        GateResult(
            "frozen_populated_reference_corpus",
            verification.valid and verification.scientific_ready,
            verification.corpus_sha256 or "",
            verification.issues
            or (() if verification.scientific_ready else ("corpus is not scientifically ready",)),
        )
    )

    reviewer_roster = root / "review" / "reviewer_roster.sealed.json"

    def reviewers() -> str:
        payload = json.loads(reviewer_roster.read_text(encoding="utf-8"))
        if payload.get("schema_name") != "IndependentReviewerRoster":
            raise ValueError("reviewer roster has the wrong schema")
        if payload.get("schema_version") != "1.0" or payload.get("sealed") is not True:
            raise ValueError("reviewer roster is not a sealed v1.0 record")
        roster = payload.get("reviewers")
        if not isinstance(roster, list) or len(roster) < 3:
            raise ValueError("at least three independent reviewers are required")
        pseudonyms = [item.get("reviewer_pseudonym") for item in roster]
        if any(not value for value in pseudonyms) or len(set(pseudonyms)) != len(roster):
            raise ValueError("reviewer pseudonyms must be present and unique")
        if any(item.get("independence_attested") is not True for item in roster):
            raise ValueError("every reviewer requires an independence attestation")
        if not payload.get("custodian") or not _sha256_digest(payload.get("roster_hash")):
            raise ValueError("reviewer roster lacks custody or hash evidence")
        return f"{reviewer_roster} reviewers={len(roster)}"

    results.append(_gate("independent_blinded_reviewer_roster", reviewers))

    mechanism_path = mechanism_plan or root / "mechanism" / "frozen_plan.json"

    protocol_path = (
        research_protocol or root / "research_ledger" / "frozen_protocol.json"
    )

    def protocol() -> str:
        receipt = load_frozen_protocol(protocol_path)
        receipt.verify()
        spec = load_executable_spec()
        if not receipt.protocol.scientific:
            raise ValueError("research protocol is a toy/non-scientific fixture")
        comparisons = {
            "study_id": (spec.study_id, receipt.protocol.study_id),
            "code_sha256": (spec.code_hash, receipt.protocol.code_sha256),
            "config_sha256": (
                spec.common_config_hash,
                receipt.protocol.config_sha256,
            ),
            "environment_sha256": (
                spec.environment_hash,
                receipt.protocol.environment_sha256,
            ),
            "pi_decision_sha256": (
                decisions_hash,
                receipt.protocol.pi_decision_sha256,
            ),
        }
        mismatches = {
            name: {"expected": expected, "observed": observed}
            for name, (expected, observed) in comparisons.items()
            if observed != expected
        }
        if mismatches:
            raise ValueError(f"research protocol cross-link mismatch: {mismatches}")
        return f"{protocol_path} protocol_hash={receipt.protocol_sha256}"

    results.append(_gate("frozen_research_protocol", protocol))

    def mechanism() -> str:
        receipt = load_frozen_mechanism_plan(mechanism_path)
        if not receipt.plan.scientific:
            raise ValueError("mechanism plan is a toy/non-scientific fixture")
        receipt.verify()
        spec = load_executable_spec()
        if receipt.plan.study_id != spec.study_id:
            raise ValueError("mechanism plan belongs to a different study")
        return f"{mechanism_path} plan_hash={receipt.plan_hash}"

    results.append(_gate("frozen_mechanism_plan", mechanism))

    replication_path = replication_policy or root / "replication" / "frozen_policy.json"

    def replication() -> str:
        receipt = load_frozen_replication_policy(replication_path)
        if not receipt.policy.scientific:
            raise ValueError("replication policy is a toy/non-scientific fixture")
        receipt.verify()
        spec = load_executable_spec()
        mechanism_receipt = load_frozen_mechanism_plan(mechanism_path)
        mechanism_receipt.verify()
        comparisons = {
            "study_id": (spec.study_id, receipt.policy.study_id),
            "claim_id": (
                mechanism_receipt.plan.claim.claim_id,
                receipt.policy.claim_id,
            ),
            "snapshot_id": (
                mechanism_receipt.plan.frozen_snapshot_id,
                receipt.policy.frozen_snapshot_id,
            ),
            "snapshot_sha256": (
                mechanism_receipt.plan.frozen_snapshot_sha256,
                receipt.policy.frozen_snapshot_sha256,
            ),
        }
        mismatches = {
            name: {"expected": expected, "observed": observed}
            for name, (expected, observed) in comparisons.items()
            if observed != expected
        }
        if mismatches:
            raise ValueError(f"replication cross-link mismatch: {mismatches}")
        return f"{replication_path} policy_hash={receipt.policy_hash}"

    results.append(_gate("frozen_replication_policy", replication))

    analysis_path = analysis_plan or root / "analysis" / "frozen_analysis_plan.json"

    def analysis() -> str:
        frozen = load_frozen_analysis_plan(analysis_path)
        if not frozen.plan.scientific:
            raise ValueError("analysis plan is a toy/non-scientific fixture")
        spec = load_executable_spec()
        if frozen.plan.study_id != spec.study_id:
            raise ValueError("analysis plan belongs to a different study")
        if frozen.plan.pi_decision_record_hash != decisions_hash:
            raise ValueError("analysis plan is not bound to the decision ledger")
        return f"{analysis_path} plan_hash={frozen.plan_hash}"

    results.append(_gate("frozen_analysis_and_power_plan", analysis))

    evidence_path = (
        mps_evidence
        or root / "outputs" / "readiness" / "full_train_v1_mps_evidence.json"
    )

    def mps_validation() -> str:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        if payload.get("schema_name") != "FullProfileMPSValidationEvidence":
            raise ValueError("MPS evidence has the wrong schema")
        required_fields = {
            "schema_name",
            "schema_version",
            "recorded_at_utc",
            "profile_name",
            "profile_version",
            "profile_hash",
            "requested_device",
            "selected_device",
            "mps_available",
            "cpu_fallback",
            "success",
            "steps_completed",
            "candidate_source_hash",
            "training_manifest_hash",
            "training_summary_hash",
            "training_output_dir",
        }
        if set(payload) != required_fields:
            raise ValueError("MPS evidence fields differ from the frozen v1.0 schema")
        for field in ("mps_available", "cpu_fallback", "success"):
            if type(payload[field]) is not bool:
                raise ValueError(f"MPS evidence field {field} must be boolean")
        if not isinstance(payload["steps_completed"], int) or isinstance(
            payload["steps_completed"], bool
        ):
            raise ValueError("MPS evidence steps_completed must be an integer")
        if not _valid_utc_timestamp(payload["recorded_at_utc"]):
            raise ValueError("MPS evidence recorded_at_utc is invalid")
        expected = {
            "schema_version": "1.0",
            "profile_name": FULL_TRAIN_V1.name,
            "profile_version": FULL_TRAIN_V1.version,
            "profile_hash": FULL_TRAIN_V1.profile_hash,
            "requested_device": "mps",
            "selected_device": "mps",
            "mps_available": True,
            "cpu_fallback": False,
            "success": True,
            "steps_completed": FULL_TRAIN_V1.max_steps,
        }
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(f"MPS evidence mismatch: {mismatches}")
        for field in (
            "candidate_source_hash",
            "training_manifest_hash",
            "training_summary_hash",
        ):
            if not _sha256_digest(payload.get(field)):
                raise ValueError(f"MPS evidence field {field} is not a SHA-256")
        output_dir_value = payload["training_output_dir"]
        if not isinstance(output_dir_value, str) or not output_dir_value:
            raise ValueError("MPS evidence training_output_dir must be non-empty text")
        output_dir = Path(output_dir_value).resolve()
        manifest_path = output_dir / "training_manifest.json"
        summary_path = output_dir / "training_summary.json"
        for path in (manifest_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate_format = manifest.get("candidate_format", "arbitrary_python")
        if candidate_format not in {"architecture_ir", "arbitrary_python"}:
            raise ValueError("MPS evidence manifest has unsupported candidate_format")
        candidate_path = output_dir / (
            "candidate_graph.json"
            if candidate_format == "architecture_ir"
            else "candidate_source.py"
        )
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        observed_hashes = {
            "training_manifest_hash": _sha256_file(manifest_path),
            "training_summary_hash": _sha256_file(summary_path),
            "candidate_source_hash": _sha256_file(candidate_path),
        }
        hash_mismatches = {
            field: {"expected": payload[field], "observed": observed}
            for field, observed in observed_hashes.items()
            if payload[field] != observed
        }
        if hash_mismatches:
            raise ValueError(f"MPS artifact hash mismatch: {hash_mismatches}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        exact_boolean_summary = {
            "success": True,
            "scientific": True,
            "hardware_matched": True,
            "unsupported_operation_fallback": False,
            "cleanup_completed": True,
        }
        for field, expected_value in exact_boolean_summary.items():
            if type(summary.get(field)) is not bool or summary[field] is not expected_value:
                raise ValueError(f"training summary field {field} is invalid")
        summary_expected = {
            "profile_name": FULL_TRAIN_V1.name,
            "profile_version": FULL_TRAIN_V1.version,
            "profile_hash": FULL_TRAIN_V1.profile_hash,
            "device": "mps",
            "steps_completed": FULL_TRAIN_V1.max_steps,
            "candidate_source_hash": payload["candidate_source_hash"],
        }
        for field, expected_value in summary_expected.items():
            observed = summary.get(field)
            if field == "steps_completed" and (
                not isinstance(observed, int) or isinstance(observed, bool)
            ):
                raise ValueError("training summary steps_completed must be an integer")
            if observed != expected_value:
                raise ValueError(
                    f"training summary field {field} does not match full_train_v1"
                )
        return f"{evidence_path} summary={payload['training_summary_hash']}"

    historical_mps_gate_name: str | None = None
    if not active_accelerator_manifest or mps_evidence is not None:
        historical_mps_gate_name = "full_profile_mps_validation"
        results.append(_gate(historical_mps_gate_name, mps_validation))
    else:
        historical_mps_gate_name = "historical_mps_evidence_compatibility"
        recorder = root / "scripts" / "record_mps_validation.py"
        results.append(
            GateResult(
                historical_mps_gate_name,
                recorder.is_file(),
                str(recorder),
                ()
                if recorder.is_file()
                else ("historical MPS evidence reader/recorder is missing",),
            )
        )

    accelerator_evidence_gate_name: str | None = None
    if active_accelerator_manifest:
        accelerator_evidence_gate_name = "full_profile_accelerator_validation"
        active_evidence_path = (
            accelerator_evidence
            or root
            / "outputs"
            / "readiness"
            / "full_train_cuda_v2_accelerator_evidence.json"
        )

        def accelerator_validation() -> str:
            payload = validate_accelerator_validation_evidence(
                active_evidence_path
            )
            expected = {
                "training_profile_name": FULL_TRAIN_CUDA_V2.name,
                "training_profile_version": FULL_TRAIN_CUDA_V2.version,
                "training_profile_hash": FULL_TRAIN_CUDA_V2.profile_hash,
                "execution_backend": "modal",
                "requested_gpu_kind": "cuda",
                "observed_gpu_kind": "cuda",
                "success": True,
                "scientific": True,
                "hardware_matched": True,
                "unsupported_operation_fallback": False,
                "cleanup_completed": True,
                "steps_completed": FULL_TRAIN_CUDA_V2.max_steps,
            }
            mismatches = {
                field: {"expected": value, "observed": payload.get(field)}
                for field, value in expected.items()
                if payload.get(field) != value
                or (type(value) is bool and type(payload.get(field)) is not bool)
            }
            if mismatches:
                raise ValueError(
                    f"accelerator validation evidence mismatch: {mismatches}"
                )
            manifest_image_hash = manifest.get("remote_execution", {}).get(
                "image_source_sha256"
            )
            if (
                manifest_image_hash is not None
                and manifest_image_hash != payload["image_source_hash"]
            ):
                raise ValueError(
                    "accelerator evidence image source differs from the manifest"
                )
            return (
                f"{active_evidence_path} summary="
                f"{payload['training_summary_hash']} gpu="
                f"{payload['observed_gpu_name']}"
            )

        results.append(
            _gate(accelerator_evidence_gate_name, accelerator_validation)
        )
    results.append(
        GateResult(
            "artifact_reconstruction_implementation",
            all(
                path.is_file()
                for path in (
                    root / "artifacts" / "records.py",
                    root / "artifacts" / "store.py",
                    root / "artifacts" / "index.py",
                    root / "artifacts" / "study_sink.py",
                    root / "reconstruction" / "rebuild.py",
                    root / "reconstruction" / "tables.py",
                )
            ),
            "artifacts/ and reconstruction/",
            ()
            if all(
                path.is_file()
                for path in (
                    root / "artifacts" / "records.py",
                    root / "artifacts" / "store.py",
                    root / "artifacts" / "index.py",
                    root / "artifacts" / "study_sink.py",
                    root / "reconstruction" / "rebuild.py",
                    root / "reconstruction" / "tables.py",
                )
            )
            else ("immutable event and reconstruction packages are incomplete",),
        )
    )

    def artifact_integration() -> str:
        for path in (
            root / "scripts" / "study_offline_smoke.py",
            root / "scripts" / "study_scientific_run.py",
        ):
            source = path.read_text(encoding="utf-8")
            if "ArtifactEmittingStudyEngine" not in source:
                raise ValueError(f"{path.name} bypasses the immutable event sink")
        if not (root / "tests" / "test_artifact_study_sink.py").is_file():
            raise FileNotFoundError("missing artifact study-sink regression tests")
        return "offline and gated scientific entrypoints use ArtifactEmittingStudyEngine"

    results.append(_gate("artifact_ledger_integrated", artifact_integration))
    results.append(
        GateResult(
            "reproducibility_reporting_implementation",
            all(
                path.is_file()
                for path in (
                    root / "research_ledger" / "ledger.py",
                    root / "research_ledger" / "protocol.py",
                    root / "reporting" / "report.py",
                    root / "reporting" / "synthetic.py",
                )
            ),
            "research_ledger/ and reporting/",
            ()
            if all(
                path.is_file()
                for path in (
                    root / "research_ledger" / "ledger.py",
                    root / "research_ledger" / "protocol.py",
                    root / "reporting" / "report.py",
                    root / "reporting" / "synthetic.py",
                )
            )
            else ("research ledger and reporting packages are incomplete",),
        )
    )

    anchor_path = root / "outputs" / "readiness" / "external_anchor_receipt.json"

    def external_anchor() -> str:
        payload = json.loads(anchor_path.read_text(encoding="utf-8"))
        if payload.get("schema_name") != "ExternalIntegrityAnchorReceipt":
            raise ValueError("external anchor has the wrong schema")
        if payload.get("schema_version") != "1.0":
            raise ValueError("external anchor has an unsupported schema version")
        if payload.get("immutable_retention") is not True:
            raise ValueError("external anchor does not attest immutable retention")
        for field in (
            "randomization_sha256",
            "event_chain_head_sha256",
            "artifact_index_sha256",
        ):
            if not _sha256_digest(payload.get(field)):
                raise ValueError(f"external anchor field {field} is invalid")
        if not payload.get("anchor_provider") or not payload.get("receipt_id"):
            raise ValueError("external anchor lacks provider custody metadata")
        return f"{anchor_path} receipt={payload['receipt_id']}"

    results.append(_gate("external_integrity_anchor", external_anchor))

    def authenticated_external_anchor() -> str:
        verifier_path = root / "artifacts" / "external_anchor.py"
        if not verifier_path.is_file():
            raise FileNotFoundError(
                "cryptographic external-anchor verifier is not implemented"
            )
        policy = decisions.get("scope_and_release", {}).get(
            "artifact_anchor_provider_and_signature_scheme"
        )
        if policy is None:
            raise ValueError("external-anchor signature policy is unresolved")
        from artifacts.external_anchor import verify_external_anchor

        verification = verify_external_anchor(
            receipt_path=anchor_path,
            policy=policy,
            project_root=root,
        )
        if not isinstance(verification, dict) or verification.get("verified") is not True:
            raise ValueError("external custodian signature was not verified")
        if type(verification.get("verified")) is not bool:
            raise ValueError("external verification result must be boolean")
        if not _sha256_digest(verification.get("receipt_sha256")):
            raise ValueError("external verification lacks the receipt digest")
        if verification["receipt_sha256"] != _sha256_file(anchor_path):
            raise ValueError("verified external receipt digest does not match")
        return (
            f"{anchor_path} cryptographically verified under policy={policy}"
        )

    results.append(
        _gate("externally_authenticated_anchor", authenticated_external_anchor)
    )

    pilot_evidence_path = root / "outputs" / "readiness" / "pilot_evidence.json"

    def pilot_evidence() -> str:
        payload = json.loads(pilot_evidence_path.read_text(encoding="utf-8"))
        if payload.get("schema_name") != "ScientificPilotEvidence":
            raise ValueError("pilot evidence has the wrong schema")
        required_fields = {
            "schema_name",
            "schema_version",
            "recorded_at_utc",
            "completed",
            "study_id",
            "study_spec_hash",
            "decision_ledger_sha256",
            "manifest_sha256",
            "assignment_path",
            "assignment_sha256",
            "artifact_index_manifest_path",
            "artifact_index_manifest_sha256",
            "assigned_run_count",
            "condition_run_counts",
            "provider_model",
            "training_profile_hash",
        }
        if set(payload) != required_fields:
            raise ValueError("pilot evidence fields differ from the frozen v2.0 schema")
        if payload.get("schema_version") != "2.0" or payload.get("completed") is not True:
            raise ValueError("pilot evidence is incomplete")
        if type(payload["completed"]) is not bool:
            raise ValueError("pilot completed flag must be boolean")
        if not _valid_utc_timestamp(payload["recorded_at_utc"]):
            raise ValueError("pilot evidence timestamp is invalid")
        assigned_count = payload["assigned_run_count"]
        if not isinstance(assigned_count, int) or isinstance(assigned_count, bool):
            raise ValueError("assigned_run_count must be an integer")
        counts = payload["condition_run_counts"]
        if not isinstance(counts, dict) or set(counts) != {
            "C0",
            "C1",
            "C2",
            "C3",
            "NO_SEARCH",
        }:
            raise ValueError("pilot evidence lacks the C0-C3 plus no-search roster")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in counts.values()
        ):
            raise ValueError("pilot condition counts must be positive integers")
        if len(set(counts.values())) != 1:
            raise ValueError("pilot condition run counts are not matched")
        if assigned_count != sum(counts.values()):
            raise ValueError("pilot assigned-run count does not reconstruct")
        spec = load_executable_spec()
        expected_links = {
            "study_id": spec.study_id,
            "study_spec_hash": spec.spec_hash,
            "decision_ledger_sha256": decisions_hash,
            "manifest_sha256": manifest_hash,
            "provider_model": "gpt-5.6-sol",
            "training_profile_hash": active_training_profile.profile_hash,
        }
        link_mismatches = {
            field: {"expected": expected, "observed": payload.get(field)}
            for field, expected in expected_links.items()
            if payload.get(field) != expected
        }
        if link_mismatches:
            raise ValueError(f"pilot evidence cross-link mismatch: {link_mismatches}")
        for field in ("assignment_sha256", "artifact_index_manifest_sha256"):
            if not _sha256_digest(payload.get(field)):
                raise ValueError(f"pilot evidence field {field} is invalid")
        assignment_path_value = payload["assignment_path"]
        artifact_path_value = payload["artifact_index_manifest_path"]
        if not isinstance(assignment_path_value, str) or not isinstance(
            artifact_path_value, str
        ):
            raise ValueError("pilot artifact paths must be strings")
        assignment_path = Path(assignment_path_value).resolve()
        artifact_path = Path(artifact_path_value).resolve()
        if _sha256_file(assignment_path) != payload["assignment_sha256"]:
            raise ValueError("pilot randomization artifact hash mismatch")
        if _sha256_file(artifact_path) != payload["artifact_index_manifest_sha256"]:
            raise ValueError("pilot artifact-index manifest hash mismatch")
        plan = RandomizationPlan.from_dict(read_json(assignment_path))
        if plan.study_id != spec.study_id or plan.study_spec_hash != spec.spec_hash:
            raise ValueError("pilot randomization is not bound to the StudySpec")
        artifact_manifest = read_json(artifact_path)
        if artifact_manifest.get("schema_name") != "StudyArtifactIndexManifest":
            raise ValueError("pilot artifact-index manifest has the wrong schema")
        if artifact_manifest.get("study_id") != spec.study_id:
            raise ValueError("pilot artifact-index manifest belongs to another study")
        if artifact_manifest.get("assignment_hash") != plan.assignment_hash:
            raise ValueError("pilot artifact indexes are not bound to randomization")
        indexed_runs = artifact_manifest.get("run_indexes")
        if not isinstance(indexed_runs, list) or len(indexed_runs) != sum(
            counts[name] for name in ("C0", "C1", "C2", "C3")
        ):
            raise ValueError("pilot primary-run artifact indexes are incomplete")
        return f"{pilot_evidence_path} runs={assigned_count}"

    results.append(_gate("scientific_pilot_completed", pilot_evidence))

    def recorded_readiness_level(level_name: str) -> str:
        levels = evidence_ledger.get("levels")
        if not isinstance(levels, dict):
            raise ValueError("readiness evidence lacks level records")
        record = levels.get(level_name)
        active_local_contract = (
            evidence_ledger.get("schema_version") in {"3", "4"}
            and level_name in LOCAL_ENGINEERING_RECEIPT_CONTRACTS
        )
        expected_fields = {"passed", "evidence"}
        if active_local_contract:
            expected_fields |= {"receipt_path", "receipt_contract"}
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError(
                f"readiness level {level_name} has an invalid exact schema"
            )
        if type(record["passed"]) is not bool or record["passed"] is not True:
            raise ValueError(f"readiness level {level_name} is not passed")
        if not isinstance(record["evidence"], str) or not record["evidence"].strip():
            raise ValueError(f"readiness level {level_name} lacks evidence")
        provenance = evidence_ledger.get("engineering_evidence_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("engineering evidence lacks provenance")
        if provenance.get("authority") != "local_self_report":
            raise ValueError("engineering evidence authority is invalid")
        if provenance.get("source_revision_bound") is not True:
            raise ValueError("engineering evidence is not bound to a source revision")
        if active_local_contract:
            contract = LOCAL_ENGINEERING_RECEIPT_CONTRACTS[level_name]
            if record["receipt_contract"] != contract["receipt_contract"]:
                raise ValueError(
                    f"readiness level {level_name} receipt contract drifted"
                )
            expected_receipt = current_local_engineering_receipt_path(
                level_name,
                root=root,
            ).as_posix()
            if record["receipt_path"] != expected_receipt:
                if record["receipt_path"] is None:
                    raise FileNotFoundError(
                        f"readiness level {level_name} receipt path is pending"
                    )
                raise ValueError(
                    f"readiness level {level_name} receipt path drifted"
                )
            receipt_path = root / expected_receipt
            return validate_local_engineering_receipt(
                level_name,
                receipt_path,
                root=root,
            )
        # Preserve historical schema-v1/v2 semantics for old readiness ledgers.
        if provenance.get("externally_attested") is not True:
            raise ValueError("engineering evidence is not externally attested")
        return record["evidence"]

    results.append(
        _gate(
            "local_engineering_freeze_validated",
            lambda: (
                f"{validate_local_engineering_freeze_receipt(root=root)}"
            ),
        )
    )

    results.append(
        _gate(
            "recorded_unit_test_evidence",
            lambda: recorded_readiness_level("unit_tested"),
        )
    )
    results.append(
        _gate(
            "recorded_offline_smoke_evidence",
            lambda: recorded_readiness_level("offline_smoke_tested"),
        )
    )

    if active_accelerator_manifest:
        def external_scientific_attestation() -> str:
            provenance = evidence_ledger.get("engineering_evidence_provenance")
            if not isinstance(provenance, dict) or set(provenance) != {
                "authority",
                "source_revision_bound",
                "externally_attested",
                "scientific_launch_authority",
            }:
                raise ValueError("external evidence provenance schema is invalid")
            if type(provenance["externally_attested"]) is not bool:
                raise ValueError("externally_attested must be boolean")
            if provenance["externally_attested"] is not True:
                raise ValueError("scientific evidence lacks external attestation")
            if type(provenance["scientific_launch_authority"]) is not bool:
                raise ValueError("scientific_launch_authority must be boolean")
            if provenance["scientific_launch_authority"] is not True:
                raise ValueError("external attestation has no launch authority")
            return "external scientific evidence attestation and authority recorded"

        results.append(
            _gate(
                "external_scientific_evidence_attestation",
                external_scientific_attestation,
            )
        )

    by_name = {result.gate: result for result in results}
    main_only_gates = {
        "frozen_analysis_and_power_plan",
        "scientific_pilot_completed",
        "main_launch_authorization",
    }
    compatibility_only_gates = (
        {historical_mps_gate_name}
        if active_accelerator_manifest and historical_mps_gate_name is not None
        else set()
    )
    pilot_ready = all(
        result.passed
        for result in results
        if result.gate not in main_only_gates | compatibility_only_gates
    )
    main_ready = all(
        result.passed
        for result in results
        if result.gate not in compatibility_only_gates
    )
    passed_count = sum(result.passed for result in results)
    readiness_levels = {
        "infrastructure_implemented": all(
            by_name[name].passed
            for name in (
                "common_c0_c3_engine",
                "evaluation_and_novelty_firewall",
                "artifact_reconstruction_implementation",
                "artifact_ledger_integrated",
                "reproducibility_reporting_implementation",
                *(
                    ("modal_cuda_execution_configured",)
                    if active_accelerator_manifest
                    else ()
                ),
            )
        ),
        "unit_tested": by_name["recorded_unit_test_evidence"].passed,
        "offline_smoke_tested": by_name[
            "recorded_offline_smoke_evidence"
        ].passed,
        "accelerator_validated": (
            by_name[accelerator_evidence_gate_name].passed
            if accelerator_evidence_gate_name is not None
            else False
        ),
        "pilot_ready": pilot_ready,
        "pilot_validated": by_name["scientific_pilot_completed"].passed,
        "main_study_ready": main_ready,
    }
    if active_accelerator_manifest:
        readiness_levels["modal_infrastructure_validated"] = all(
            by_name[gate_name].passed
            for gate_name in MODAL_READINESS_RECEIPT_CONTRACTS
        )
    else:
        # Preserve the version-1/version-2 report reader exactly.  Schema v3
        # intentionally drops this stale active-readiness label and retains MPS
        # only as historical compatibility evidence.
        readiness_levels["mps_validated"] = (
            by_name["full_profile_mps_validation"].passed
            if "full_profile_mps_validation" in by_name
            else False
        )
    payload = {
        "schema_name": "ScientificReadinessAudit",
        "schema_version": "1.0",
        "ready": main_ready,
        "pilot_ready": pilot_ready,
        "main_study_ready": main_ready,
        "readiness_levels": readiness_levels,
        "decision_ledger_sha256": decisions_hash,
        "passed_gate_count": passed_count,
        "total_gate_count": len(results),
        "provider_calls": 0,
        "training_runs": 0,
        "gates": [asdict(result) for result in results],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit scientific readiness without provider or training calls."
    )
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--corpus-seal", type=Path)
    parser.add_argument("--analysis-plan", type=Path)
    parser.add_argument("--mechanism-plan", type=Path)
    parser.add_argument("--replication-policy", type=Path)
    parser.add_argument("--research-protocol", type=Path)
    parser.add_argument("--mps-evidence", type=Path)
    parser.add_argument("--accelerator-evidence", type=Path)
    parser.add_argument("--study-spec", type=Path)
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args()
    report = audit_readiness(
        corpus_manifest=arguments.corpus_manifest,
        corpus_seal=arguments.corpus_seal,
        analysis_plan=arguments.analysis_plan,
        mechanism_plan=arguments.mechanism_plan,
        replication_policy=arguments.replication_policy,
        research_protocol=arguments.research_protocol,
        mps_evidence=arguments.mps_evidence,
        accelerator_evidence=arguments.accelerator_evidence,
        study_spec=arguments.study_spec,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
