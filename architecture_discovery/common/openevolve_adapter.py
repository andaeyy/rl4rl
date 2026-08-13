"""One adapter bridge shared by both OpenEvolve conditions."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

from architecture_ir.interpreter import validate_ir_candidate_path
from common.architecture_dedup import ArchitectureHashRegistry
from common.evaluator import (
    SearchEvaluationContext,
    evaluate_candidate,
    file_hash,
    validate_controller_view_binding,
)
from common.descriptor_schema import SEMANTIC_METRIC_NAMES
from common.openevolve_policy import canonical_combined_score


ROOT = Path(__file__).resolve().parents[1]
_PARENT_CHANGE_ENFORCEMENT_ENV = "DISCOVERY_ENFORCE_PARENT_ARCHITECTURE_CHANGE"
_PARENT_ARCHITECTURE_HASH_ENV = "DISCOVERY_PARENT_ARCHITECTURE_HASH"
_INITIAL_ARCHITECTURE_HASH_ENV = "DISCOVERY_INITIAL_ARCHITECTURE_HASH"
_INITIAL_EVALUATION_AUTH_ENV = "DISCOVERY_OPENEVOLVE_INITIAL_EVALUATION_AUTH"
_ARCHITECTURE_REGISTRY_ENV = "DISCOVERY_ARCHITECTURE_HASH_REGISTRY"
_SHA256_HEX_LENGTH = 64


def _validated_architecture_hash_environment(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    if len(value) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_parent_relative_architecture_change(graph) -> str | None:
    """Reject executable no-ops before the shared evaluator can train them.

    Initial-program evaluation has no parent.  In that one case the graph must
    match the runner-bound initial architecture.  Every evolved child must
    carry a trusted parent hash supplied by the worker wrapper.
    """

    enforcement = os.environ.get(_PARENT_CHANGE_ENFORCEMENT_ENV, "0")
    if enforcement not in {"0", "1"}:
        raise ValueError(
            f"{_PARENT_CHANGE_ENFORCEMENT_ENV} must be exactly '0' or '1'"
        )
    if enforcement == "0":
        return None

    parent_hash = _validated_architecture_hash_environment(
        _PARENT_ARCHITECTURE_HASH_ENV
    )
    if parent_hash is not None:
        if graph.architecture_hash == parent_hash:
            raise ValueError(
                "OpenEvolve proposal is an executable architecture no-op: "
                "its architecture_hash is unchanged from its selected parent"
            )
        return parent_hash

    initial_hash = _validated_architecture_hash_environment(
        _INITIAL_ARCHITECTURE_HASH_ENV
    )
    if initial_hash is None:
        raise ValueError(
            "OpenEvolve parent architecture binding is missing; refusing "
            "candidate evaluation"
        )
    if graph.architecture_hash != initial_hash:
        raise ValueError(
            "OpenEvolve parent architecture binding is missing for a "
            "non-initial candidate; refusing candidate evaluation"
        )
    initial_authorization = os.environ.pop(_INITIAL_EVALUATION_AUTH_ENV, None)
    if initial_authorization != initial_hash:
        raise ValueError(
            "OpenEvolve one-time initial evaluation authorization is missing; "
            "refusing unbound candidate evaluation"
        )
    return None


def _load_strict_ir_candidate(program_path: str | Path):
    """Reject every non-IR artifact before the shared evaluator is entered."""

    candidate = Path(program_path).resolve()
    if candidate.suffix.lower() != ".json":
        raise ValueError(
            "OpenEvolve accepts only declarative Architecture IR .json "
            "candidates; generated Python is never executed"
        )
    validation = validate_ir_candidate_path(candidate)
    if not validation.valid:
        reasons = "; ".join(
            f"{issue.code}: {issue.message}" for issue in validation.issues
        )
        raise ValueError(f"invalid Architecture IR candidate: {reasons}")
    if validation.graph is None:  # pragma: no cover - validation invariant
        raise ValueError("invalid Architecture IR candidate: decoded graph is absent")
    return validation.graph


def evaluate_for_openevolve(program_path: str) -> EvaluationResult:
    graph = _load_strict_ir_candidate(program_path)
    parent_architecture_hash = _require_parent_relative_architecture_change(graph)
    if parent_architecture_hash is not None:
        registry_path = os.environ.get(_ARCHITECTURE_REGISTRY_ENV)
        if not registry_path:
            raise ValueError(
                "OpenEvolve run-wide architecture registry is missing; refusing "
                "candidate evaluation"
            )
        registry = ArchitectureHashRegistry(registry_path)
        if not registry.claim(graph.architecture_hash):
            raise ValueError(
                "OpenEvolve proposal duplicates an architecture already proposed "
                "or evaluated in this run"
            )
    profile = os.environ.get("DISCOVERY_TRAINING_PROFILE", "full_train_cuda_v2")
    run_seed = int(os.environ.get("DISCOVERY_TRAINING_SEED", "1"))
    device = os.environ.get("DISCOVERY_TRAIN_DEVICE", "cuda")
    allow_cpu = os.environ.get("DISCOVERY_ALLOW_CPU_TRAINING", "0") == "1"
    eligibility_threshold = float(
        os.environ.get("DISCOVERY_ELIGIBILITY_THRESHOLD", "0.99")
    )
    if not 0.0 <= eligibility_threshold <= 1.0:
        raise ValueError("DISCOVERY_ELIGIBILITY_THRESHOLD must be in [0, 1]")
    training_root = Path(
        os.environ.get(
            "DISCOVERY_TRAINING_OUTPUT_ROOT",
            str(ROOT / "outputs" / "candidate_training"),
        )
    ).resolve()
    source_hash = file_hash(program_path)
    identifier = f"{source_hash[:12]}_{uuid.uuid4().hex[:8]}"
    context = SearchEvaluationContext(
        study_id=os.environ.get("DISCOVERY_STUDY_ID", "native-replication"),
        block_id=os.environ.get("DISCOVERY_BLOCK_ID", "native-block"),
        run_id=os.environ.get("DISCOVERY_RUN_ID", "native-openevolve"),
        condition_id=os.environ.get(
            "DISCOVERY_CONDITION_ID", "native-openevolve"
        ),
    )
    result = evaluate_candidate(
        program_path,
        training_profile=profile,
        training_seed=run_seed,
        training_output_dir=training_root / identifier,
        device=device,
        allow_cpu_for_tests=allow_cpu,
        evaluation_profile=os.environ.get("DISCOVERY_LAYER_A_PROFILE"),
        evaluation_case_count=(
            int(os.environ["DISCOVERY_LAYER_A_CASES"])
            if os.environ.get("DISCOVERY_LAYER_A_CASES")
            else None
        ),
        pi_decision_record_id=os.environ.get(
            "DISCOVERY_SCIENTIFIC_DECISION_RECORD"
        ),
        eligibility_threshold=eligibility_threshold,
        context=context,
    )
    view = result.controller_view()
    validate_controller_view_binding(
        view,
        candidate_source_hash=source_hash,
        context=context,
    )
    semantic_metrics = {
        metric_name: 0.0 for metric_name in SEMANTIC_METRIC_NAMES.values()
    }
    semantic_metrics.update(dict(view.online_descriptor_codes))
    metrics = {
        "execution_ok": float(view.execution_ok),
        "transformer_valid": float(view.transformer_valid),
        "public_accuracy": view.public_accuracy,
        "search_score": view.search_score,
        "eligible_for_parent": float(view.eligible_for_parent),
        **semantic_metrics,
    }
    metrics["combined_score"] = canonical_combined_score(metrics)
    artifacts = {
        "layer_a_record_id": view.record_id,
        "candidate_graph_hash": graph.graph_hash,
        "candidate_architecture_hash": graph.architecture_hash,
        "parent_architecture_hash": parent_architecture_hash,
        "failure_stage": view.failure_stage,
        "infrastructure_failure": view.infrastructure_failure,
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
