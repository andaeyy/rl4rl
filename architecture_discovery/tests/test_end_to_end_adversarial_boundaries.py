from __future__ import annotations

from pathlib import Path

import pytest

from common.evaluation_profiles import EvaluationLayer, resolve_evaluation_plan
from evaluation.dependency_audit import assert_controller_dependencies_clean
from evaluation.records import (
    CONTROLLER_SEARCH_FIELDS,
    RecordEnvelope,
    SearchEvaluationRecord,
    search_evaluation_from_dict,
)
from novelty.dependency_audit import audit_science_boundary


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


def _search_record() -> SearchEvaluationRecord:
    return SearchEvaluationRecord(
        envelope=RecordEnvelope.create(
            schema_name="search_evaluation",
            study_id="study",
            block_id="block",
            run_id="run",
            condition_id="C0",
            writer_component="adversarial-test",
            code_sha256=SHA,
            config_sha256=SHA,
            environment_sha256=SHA,
        ),
        candidate_id="candidate",
        training_record_id="training",
        execution_ok=True,
        transformer_valid=True,
        public_accuracy=0.8,
        search_score=0.8,
        eligible_for_parent=True,
    )


def test_primary_controller_graph_has_no_sealed_or_postsearch_route() -> None:
    entries = (
        ROOT / "agents" / "greedy_autoresearch" / "run.py",
        ROOT / "common" / "openevolve_runner.py",
        ROOT / "study" / "runtime_adapters.py",
        ROOT / "scripts" / "study_offline_smoke.py",
    )
    assert_controller_dependencies_clean(entries, project_root=ROOT)
    assert audit_science_boundary(ROOT) == ()


def test_layer_a_view_rejects_shadow_or_sealed_field_injection() -> None:
    record = _search_record()
    view = dict(record.controller_view().as_dict())
    forbidden_tokens = (
        "shadow",
        "sealed",
        "qualification",
        "confirmation",
        "combined_score",
        "robustness",
    )
    assert set(view) == CONTROLLER_SEARCH_FIELDS
    assert not any(
        token in field.lower() for field in view for token in forbidden_tokens
    )

    injected = record.to_dict()
    injected["shadow_accuracy"] = 1.0
    with pytest.raises(ValueError, match="unexpected or missing fields"):
        search_evaluation_from_dict(injected)


def test_untrusted_layer_a_boolean_strings_are_rejected_not_coerced() -> None:
    payload = _search_record().to_dict()
    payload["execution_ok"] = "false"
    payload["transformer_valid"] = "false"
    payload["eligible_for_parent"] = "false"
    with pytest.raises((TypeError, ValueError), match="boolean"):
        search_evaluation_from_dict(payload)


def test_smoke_profile_cannot_be_relabelled_as_scientific_layer_b() -> None:
    with pytest.raises(ValueError, match="requires at least 10000 cases"):
        resolve_evaluation_plan(
            "scientific_layer_b_v1",
            layer=EvaluationLayer.QUALIFICATION,
            case_source_id="sealed-source",
            case_source_sha256=SHA,
            case_count=64,
            pi_decision_record_id="frozen-pi-decision",
        )
