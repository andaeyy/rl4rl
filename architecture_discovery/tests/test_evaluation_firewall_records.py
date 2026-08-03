from dataclasses import fields
from pathlib import Path

import pytest

from common.evaluation_profiles import EvaluationLayer, resolve_evaluation_plan
from evaluation.artifacts import EvaluationArtifactRoots, JsonEvaluationArtifactStore
from evaluation.firewall import ControllerEvaluationInbox, LayerBoundaryError
from evaluation.records import (
    CONTROLLER_SEARCH_FIELDS,
    QualificationEvaluationRecord,
    RecordEnvelope,
    SearchEvaluationRecord,
)


def _hash(character: str = "a") -> str:
    return character * 64


def _envelope(schema: str, *, record_id: str) -> RecordEnvelope:
    return RecordEnvelope.create(
        schema_name=schema,
        record_id=record_id,
        study_id="study-1",
        block_id="block-1",
        run_id="run-1",
        condition_id="C0",
        writer_component="test-suite",
        code_sha256=_hash("a"),
        config_sha256=_hash("b"),
        environment_sha256=_hash("c"),
    )


def _search_record(*, score: float = 0.75) -> SearchEvaluationRecord:
    return SearchEvaluationRecord(
        envelope=_envelope("search_evaluation", record_id="search-1"),
        candidate_id="candidate-1",
        training_record_id="training-1",
        execution_ok=True,
        transformer_valid=True,
        public_accuracy=0.75,
        search_score=score,
        eligible_for_parent=True,
        parameter_count_metadata=10_000_000,
        online_descriptor_codes=(("attention_organization", 2.0),),
    )


def _qualification_record(*, accuracy: float) -> QualificationEvaluationRecord:
    return QualificationEvaluationRecord(
        envelope=_envelope("qualification_evaluation", record_id="sealed-b-1"),
        candidate_id="candidate-1",
        frozen_snapshot_id="snapshot-1",
        frozen_snapshot_sha256=_hash("d"),
        evaluation_plan_sha256=_hash("e"),
        exact_match_accuracy=accuracy,
        qualifies=accuracy >= 0.99,
        evaluation_complete=True,
        sealed_metrics=(("private_shift_accuracy", accuracy),),
    )


def test_controller_view_is_an_exact_allowlist_without_sealed_or_size_fields():
    record = _search_record()
    view = record.controller_view()
    assert {field.name for field in fields(view)} == CONTROLLER_SEARCH_FIELDS
    assert set(view.as_dict()) == CONTROLLER_SEARCH_FIELDS
    forbidden = {
        "sealed_metrics",
        "qualifies",
        "shadow_accuracy",
        "edge_accuracy",
        "carry_accuracy",
        "parameter_count_metadata",
        "public_artifacts",
    }
    assert forbidden.isdisjoint(view.as_dict())


def test_controller_inbox_rejects_generic_dicts_and_layer_b_records():
    inbox = ControllerEvaluationInbox()
    inbox.publish(_search_record())
    with pytest.raises(LayerBoundaryError, match="only exact"):
        inbox.publish(_search_record().to_dict())
    with pytest.raises(LayerBoundaryError, match="only exact"):
        inbox.publish(_qualification_record(accuracy=1.0))
    assert len(inbox.views) == 1


def test_synthetic_layer_b_outcome_cannot_change_continued_fake_search():
    """The fake controller hashes only its typed Layer A inbox."""

    def next_proposal(inbox: ControllerEvaluationInbox) -> str:
        import hashlib
        import json

        payload = [dict(view.as_dict()) for view in inbox.views]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    inbox = ControllerEvaluationInbox()
    inbox.publish(_search_record())
    proposal_before_b = next_proposal(inbox)
    high_b = _qualification_record(accuracy=1.0)
    low_b = _qualification_record(accuracy=0.0)

    for sealed_result in (high_b, low_b):
        with pytest.raises(LayerBoundaryError):
            inbox.publish(sealed_result)
        assert next_proposal(inbox) == proposal_before_b


def test_evaluation_artifacts_use_distinct_roots_and_layer_typed_refs(tmp_path):
    roots = EvaluationArtifactRoots.under(tmp_path / "evaluations")
    roots.prepare()
    assert len({roots.layer_a, roots.layer_b, roots.layer_c}) == 3
    stores = {
        layer: JsonEvaluationArtifactStore(roots, layer)
        for layer in EvaluationLayer
    }
    references = {
        layer: store.write_json(f"record-{layer.value}", {"layer": layer.value})
        for layer, store in stores.items()
    }
    for layer, reference in references.items():
        assert stores[layer].read_json(reference) == {"layer": layer.value}
        other = next(candidate for candidate in EvaluationLayer if candidate is not layer)
        with pytest.raises(ValueError, match="expected"):
            stores[other].read_json(reference)
    with pytest.raises(FileExistsError, match="already exists"):
        stores[EvaluationLayer.SEARCH].write_json(
            "record-layer_a", {"layer": "silently-replaced"}
        )


def test_synthetic_layer_b_plan_is_never_controller_visible():
    plan = resolve_evaluation_plan(
        "unit_eval_v1",
        layer=EvaluationLayer.QUALIFICATION,
        case_source_id="synthetic-b-fixture",
        case_source_sha256=_hash("f"),
    )
    assert plan.synthetic
    assert plan.sealed
    assert not plan.controller_visible
