from dataclasses import replace

import pytest

from common.evaluation_profiles import EvaluationLayer, resolve_evaluation_plan
from evaluation.artifacts import EvaluationArtifactRoots
from evaluation.records import RecordEnvelope, sha256_text
from sealed_eval.confirmation import (
    ConfirmationMeasurements,
    ConfirmationReleaseManifest,
    LayerCReleaseGate,
)
from sealed_eval.qualification import (
    LayerBQualificationRunner,
    QualificationMeasurements,
    QualificationPolicy,
)
from sealed_eval.snapshot import freeze_completed_run


def _hash(character: str = "a") -> str:
    return character * 64


def _envelope(schema: str, record_id: str) -> RecordEnvelope:
    return RecordEnvelope.create(
        schema_name=schema,
        record_id=record_id,
        study_id="study-1",
        block_id="block-1",
        run_id="run-1",
        condition_id="C2",
        writer_component="sealed-test-service",
        code_sha256=_hash("a"),
        config_sha256=_hash("b"),
        environment_sha256=_hash("c"),
    )


def _synthetic_plan(layer: EvaluationLayer):
    return resolve_evaluation_plan(
        "unit_eval_v1",
        layer=layer,
        case_source_id=f"synthetic-{layer.value}-fixture",
        case_source_sha256={
            EvaluationLayer.SEARCH: _hash("d"),
            EvaluationLayer.QUALIFICATION: _hash("e"),
            EvaluationLayer.CONFIRMATION: _hash("f"),
        }[layer],
    )


def _snapshot():
    return freeze_completed_run(
        snapshot_id="snapshot-1",
        run_id="run-1",
        budget_checkpoint_id="budget-final",
        terminal_event_sha256=_hash("1"),
        candidate_artifacts={"candidate-1": (_hash("2"), _hash("3"))},
        run_complete=True,
    )


def _qualification(snapshot=None):
    frozen = snapshot or _snapshot()
    runner = LayerBQualificationRunner(
        evaluation_plan=_synthetic_plan(EvaluationLayer.QUALIFICATION),
        policy=QualificationPolicy(
            exact_match_threshold=0.99,
            decision_record_id="toy-policy-not-scientific",
        ),
    )
    return runner.evaluate_frozen_candidate(
        snapshot=frozen,
        candidate_id="candidate-1",
        measurements=QualificationMeasurements(
            exact_match_accuracy=1.0,
            metrics=(("synthetic_shift_accuracy", 1.0),),
        ),
        envelope=_envelope("qualification_evaluation", "qualification-1"),
    )


def test_layer_b_refuses_an_incomplete_or_mutated_snapshot():
    snapshot = _snapshot()
    runner = LayerBQualificationRunner(
        evaluation_plan=_synthetic_plan(EvaluationLayer.QUALIFICATION),
        policy=QualificationPolicy(0.99, "toy-policy-not-scientific"),
    )
    incomplete = replace(snapshot, run_complete=False)
    with pytest.raises(ValueError, match="completed, frozen"):
        runner.evaluate_frozen_candidate(
            snapshot=incomplete,
            candidate_id="candidate-1",
            measurements=QualificationMeasurements(1.0),
            envelope=_envelope("qualification_evaluation", "qualification-bad"),
        )
    tampered = replace(snapshot, budget_checkpoint_id="changed-after-freeze")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.evaluate_frozen_candidate(
            snapshot=tampered,
            candidate_id="candidate-1",
            measurements=QualificationMeasurements(1.0),
            envelope=_envelope("qualification_evaluation", "qualification-bad-2"),
        )


def test_layer_b_accepts_only_candidates_in_the_frozen_snapshot():
    runner = LayerBQualificationRunner(
        evaluation_plan=_synthetic_plan(EvaluationLayer.QUALIFICATION),
        policy=QualificationPolicy(0.99, "toy-policy-not-scientific"),
    )
    with pytest.raises(ValueError, match="absent from frozen snapshot"):
        runner.evaluate_frozen_candidate(
            snapshot=_snapshot(),
            candidate_id="candidate-not-frozen",
            measurements=QualificationMeasurements(1.0),
            envelope=_envelope("qualification_evaluation", "qualification-absent"),
        )


def _release_manifest(snapshot, qualification, plan, token):
    candidate = snapshot.candidate("candidate-1")
    return ConfirmationReleaseManifest(
        authorization_id="release-1",
        enabled=True,
        candidate_id="candidate-1",
        frozen_snapshot_id=snapshot.snapshot_id,
        frozen_candidate_sha256=candidate.artifact_sha256,
        qualification_record_id=qualification.envelope.record_id,
        qualification_record_sha256=qualification.record_hash,
        confirmation_plan_sha256=plan.plan_hash,
        confirmation_threshold=0.99,
        token_sha256=sha256_text(token),
        pi_release_record_id="toy-release-not-scientific",
    )


def test_layer_c_is_disabled_by_default(tmp_path):
    roots = EvaluationArtifactRoots.under(tmp_path / "evaluation-artifacts")
    plan = _synthetic_plan(EvaluationLayer.CONFIRMATION)
    snapshot = _snapshot()
    qualification = _qualification(snapshot)
    token = "test-token-never-a-real-secret"
    manifest = _release_manifest(snapshot, qualification, plan, token)
    gate = LayerCReleaseGate(roots=roots, evaluation_plan=plan)
    assert not gate.enabled
    with pytest.raises(PermissionError, match="disabled"):
        gate.authorize_once(
            manifest=manifest,
            token=token,
            snapshot=snapshot,
            qualification=qualification,
        )


def test_layer_c_requires_correct_token_and_consumes_release_once(tmp_path):
    roots = EvaluationArtifactRoots.under(tmp_path / "evaluation-artifacts")
    plan = _synthetic_plan(EvaluationLayer.CONFIRMATION)
    snapshot = _snapshot()
    qualification = _qualification(snapshot)
    token = "test-token-never-a-real-secret"
    manifest = _release_manifest(snapshot, qualification, plan, token)
    gate = LayerCReleaseGate(
        roots=roots,
        evaluation_plan=plan,
        enabled=True,
    )
    with pytest.raises(PermissionError, match="invalid"):
        gate.authorize_once(
            manifest=manifest,
            token="wrong-token",
            snapshot=snapshot,
            qualification=qualification,
        )
    receipt = gate.authorize_once(
        manifest=manifest,
        token=token,
        snapshot=snapshot,
        qualification=qualification,
    )
    with pytest.raises(PermissionError, match="already consumed"):
        gate.authorize_once(
            manifest=manifest,
            token=token,
            snapshot=snapshot,
            qualification=qualification,
        )
    confirmation = gate.build_confirmation_record(
        receipt=receipt,
        manifest=manifest,
        snapshot=snapshot,
        qualification=qualification,
        measurements=ConfirmationMeasurements(
            exact_match_accuracy=1.0,
            metrics=(("synthetic_final_accuracy", 1.0),),
        ),
        envelope=_envelope("confirmation_evaluation", "confirmation-1"),
    )
    assert confirmation.confirmed
    assert confirmation.release_authorization_id == "release-1"
    with pytest.raises(PermissionError, match="already produced"):
        gate.build_confirmation_record(
            receipt=receipt,
            manifest=manifest,
            snapshot=snapshot,
            qualification=qualification,
            measurements=ConfirmationMeasurements(exact_match_accuracy=1.0),
            envelope=_envelope("confirmation_evaluation", "confirmation-2"),
        )


def test_layer_c_rejects_nonqualifying_or_incomplete_layer_b(tmp_path):
    roots = EvaluationArtifactRoots.under(tmp_path / "evaluation-artifacts")
    plan = _synthetic_plan(EvaluationLayer.CONFIRMATION)
    snapshot = _snapshot()
    qualification = replace(_qualification(snapshot), qualifies=False)
    token = "test-token-never-a-real-secret"
    manifest = _release_manifest(snapshot, qualification, plan, token)
    gate = LayerCReleaseGate(
        roots=roots,
        evaluation_plan=plan,
        enabled=True,
    )
    with pytest.raises(PermissionError, match="qualifying Layer B"):
        gate.authorize_once(
            manifest=manifest,
            token=token,
            snapshot=snapshot,
            qualification=qualification,
        )
