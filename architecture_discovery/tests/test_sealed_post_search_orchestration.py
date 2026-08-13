from __future__ import annotations

import hashlib
from dataclasses import asdict, replace

import pytest
from artifacts.records import ArtifactContext, EventKind
from artifacts.store import RunArtifactStore
from common.evaluation_profiles import EvaluationLayer, resolve_evaluation_plan
from evaluation.artifacts import EvaluationArtifactRoots
from evaluation.records import (
    confirmation_evaluation_from_dict,
    content_sha256,
    qualification_evaluation_from_dict,
    sha256_text,
)
from sealed_eval.confirmation import (
    ConfirmationMeasurements,
    ConfirmationReleaseManifest,
    LayerCReleaseGate,
)
from sealed_eval.orchestration import (
    QualificationAuthorizationManifest,
    SealedCandidateArtifactBinding,
    SealedPostSearchOrchestrator,
)
from sealed_eval.qualification import QualificationMeasurements, QualificationPolicy
from sealed_eval.snapshot import FrozenRunSnapshot, freeze_completed_run


def _hash(character: str) -> str:
    return character * 64


def _plan(layer: EvaluationLayer):
    return resolve_evaluation_plan(
        "unit_eval_v1",
        layer=layer,
        case_source_id=f"synthetic-{layer.value}",
        case_source_sha256={
            EvaluationLayer.QUALIFICATION: _hash("b"),
            EvaluationLayer.CONFIRMATION: _hash("c"),
        }[layer],
    )


def _fixture(tmp_path):
    context = ArtifactContext(
        study_id="study-1",
        block_id="block-1",
        run_id="run-1",
        condition_id="C0",
        writer_component="test.sealed",
        code_sha256=_hash("1"),
        config_sha256=_hash("2"),
        environment_sha256=_hash("3"),
    )
    store = RunArtifactStore(tmp_path / "ledger", context)
    source = store.objects.put_bytes(
        b'{"schema_name":"architecture_ir"}\n',
        media_type="application/vnd.rl4rl.architecture-ir+json",
    )
    store.append(
        EventKind.CANDIDATE,
        {
            "candidate_id": "candidate-1",
            "source_object_sha256": source.sha256,
            "object_sha256s": [source.sha256],
        },
    )
    store.append(
        EventKind.RUN_STATUS,
        {"transition_key": "run:completed", "status": "completed"},
    )
    search_index = store.freeze_index("search_completion")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "candidate-1.pt"
    checkpoint.write_bytes(b"trusted-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    binding = SealedCandidateArtifactBinding(
        candidate_id="candidate-1",
        source_object_sha256=source.sha256,
        checkpoint_relative_path=checkpoint.name,
        checkpoint_sha256=checkpoint_sha256,
    )
    plan = _plan(EvaluationLayer.QUALIFICATION)
    policy = QualificationPolicy(0.9, "synthetic-decision")
    authorization = QualificationAuthorizationManifest(
        authorization_id="layer-b-authorization-1",
        enabled=True,
        run_id=context.run_id,
        search_index_sha256=search_index.index_sha256,
        snapshot_id="snapshot-1",
        budget_checkpoint_id="budget-final",
        qualification_plan_sha256=plan.plan_hash,
        qualification_policy_sha256=content_sha256(asdict(policy)),
        decision_record_id=policy.decision_record_id,
        candidate_ids=(binding.candidate_id,),
    )
    roots = EvaluationArtifactRoots.under(tmp_path / "evaluations")
    orchestrator = SealedPostSearchOrchestrator(
        artifact_store=store,
        evaluation_roots=roots,
        checkpoint_root=checkpoints,
    )
    return orchestrator, store, roots, binding, plan, policy, authorization


def test_sealed_orchestrator_freezes_qualifies_confirms_and_indexes(tmp_path):
    (
        orchestrator,
        store,
        _roots,
        binding,
        layer_b_plan,
        policy,
        authorization,
    ) = _fixture(tmp_path)
    qualified = orchestrator.qualify(
        authorization=authorization,
        evaluation_plan=layer_b_plan,
        policy=policy,
        candidate_bindings=(binding,),
        measurements={
            binding.candidate_id: QualificationMeasurements(
                exact_match_accuracy=1.0,
                metrics=(("sealed_fixture", 1.0),),
            )
        },
    )
    assert qualified.records[0].qualifies is True
    assert (
        qualification_evaluation_from_dict(qualified.records[0].to_dict())
        == qualified.records[0]
    )
    assert (
        FrozenRunSnapshot.from_dict(qualified.snapshot.to_dict())
        == qualified.snapshot
    )

    token = "synthetic-one-shot-token"
    layer_c_plan = _plan(EvaluationLayer.CONFIRMATION)
    candidate = qualified.snapshot.candidate(binding.candidate_id)
    release = ConfirmationReleaseManifest(
        authorization_id="layer-c-release-1",
        enabled=True,
        candidate_id=binding.candidate_id,
        frozen_snapshot_id=qualified.snapshot.snapshot_id,
        frozen_candidate_sha256=candidate.artifact_sha256,
        qualification_record_id=qualified.records[0].envelope.record_id,
        qualification_record_sha256=qualified.records[0].record_hash,
        confirmation_plan_sha256=layer_c_plan.plan_hash,
        confirmation_threshold=0.9,
        token_sha256=sha256_text(token),
        pi_release_record_id="synthetic-release",
    )
    confirmation = orchestrator.confirm_once(
        snapshot=qualified.snapshot,
        qualification=qualified.records[0],
        evaluation_plan=layer_c_plan,
        release_manifest=release,
        release_token=token,
        measurements=ConfirmationMeasurements(1.0),
    )
    assert confirmation.record.confirmed is True
    assert (
        confirmation_evaluation_from_dict(confirmation.record.to_dict())
        == confirmation.record
    )
    assert confirmation.final_index.event_count == len(store.scan().events)
    with pytest.raises(PermissionError, match="already consumed"):
        orchestrator.confirm_once(
            snapshot=qualified.snapshot,
            qualification=qualified.records[0],
            evaluation_plan=layer_c_plan,
            release_manifest=release,
            release_token=token,
            measurements=ConfirmationMeasurements(1.0),
        )


@pytest.mark.parametrize("invalid", ["false", 0, 1, None])
def test_sealed_boundaries_reject_non_boolean_values(tmp_path, invalid):
    snapshot = freeze_completed_run(
        snapshot_id="snapshot",
        run_id="run",
        budget_checkpoint_id="budget",
        terminal_event_sha256=_hash("a"),
        candidate_artifacts={"candidate": (_hash("b"), _hash("c"))},
        run_complete=True,
    )
    with pytest.raises(ValueError, match="boolean"):
        replace(snapshot, frozen=invalid).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        QualificationMeasurements(1.0, complete=invalid).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        ConfirmationMeasurements(1.0, complete=invalid).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        LayerCReleaseGate(
            roots=EvaluationArtifactRoots.under(tmp_path / f"roots-{invalid!s}"),
            evaluation_plan=_plan(EvaluationLayer.CONFIRMATION),
            enabled=invalid,  # type: ignore[arg-type]
        )


def test_layer_b_rejects_disabled_authorization_and_symlink_checkpoint(tmp_path):
    (
        orchestrator,
        _store,
        _roots,
        binding,
        plan,
        policy,
        authorization,
    ) = _fixture(tmp_path)
    with pytest.raises(PermissionError, match="disabled"):
        orchestrator.freeze_authorized_snapshot(
            authorization=replace(authorization, enabled=False),
            evaluation_plan=plan,
            policy=policy,
            candidate_bindings=(binding,),
        )

    checkpoint = tmp_path / "checkpoints" / binding.checkpoint_relative_path
    target = tmp_path / "outside.pt"
    target.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    checkpoint.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        orchestrator.freeze_authorized_snapshot(
            authorization=authorization,
            evaluation_plan=plan,
            policy=policy,
            candidate_bindings=(binding,),
        )


def test_sealed_record_loaders_reject_extensions_and_boolean_type_confusion(
    tmp_path,
):
    orchestrator, _store, _roots, binding, plan, policy, authorization = _fixture(
        tmp_path
    )
    qualified = orchestrator.qualify(
        authorization=authorization,
        evaluation_plan=plan,
        policy=policy,
        candidate_bindings=(binding,),
        measurements={binding.candidate_id: QualificationMeasurements(1.0)},
    )
    payload = qualified.records[0].to_dict()
    with pytest.raises(ValueError, match="invalid fields"):
        qualification_evaluation_from_dict({**payload, "sealed_hint": "forbidden"})
    with pytest.raises(ValueError, match="boolean"):
        qualification_evaluation_from_dict({**payload, "qualifies": 1})
