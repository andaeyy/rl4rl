from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.outcomes import RunOutcome, RunTerminalStatus
from analysis.plan import AnalysisPlan
from analysis.time_to_first import DiscoveryRecord, TimeToFirstRecord
from mechanism.claims import ClaimEvidence
from mechanism.fakes import toy_mechanism_plan
from mechanism.plans import MechanismExperimentPlan
from replication.clean_room import CleanRoomReimplementationRecord
from replication.fakes import toy_replication_policy
from replication.ledger import ReplicationAttemptRecord, ReplicationStatus
from replication.policy import ReplicationPolicy
from study.budget import OpportunityOutcome
from study.contracts import StudySpec
from study.interfaces import EvaluationResult
from study.runtime_adapters import LayerACandidateEvaluator


@pytest.mark.parametrize(
    ("payload", "loader"),
    (
        (
            {**StudySpec.toy().to_dict(), "scientific": "false"},
            StudySpec.from_dict,
        ),
        (
            {**AnalysisPlan.toy().to_dict(), "scientific": "false"},
            AnalysisPlan.from_dict,
        ),
        (
            {**toy_mechanism_plan().to_dict(), "scientific": "false"},
            MechanismExperimentPlan.from_dict,
        ),
        (
            {**toy_replication_policy().to_dict(), "scientific": "false"},
            ReplicationPolicy.from_dict,
        ),
    ),
)
def test_frozen_scientific_flags_reject_string_truthiness(payload, loader) -> None:
    with pytest.raises(ValueError, match="scientific must be boolean"):
        loader(payload)


def test_reference_corpus_flags_reject_string_truthiness() -> None:
    from novelty.corpus import ReferenceCorpusManifest

    template = Path(__file__).parents[1] / "novelty" / "reference_corpus.template.json"
    payload = json.loads(template.read_text(encoding="utf-8"))
    payload["population_complete"] = "false"

    with pytest.raises(ValueError, match="population_complete must be boolean"):
        ReferenceCorpusManifest.from_dict(payload)


def test_replication_mandatory_guards_reject_string_truthiness() -> None:
    payload = toy_replication_policy().to_dict()
    payload["clean_room_reimplementation_required"] = "false"

    with pytest.raises(
        ValueError,
        match="clean_room_reimplementation_required must be boolean",
    ):
        ReplicationPolicy.from_dict(payload)


def test_scientific_integer_fields_reject_booleans_and_fractions() -> None:
    study_payload = StudySpec.toy().to_dict()
    study_payload["block_count"] = True
    with pytest.raises(ValueError, match="block_count must be an integer"):
        StudySpec.from_dict(study_payload)

    mechanism_payload = toy_mechanism_plan().to_dict()
    mechanism_payload["training_budget"]["max_steps"] = True
    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        MechanismExperimentPlan.from_dict(mechanism_payload)

    replication_payload = toy_replication_policy().to_dict()
    replication_payload["success_rule"]["minimum_successful_seeds"] = True
    with pytest.raises(
        ValueError, match="minimum_successful_seeds must be a positive integer"
    ):
        ReplicationPolicy.from_dict(replication_payload)

    evaluation_payload = EvaluationResult(
        outcome=OpportunityOutcome.ACCEPTED,
        score=1.0,
        training_attempts=1,
        training_steps=1,
        training_examples=1,
        mps_seconds=1.0,
        evaluation_cases=1,
    ).to_dict()
    evaluation_payload["training_steps"] = -0.5
    with pytest.raises(ValueError, match="training_steps must be an integer"):
        EvaluationResult.from_dict(evaluation_payload)


def test_primary_outcome_loader_does_not_truncate_fractional_exposure() -> None:
    payload = RunOutcome(
        study_id="study",
        block_id="block",
        run_id="run",
        condition_id="C0",
        run_seed=1,
        terminal_status=RunTerminalStatus.COMPLETED,
        qualifying_cluster_count=0,
        proposal_exposure=1,
        token_exposure=1,
    ).to_dict()
    payload["proposal_exposure"] = -0.5

    with pytest.raises(ValueError, match="proposal_exposure must be an integer"):
        RunOutcome.from_dict(payload)


def test_mechanism_and_survival_booleans_reject_string_truthiness() -> None:
    with pytest.raises(ValueError, match="supports_prediction must be boolean"):
        ClaimEvidence(
            evidence_id="evidence:test",
            requirement_id="requirement:test",
            discriminating_test_ids=("test:one",),
            artifact_sha256="a" * 64,
            supports_prediction="false",
            summary="A deliberately malformed evidence record.",
        )
    with pytest.raises(ValueError, match="qualifies must be boolean"):
        DiscoveryRecord("run", "cluster", 1, 1, "false")
    with pytest.raises(ValueError, match="event must be boolean"):
        TimeToFirstRecord("run", "C0", "false", 1, 1)


def test_clean_room_and_itt_attestations_require_exact_booleans() -> None:
    with pytest.raises(
        ValueError, match="original_candidate_source_accessed must be boolean"
    ):
        CleanRoomReimplementationRecord(
            record_id="record:test",
            candidate_snapshot_id="snapshot:test",
            candidate_snapshot_sha256="a" * 64,
            architecture_spec_sha256="b" * 64,
            implementation_sha256="c" * 64,
            original_checkpoint_sha256="d" * 64,
            builder_id="builder:test",
            implementer_id="implementer:test",
            protocol_id="protocol:test",
            original_candidate_source_accessed="false",
            original_checkpoint_accessed=False,
        )
    with pytest.raises(ValueError, match="counts_in_intent_to_treat must be boolean"):
        ReplicationAttemptRecord(
            policy_id="policy:test",
            seed_id="seed:test",
            attempt_index=0,
            status=ReplicationStatus.SCIENTIFIC_FAILURE,
            build_id=None,
            metric_name=None,
            metric_value=None,
            final_state_sha256=None,
            error_class="ScientificFailure",
            error_message="synthetic failure",
            counts_in_intent_to_treat="true",
        )


def test_training_resource_reader_rejects_fractional_step_counts(tmp_path) -> None:
    output = tmp_path / "training"
    output.mkdir()
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "candidate_source_hash": "candidate",
                "profile_name": "profile",
                "steps_completed": -0.5,
                "examples_processed": 0,
                "train_seconds": 0.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="steps_completed must be an integer"):
        LayerACandidateEvaluator._training_resources(
            output,
            expected_candidate_hash="candidate",
            expected_profile_name="profile",
        )
