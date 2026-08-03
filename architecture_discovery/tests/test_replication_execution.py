from dataclasses import replace

import pytest

from replication.execution import (
    ReplicationBuild,
    ReplicationIntegrityError,
    ReplicationRunner,
)
from replication.fakes import (
    DeterministicCleanRoomBuilder,
    DeterministicReplicationTrainer,
    TinyCleanRoomModel,
    toy_clean_room_record,
    toy_replication_policy,
)
from replication.ledger import (
    IntentToTreatLedger,
    ReplicationAttemptRecord,
    ReplicationStatus,
)
from replication.policy import freeze_replication_policy


def test_replication_builds_each_preselected_seed_from_scratch(tmp_path):
    policy = toy_replication_policy(seed_count=3)
    receipt = freeze_replication_policy(policy, tmp_path / "policy.json")
    builder = DeterministicCleanRoomBuilder()
    trainer = DeterministicReplicationTrainer()
    result = ReplicationRunner().run(
        receipt,
        clean_room=toy_clean_room_record(),
        builder=builder,
        trainer=trainer,
    )
    assert result.success
    assert result.ledger.complete
    assert builder.calls == [seed.initialization_seed for seed in policy.seeds]
    assert trainer.calls == [
        (
            seed.seed_id,
            seed.training_seed,
            seed.data_order_seed,
            seed.evaluation_seed,
        )
        for seed in policy.seeds
    ]
    builds = [
        attempt.build_id
        for seed_id in result.ledger.seed_ids
        for attempt in result.ledger.attempts_for(seed_id)
    ]
    assert len(builds) == len(set(builds)) == len(policy.seeds)


def test_failed_seed_remains_in_intent_to_treat_record(tmp_path):
    policy = toy_replication_policy(seed_count=3)
    receipt = freeze_replication_policy(policy, tmp_path / "policy.json")
    trainer = DeterministicReplicationTrainer(
        failures={"replication-seed:1": ["candidate"]}
    )
    result = ReplicationRunner().run(
        receipt,
        clean_room=toy_clean_room_record(),
        builder=DeterministicCleanRoomBuilder(),
        trainer=trainer,
    )
    assert result.success
    assert result.ledger.terminal_status("replication-seed:1") is ReplicationStatus.CANDIDATE_FAILURE
    serialized = result.ledger.to_dict()
    failed = serialized["assignments"][1]
    assert failed["seed"]["seed_id"] == "replication-seed:1"
    assert failed["assignment_status"] == "candidate_failure"
    assert failed["attempts"][0]["counts_in_intent_to_treat"] is True
    assert len(serialized["assignments"]) == 3


def test_preregistered_infrastructure_retry_is_retained_and_rebuilds(tmp_path):
    policy = toy_replication_policy(seed_count=2)
    receipt = freeze_replication_policy(policy, tmp_path / "policy.json")
    builder = DeterministicCleanRoomBuilder()
    trainer = DeterministicReplicationTrainer(
        failures={"replication-seed:0": ["infrastructure"]}
    )
    result = ReplicationRunner().run(
        receipt,
        clean_room=toy_clean_room_record(),
        builder=builder,
        trainer=trainer,
    )
    attempts = result.ledger.attempts_for("replication-seed:0")
    assert [item.status for item in attempts] == [
        ReplicationStatus.INFRASTRUCTURE_FAILURE,
        ReplicationStatus.SUCCEEDED,
    ]
    assert len(builder.calls) == 3
    assert attempts[0].build_id != attempts[1].build_id


def test_exhausted_infrastructure_failure_is_terminal_and_not_replaced(tmp_path):
    policy = toy_replication_policy(seed_count=2)
    policy = replace(
        policy,
        success_rule=replace(
            policy.success_rule,
            minimum_successful_seeds=2,
        ),
    )
    receipt = freeze_replication_policy(policy, tmp_path / "policy.json")
    result = ReplicationRunner().run(
        receipt,
        clean_room=toy_clean_room_record(),
        builder=DeterministicCleanRoomBuilder(),
        trainer=DeterministicReplicationTrainer(
            failures={
                "replication-seed:0": ["infrastructure", "infrastructure"]
            }
        ),
    )
    assert result.ledger.complete
    assert result.ledger.terminal_status("replication-seed:0") is ReplicationStatus.INFRASTRUCTURE_FAILURE
    assert len(result.ledger.attempts_for("replication-seed:0")) == 2
    assert not result.success


def test_ledger_has_no_seed_replacement_path():
    policy = toy_replication_policy(seed_count=1)
    ledger = IntentToTreatLedger(policy)
    alien = ReplicationAttemptRecord(
        policy_id=policy.policy_id,
        seed_id="replacement-seed",
        attempt_index=0,
        status=ReplicationStatus.CANDIDATE_FAILURE,
        build_id=None,
        metric_name=None,
        metric_value=None,
        final_state_sha256=None,
        error_class="CandidateReplicationError",
        error_message="failed",
    )
    with pytest.raises(ValueError, match="add or replace"):
        ledger.record(alien)


def test_runner_requires_frozen_policy_before_any_outcome(tmp_path):
    policy = toy_replication_policy(seed_count=1)
    with pytest.raises(TypeError, match="frozen policy"):
        ReplicationRunner().run(
            policy,  # type: ignore[arg-type]
            clean_room=toy_clean_room_record(),
            builder=DeterministicCleanRoomBuilder(),
            trainer=DeterministicReplicationTrainer(),
        )


def test_checkpoint_reuse_and_original_state_are_rejected(tmp_path):
    policy = toy_replication_policy(seed_count=1)
    receipt = freeze_replication_policy(policy, tmp_path / "policy.json")
    clean_room = toy_clean_room_record()

    class CheckpointBuilder:
        def build_untrained(self, record, *, initialization_seed):
            return ReplicationBuild(
                model=TinyCleanRoomModel("bad"),
                build_id="repbuild:bad",
                clean_room_record_id=record.record_id,
                implementation_sha256=record.implementation_sha256,
                initialization_seed=initialization_seed,
                initial_state_sha256="e" * 64,
                parameter_count_metadata=10,
                loaded_checkpoint_sha256=record.original_checkpoint_sha256,
            )

    with pytest.raises(ReplicationIntegrityError, match="checkpoint"):
        ReplicationRunner().run(
            receipt,
            clean_room=clean_room,
            builder=CheckpointBuilder(),
            trainer=DeterministicReplicationTrainer(),
        )

    class OriginalStateBuilder:
        def build_untrained(self, record, *, initialization_seed):
            return ReplicationBuild(
                model=TinyCleanRoomModel("copied"),
                build_id="repbuild:copied",
                clean_room_record_id=record.record_id,
                implementation_sha256=record.implementation_sha256,
                initialization_seed=initialization_seed,
                initial_state_sha256=record.original_checkpoint_sha256,
                parameter_count_metadata=10,
            )

    with pytest.raises(ReplicationIntegrityError, match="prohibited"):
        ReplicationRunner().run(
            receipt,
            clean_room=clean_room,
            builder=OriginalStateBuilder(),
            trainer=DeterministicReplicationTrainer(),
        )
