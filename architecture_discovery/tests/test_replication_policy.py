import json
from dataclasses import replace

import pytest

from replication.fakes import toy_clean_room_record, toy_replication_policy
from replication.policy import (
    FailurePolicy,
    ReplicationPolicy,
    freeze_replication_policy,
    load_frozen_replication_policy,
)


def test_policy_freezes_preselected_seed_bundles_and_success_rule(tmp_path):
    policy = toy_replication_policy(seed_count=3)
    path = tmp_path / "replication-policy.json"
    receipt = freeze_replication_policy(policy, path)
    restored = load_frozen_replication_policy(path)
    assert restored.policy_hash == policy.policy_hash
    assert restored.policy == policy
    assert tuple(seed.seed_id for seed in policy.seeds) == (
        "replication-seed:0",
        "replication-seed:1",
        "replication-seed:2",
    )
    assert receipt.policy.failure_policy.seed_replacement_policy == "forbidden"
    with pytest.raises(FileExistsError):
        freeze_replication_policy(policy, path)


def test_frozen_policy_detects_outcome_driven_tampering(tmp_path):
    policy = toy_replication_policy()
    path = tmp_path / "replication-policy.json"
    receipt = freeze_replication_policy(policy, path)
    payload = json.loads(path.read_text())
    payload["policy"]["success_rule"]["minimum_successful_seeds"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch|modified"):
        receipt.verify()


def test_seed_replacement_and_missing_preselected_seeds_are_rejected():
    policy = toy_replication_policy(seed_count=2)
    duplicate = replace(policy.seeds[1], seed_id="replication-seed:0")
    with pytest.raises(ValueError, match="seed IDs"):
        replace(policy, seeds=(policy.seeds[0], duplicate))
    with pytest.raises(ValueError, match="replacement"):
        FailurePolicy(
            infrastructure_retry_classes=(),
            max_infrastructure_retries=0,
            seed_replacement_policy="replace_failed_seed",
        )


def test_promotion_rule_requires_layer_b_claim_and_frozen_metric():
    rule = toy_replication_policy().promotion_rule
    assert not rule.evaluate(
        layer_b_qualified=False,
        mechanism_claim_present=True,
        qualification_metrics={"layer_b_accuracy": 1.0},
    )
    assert not rule.evaluate(
        layer_b_qualified=True,
        mechanism_claim_present=False,
        qualification_metrics={"layer_b_accuracy": 1.0},
    )
    assert rule.evaluate(
        layer_b_qualified=True,
        mechanism_claim_present=True,
        qualification_metrics={"layer_b_accuracy": 0.95},
    )


def test_clean_room_record_rejects_source_or_checkpoint_access():
    record = toy_clean_room_record()
    record.assert_compatible(toy_replication_policy())
    with pytest.raises(ValueError, match="source"):
        replace(record, original_candidate_source_accessed=True)
    with pytest.raises(ValueError, match="checkpoint"):
        replace(record, original_checkpoint_accessed=True)
    with pytest.raises(ValueError, match="checkpoint"):
        replace(record, initialization_checkpoint_id="best-pt")


def test_policy_round_trip_preserves_hash():
    policy = toy_replication_policy()
    restored = ReplicationPolicy.from_dict(policy.to_dict())
    assert restored == policy
    assert restored.policy_hash == policy.policy_hash
