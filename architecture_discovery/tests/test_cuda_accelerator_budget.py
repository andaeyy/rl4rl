from __future__ import annotations

import pytest

from study.budget import (
    AcceleratorBackend,
    AcceleratorResourceLedger,
    AcceleratorResourceSpec,
    BudgetLedger,
    BudgetSpec,
)
from study.runtime_adapters import (
    CandidateSourceStore,
    LayerACandidateEvaluator,
    ScientificReadinessBlocked,
)


def test_legacy_mps_budget_records_round_trip_byte_for_field() -> None:
    spec = BudgetSpec.toy(0)
    ledger = BudgetLedger(spec)
    ledger.record_seed_evaluation(
        training_attempts=1,
        training_steps=4,
        training_examples=16,
        mps_seconds=0.25,
        evaluation_cases=8,
    )
    spec_payload = spec.to_dict()
    ledger_payload = ledger.to_dict()

    assert BudgetSpec.from_dict(spec_payload).to_dict() == spec_payload
    assert BudgetLedger.from_dict(ledger_payload).to_dict() == ledger_payload
    assert spec_payload["schema_version"] == "1.0"
    assert ledger_payload["schema_version"] == "1.0"
    assert "accelerator_seconds" not in spec_payload
    assert "accelerator_seconds" not in ledger_payload


def test_v1_mps_records_adapt_to_backend_neutral_v2_schema() -> None:
    spec = BudgetSpec.toy(0)
    ledger = BudgetLedger(spec)
    ledger.record_seed_evaluation(
        training_attempts=1,
        training_steps=4,
        training_examples=16,
        mps_seconds=0.25,
        evaluation_cases=8,
    )

    resource_spec = AcceleratorResourceSpec.from_dict(spec.to_dict())
    resource_ledger = AcceleratorResourceLedger.from_dict(ledger.to_dict())

    assert resource_spec.backend is AcceleratorBackend.MPS
    assert resource_spec.accelerator_seconds == spec.mps_seconds
    assert resource_ledger.accelerator_seconds == ledger.mps_seconds
    assert resource_ledger.to_dict()["schema_version"] == "2.0"


def test_cuda_and_mps_seconds_cannot_be_pooled() -> None:
    spec = AcceleratorResourceSpec(
        backend=AcceleratorBackend.CUDA,
        hardware_condition="nvidia_a40_cuda_smoke_v1",
        accelerator_seconds=10.0,
    )
    ledger = AcceleratorResourceLedger(spec)
    ledger.record(
        1.5,
        backend="cuda",
        hardware_condition="nvidia_a40_cuda_smoke_v1",
    )
    assert ledger.accelerator_seconds == 1.5

    with pytest.raises(ValueError, match="cannot be pooled"):
        ledger.record(
            1.0,
            backend="mps",
            hardware_condition="mps_full_train_v1",
        )

    restored = AcceleratorResourceLedger.from_dict(ledger.to_dict())
    assert restored.to_dict() == ledger.to_dict()


def test_legacy_primary_engine_refuses_to_label_cuda_time_as_mps(tmp_path) -> None:
    with pytest.raises(ScientificReadinessBlocked, match="mps_seconds"):
        LayerACandidateEvaluator(
            study_id="study",
            block_id="block",
            run_id="run",
            condition_id="C0",
            initial_candidate_id="candidate",
            source_store=CandidateSourceStore(tmp_path / "sources"),
            output_root=tmp_path / "outputs",
            training_profile="full_train_cuda_a40_v1",
            device="cuda",
            allow_cpu_for_tests=False,
            evaluation_profile="scientific_layer_a_v1",
            evaluation_case_count=1,
            pi_decision_record_id=None,
            eligibility_threshold=0.99,
        )
