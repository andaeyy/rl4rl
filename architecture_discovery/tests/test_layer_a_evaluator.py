from dataclasses import replace

import common.evaluator as evaluator_module
from common.evaluation_profiles import EvaluationLayer, resolve_evaluation_plan
from common.evaluator import (
    SearchEvaluationContext,
    evaluate_candidate,
    evaluate_trained_candidate_in_process,
)
from common.public_evaluation import (
    PUBLIC_LAYER_A_SOURCE_ID,
    PUBLIC_LAYER_A_SOURCE_SHA256,
)
from common.training_config import TrainingSeedBundle
from evaluation.records import CONTROLLER_SEARCH_FIELDS, SearchEvaluationRecord


def test_trained_candidate_returns_typed_public_layer_a_only(cpu_smoke_training):
    training, _output = cpu_smoke_training
    plan = resolve_evaluation_plan(
        "smoke_eval_v1",
        layer=EvaluationLayer.SEARCH,
        case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
        case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
    )
    record = evaluate_trained_candidate_in_process(
        candidate_path="common/initial_candidate.py",
        training=training,
        seeds=TrainingSeedBundle.from_run_seed(17),
        requested_device="cpu",
        allow_cpu_for_tests=True,
        evaluation_plan=plan,
        context=SearchEvaluationContext(
            study_id="test-study",
            block_id="test-block",
            run_id="test-run",
            condition_id="C0",
        ),
        eligibility_threshold=0.0,
    )
    assert isinstance(record, SearchEvaluationRecord)
    assert set(record.controller_view().as_dict()) == CONTROLLER_SEARCH_FIELDS
    assert record.execution_ok
    assert record.transformer_valid
    assert record.eligible_for_parent
    assert "shadow_accuracy" not in record.to_dict()
    assert "sealed_metrics" not in record.to_dict()


def test_training_failure_never_claims_runtime_transformer_validity(
    cpu_smoke_training,
):
    training, _output = cpu_smoke_training
    failed_training = replace(
        training,
        success=False,
        failure_stage="training_timeout",
        error="wall-time cap exceeded",
    )
    plan = resolve_evaluation_plan(
        "smoke_eval_v1",
        layer=EvaluationLayer.SEARCH,
        case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
        case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
    )
    record = evaluate_trained_candidate_in_process(
        candidate_path="common/initial_candidate.py",
        training=failed_training,
        seeds=TrainingSeedBundle.from_run_seed(17),
        requested_device="cpu",
        allow_cpu_for_tests=True,
        evaluation_plan=plan,
        context=SearchEvaluationContext(
            study_id="test-study",
            block_id="test-block",
            run_id="failed-run",
            condition_id="C0",
        ),
        eligibility_threshold=0.0,
    )
    assert not record.execution_ok
    assert not record.transformer_valid
    assert not record.eligible_for_parent


def test_evaluator_cleanup_failure_is_an_infrastructure_failure(
    cpu_smoke_training,
    monkeypatch,
):
    training, _output = cpu_smoke_training
    plan = resolve_evaluation_plan(
        "smoke_eval_v1",
        layer=EvaluationLayer.SEARCH,
        case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
        case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
    )

    def fail_cleanup(_device):
        raise RuntimeError("sensitive-driver-detail")

    monkeypatch.setattr(evaluator_module, "cleanup_accelerator", fail_cleanup)
    record = evaluate_trained_candidate_in_process(
        candidate_path="common/initial_candidate.py",
        training=training,
        seeds=TrainingSeedBundle.from_run_seed(17),
        requested_device="cpu",
        allow_cpu_for_tests=True,
        evaluation_plan=plan,
        context=SearchEvaluationContext(
            study_id="test-study",
            block_id="test-block",
            run_id="cleanup-failure-run",
            condition_id="C0",
        ),
        eligibility_threshold=0.0,
    )

    assert not record.execution_ok
    assert not record.transformer_valid
    assert not record.eligible_for_parent
    assert record.failure_stage == "accelerator_cleanup_failure"
    assert record.infrastructure_failure
    assert "sensitive-driver-detail" not in str(record.to_dict())


def test_scientific_candidate_evaluation_has_no_implicit_smoke_count(monkeypatch):
    monkeypatch.delenv("DISCOVERY_LAYER_A_CASES", raising=False)
    monkeypatch.delenv("DISCOVERY_SCIENTIFIC_DECISION_RECORD", raising=False)
    try:
        evaluate_candidate(
            "common/initial_candidate.py",
            training_profile="full_train_v1",
            device="mps",
        )
    except ValueError as error:
        assert "case_count must be supplied" in str(error)
    else:  # pragma: no cover - fail closed if a future default is introduced
        raise AssertionError("scientific evaluation accepted an implicit case count")


def test_worker_round_trip_returns_exact_layer_a_record(tmp_path):
    record = evaluate_candidate(
        "common/initial_candidate.py",
        training_profile="smoke_train_v1",
        training_seed=23,
        training_output_dir=tmp_path / "training",
        device="cpu",
        allow_cpu_for_tests=True,
        evaluation_profile="smoke_eval_v1",
        evaluation_case_count=8,
        eligibility_threshold=0.0,
        context=SearchEvaluationContext(
            study_id="worker-test",
            block_id="worker-block",
            run_id="worker-run",
            condition_id="C0",
        ),
    )
    assert isinstance(record, SearchEvaluationRecord)
    assert record.envelope.study_id == "worker-test"
    assert record.execution_ok
    assert record.eligible_for_parent
