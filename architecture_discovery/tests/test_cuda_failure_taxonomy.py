from __future__ import annotations

import pytest

from artifacts.study_sink import ImmutableStudyEventSink
from artifacts.failures import (
    FailureClass,
    FailureDomain,
    FailureRecord,
    RerunNotAuthorized,
    RerunPolicy,
    authorize_rerun,
)
from common.device import DeviceUnavailableError
from common.evaluator import INFRASTRUCTURE_FAILURE_STAGES
from common.trainer import _failure_stage


@pytest.mark.parametrize(
    ("failure_class", "domain"),
    [
        (FailureClass.CUDA_UNAVAILABLE, FailureDomain.INFRASTRUCTURE),
        (FailureClass.CUDA_DRIVER_RUNTIME_FAILURE, FailureDomain.INFRASTRUCTURE),
        (FailureClass.CUDA_OOM, FailureDomain.CANDIDATE),
        (
            FailureClass.UNSUPPORTED_DETERMINISTIC_CUDA_OPERATION,
            FailureDomain.CANDIDATE,
        ),
    ],
)
def test_cuda_failure_classes_have_fail_closed_domains(
    failure_class: FailureClass,
    domain: FailureDomain,
) -> None:
    record = FailureRecord.create(
        attempt_id="cuda-attempt",
        failure_class=failure_class,
        stage="cuda-stage",
    )
    assert record.failure_domain is domain


def test_cuda_driver_failure_can_rerun_but_oom_and_determinism_cannot() -> None:
    driver = FailureRecord.create(
        attempt_id="attempt-driver",
        failure_class=FailureClass.CUDA_DRIVER_RUNTIME_FAILURE,
        stage="cuda_driver_runtime_failure",
    )
    authorization = authorize_rerun(
        assigned_run_id="assigned-cuda",
        previous_attempt_id="attempt-driver",
        attempt_number=1,
        failure=driver,
        policy=RerunPolicy(),
    )
    assert authorization.triggering_failure_class is driver.failure_class

    for failure_class in (
        FailureClass.CUDA_OOM,
        FailureClass.UNSUPPORTED_DETERMINISTIC_CUDA_OPERATION,
        FailureClass.CUDA_UNAVAILABLE,
    ):
        failure = FailureRecord.create(
            attempt_id="attempt-no-rerun",
            failure_class=failure_class,
            stage="cuda-stage",
        )
        with pytest.raises(RerunNotAuthorized):
            authorize_rerun(
                assigned_run_id="assigned-cuda",
                previous_attempt_id="attempt-no-rerun",
                attempt_number=1,
                failure=failure,
                policy=RerunPolicy(),
            )


@pytest.mark.parametrize(
    ("error", "stage", "failure_class"),
    [
        (
            DeviceUnavailableError("CUDA unavailable"),
            "cuda_unavailable",
            FailureClass.CUDA_UNAVAILABLE,
        ),
        (
            RuntimeError("CUDA out of memory"),
            "cuda_oom",
            FailureClass.CUDA_OOM,
        ),
        (
            RuntimeError("deterministic CUDA operation is unsupported"),
            "unsupported_deterministic_cuda_operation",
            FailureClass.UNSUPPORTED_DETERMINISTIC_CUDA_OPERATION,
        ),
        (
            RuntimeError("CUDA driver initialization failed"),
            "cuda_driver_runtime_failure",
            FailureClass.CUDA_DRIVER_RUNTIME_FAILURE,
        ),
    ],
)
def test_cuda_trainer_stages_reach_artifact_taxonomy(
    error, stage, failure_class
) -> None:
    assert _failure_stage(error, requested_device="cuda") == stage
    outcome = (
        "infrastructure_failure"
        if failure_class
        in {FailureClass.CUDA_UNAVAILABLE, FailureClass.CUDA_DRIVER_RUNTIME_FAILURE}
        else "scientific_failure"
    )
    assert ImmutableStudyEventSink._failure_class(outcome, stage) is failure_class


def test_cuda_candidate_failures_are_not_mislabeled_as_infrastructure() -> None:
    assert "cuda_unavailable" in INFRASTRUCTURE_FAILURE_STAGES
    assert "cuda_driver_runtime_failure" in INFRASTRUCTURE_FAILURE_STAGES
    assert "cuda_oom" not in INFRASTRUCTURE_FAILURE_STAGES
    assert (
        "unsupported_deterministic_cuda_operation"
        not in INFRASTRUCTURE_FAILURE_STAGES
    )


def test_mps_failure_stages_remain_legacy_compatible() -> None:
    assert (
        _failure_stage(
            DeviceUnavailableError("MPS unavailable"),
            requested_device="mps",
        )
        == "device_unavailable"
    )
    assert (
        _failure_stage(RuntimeError("out of memory"), requested_device="mps")
        == "training_oom"
    )
