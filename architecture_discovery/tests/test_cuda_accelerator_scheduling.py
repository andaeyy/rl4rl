from __future__ import annotations

import pytest

from study.scheduling import (
    AcceleratorLease,
    AcceleratorLeaseBusy,
    MPSLease,
    ScheduleStateError,
    cuda_accelerator_identity,
)
from study.serialization import read_json


def test_cuda_lease_is_keyed_by_gpu_uuid_and_exclusive(tmp_path) -> None:
    first = AcceleratorLease.for_cuda_allocation(
        tmp_path,
        run_id="run-one",
        gpu_uuid="GPU-a40-one",
    )
    competitor = AcceleratorLease.for_cuda_allocation(
        tmp_path,
        run_id="run-two",
        gpu_uuid="GPU-a40-one",
    )
    another_gpu = AcceleratorLease.for_cuda_allocation(
        tmp_path,
        run_id="run-three",
        gpu_uuid="GPU-a40-two",
    )
    assert first.path == competitor.path
    assert first.path != another_gpu.path

    with first:
        payload = read_json(first.path)
        assert payload["schema_name"] == "AcceleratorLease"
        assert payload["schema_version"] == "2.0"
        assert payload["accelerator_key"] == "cuda:uuid:GPU-a40-one"
        with pytest.raises(AcceleratorLeaseBusy):
            competitor.acquire()
        with another_gpu:
            assert another_gpu.path.is_file()


def test_scheduler_gpu_identity_fails_closed_without_allocation() -> None:
    with pytest.raises(ScheduleStateError, match="GPU UUID or scheduler"):
        cuda_accelerator_identity(
            environment={"CUDA_VISIBLE_DEVICES": "0"},
            hostname="gpu01",
        )
    assert cuda_accelerator_identity(
        environment={
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_GPUS": "2",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        hostname="gpu01",
    ) == "cuda:slurm:gpu01:2"


def test_existing_mps_lease_payload_remains_v1_compatible(tmp_path) -> None:
    path = tmp_path / "mps.lock"
    with MPSLease(path, run_id="legacy-run"):
        payload = read_json(path)
        assert payload["schema_name"] == "MPSLease"
        assert payload["schema_version"] == "1.0"
        assert "accelerator_key" not in payload
