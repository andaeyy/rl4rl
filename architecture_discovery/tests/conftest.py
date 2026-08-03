from dataclasses import replace
from pathlib import Path

import pytest

from common.training_config import SMOKE_TRAIN_V1, TrainingSeedBundle
from common.trainer import train_candidate_in_process


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cpu_smoke_training(tmp_path_factory):
    output = tmp_path_factory.mktemp("cpu-smoke-training")
    result = train_candidate_in_process(
        candidate_path=ROOT / "common" / "initial_candidate.py",
        output_dir=output,
        profile=SMOKE_TRAIN_V1,
        seeds=TrainingSeedBundle.from_run_seed(17),
        requested_device="cpu",
        allow_cpu_for_tests=True,
    )
    assert result.success, result.error
    return result, output
