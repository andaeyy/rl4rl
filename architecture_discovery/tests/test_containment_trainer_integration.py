import json

from common.training_config import FULL_TRAIN_V1, TrainingSeedBundle
from common.trainer import train_candidate_in_process


def test_scientific_python_training_fails_before_candidate_import(tmp_path):
    output = tmp_path / "blocked-scientific-training"
    result = train_candidate_in_process(
        candidate_path="common/initial_candidate.py",
        output_dir=output,
        profile=FULL_TRAIN_V1,
        seeds=TrainingSeedBundle.from_run_seed(1),
        requested_device="mps",
        allow_cpu_for_tests=False,
    )
    assert not result.success
    assert result.failure_stage == "containment_unproven"
    assert result.steps_completed == 0
    manifest = json.loads((output / "training_manifest.json").read_text())
    assert not manifest["containment_decision"]["allowed"]
    assert manifest["containment_decision"]["scientific"]
    assert manifest["candidate_initialization"] == "from_scratch"

