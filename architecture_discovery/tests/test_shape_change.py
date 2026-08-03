from dataclasses import replace
from pathlib import Path

import torch

from common.training_config import SMOKE_TRAIN_V1, TrainingSeedBundle
from common.trainer import train_candidate_in_process


ROOT = Path(__file__).resolve().parents[1]


def test_shape_changing_candidate_completes_fresh_smoke_training(tmp_path):
    source = (ROOT / "common" / "initial_candidate.py").read_text()
    changed = source.replace("d_model=16,", "d_model=32,", 1)
    candidate = tmp_path / "shape_candidate.py"
    candidate.write_text(changed)
    profile = replace(
        SMOKE_TRAIN_V1,
        name="shape_smoke_test",
        max_steps=2,
        validation_interval=2,
        checkpoint_interval=2,
        validation_examples=4,
        global_batch_size=4,
    )
    output = tmp_path / "training"
    result = train_candidate_in_process(
        candidate_path=candidate,
        output_dir=output,
        profile=profile,
        seeds=TrainingSeedBundle.from_run_seed(19),
        requested_device="cpu",
        allow_cpu_for_tests=True,
    )
    assert result.success, result.error
    checkpoint = torch.load(
        result.checkpoint_path, map_location="cpu", weights_only=False
    )
    assert checkpoint["model_state"]["token_emb.weight"].shape[1] == 32
