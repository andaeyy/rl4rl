import hashlib
import json
from pathlib import Path

import torch

from common.evaluator import load_candidate


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_updates_parameters_and_records_positive_time(cpu_smoke_training):
    result, _ = cpu_smoke_training
    module = load_candidate(ROOT / "common" / "initial_candidate.py")
    initial, _ = module.build_untrained_model(result.initialization_seed)
    checkpoint = torch.load(
        result.checkpoint_path, map_location="cpu", weights_only=False
    )
    assert any(
        not torch.equal(initial.state_dict()[name], checkpoint["model_state"][name])
        for name in initial.state_dict()
    )
    assert result.train_seconds > 0
    assert result.examples_processed > 0
    assert result.parameter_count_metadata > 0


def test_training_artifact_set_and_checkpoint_hash(cpu_smoke_training):
    result, output = cpu_smoke_training
    expected = {
        "training_manifest.json",
        "training_events.jsonl",
        "best_checkpoint.pt",
        "latest_resume_checkpoint.pt",
        "training_summary.json",
        "candidate_source.py",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    digest = hashlib.sha256(Path(result.checkpoint_path).read_bytes()).hexdigest()
    assert digest == result.checkpoint_sha256
    event = json.loads((output / "training_events.jsonl").read_text().splitlines()[0])
    assert {
        "timestamp",
        "optimizer_step",
        "examples_processed",
        "loss",
        "learning_rate",
        "gradient_norm",
        "validation_loss",
        "validation_exact_match_accuracy",
        "elapsed_seconds",
        "current_mps_allocated_bytes",
        "driver_mps_allocated_bytes",
        "checkpoint_decision",
    }.issubset(event)


def test_cpu_smoke_is_marked_engineering_only(cpu_smoke_training):
    result, output = cpu_smoke_training
    manifest = json.loads((output / "training_manifest.json").read_text())
    assert not result.scientific
    assert not result.hardware_matched
    assert manifest["scientific_limitations"]
