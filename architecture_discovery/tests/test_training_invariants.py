import inspect
from pathlib import Path

import yaml

from common.task_adapter import DEFAULT_TASK
from common.training_config import (
    SEED_DERIVATION_METHOD,
    TrainingSeedBundle,
    get_training_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_controllers_resolve_the_same_frozen_training_profile():
    names = (
        "greedy_autoresearch",
        "openevolve_generic",
        "openevolve_semantic",
    )
    references = [
        yaml.safe_load(
            (ROOT / "agents" / name / "config.yaml").read_text()
        )["training"]
        for name in names
    ]
    assert references[0] == references[1] == references[2]
    profile = get_training_profile(references[0]["profile"])
    assert profile.version == references[0]["profile_version"]
    assert references[0]["task_adapter"] == DEFAULT_TASK.version
    assert references[0]["seed_derivation"] == SEED_DERIVATION_METHOD
    assert profile.max_steps == 30_000
    assert profile.device_requirement == "mps"


def test_seed_bundle_is_condition_independent_and_stable():
    assert TrainingSeedBundle.from_run_seed(1) == TrainingSeedBundle.from_run_seed(1)
    assert (
        TrainingSeedBundle.from_run_seed(1).bundle_hash
        == "20e9cf13691d96bfe09725776965e6dcce315328f6f81b4671b31b31b1b5482f"
    )


def test_training_code_never_reads_official_or_shadow_seeds():
    for name in ("trainer.py", "training_data.py"):
        source = (ROOT / "common" / name).read_text()
        assert "private_eval" not in source
        assert "DISCOVERY_SHADOW_SEED" not in source
        assert "2025" not in source
