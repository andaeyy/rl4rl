from dataclasses import asdict

import pytest

from common.training_config import (
    FULL_TRAIN_CUDA_A40_V1,
    FULL_TRAIN_V1,
    PROFILES,
    SMOKE_TRAIN_CUDA_V1,
    SMOKE_TRAIN_V1,
    get_training_profile,
)
from common.trusted_candidate import (
    TRUSTED_INITIAL_CANDIDATE_PATH,
    TRUSTED_INITIAL_CANDIDATE_SHA256,
    validate_trusted_initial_candidate,
)


FROZEN_PROFILE_FIELDS = {
    "name",
    "version",
    "max_steps",
    "global_batch_size",
    "microbatch_size",
    "gradient_accumulation_steps",
    "peak_learning_rate",
    "adamw_betas",
    "weight_decay",
    "warmup_steps",
    "scheduler",
    "gradient_clip_norm",
    "validation_interval",
    "validation_examples",
    "checkpoint_interval",
    "maximum_wall_seconds",
    "dtype",
    "deterministic_algorithms",
    "device_requirement",
    "mps_memory_fraction",
    "scientific",
    "optimizer",
    "num_workers",
    "loss",
    "min_operand_digits",
    "max_operand_digits",
    "mixed_precision",
    "torch_compile",
    "automatic_batch_size_reduction",
    "cpu_fallback",
    "checkpoint_selection_rule",
}


def _without_condition_identity(profile):
    payload = asdict(profile)
    payload.pop("name")
    payload.pop("device_requirement")
    return payload


def test_original_mps_profiles_keep_exact_v1_hashes_and_serialized_fields():
    assert FULL_TRAIN_V1.profile_hash == (
        "046034a7949f3563fc13dcb38df4b34e997cb5a1ffe6b90e755e2f44bfd9f06e"
    )
    assert SMOKE_TRAIN_V1.profile_hash == (
        "1a2b04bcb966f4189f90d6b8f6ef3aa8f83fb537f0f031004d0e58d69192cb61"
    )
    assert set(asdict(FULL_TRAIN_V1)) == FROZEN_PROFILE_FIELDS
    assert set(asdict(SMOKE_TRAIN_V1)) == FROZEN_PROFILE_FIELDS
    assert FULL_TRAIN_V1.device_requirement == "mps"
    assert SMOKE_TRAIN_V1.device_requirement == "mps"


def test_full_cuda_a40_profile_changes_only_condition_identity_from_full_mps():
    assert FULL_TRAIN_CUDA_A40_V1.name == "full_train_cuda_a40_v1"
    assert FULL_TRAIN_CUDA_A40_V1.device_requirement == "cuda"
    assert _without_condition_identity(FULL_TRAIN_CUDA_A40_V1) == (
        _without_condition_identity(FULL_TRAIN_V1)
    )
    assert FULL_TRAIN_CUDA_A40_V1.profile_hash != FULL_TRAIN_V1.profile_hash
    assert FULL_TRAIN_CUDA_A40_V1.max_steps == 30_000
    assert FULL_TRAIN_CUDA_A40_V1.global_batch_size == 512
    assert FULL_TRAIN_CUDA_A40_V1.optimizer == "AdamW"
    assert FULL_TRAIN_CUDA_A40_V1.scientific


def test_cuda_smoke_is_exactly_ten_steps_and_non_scientific():
    assert SMOKE_TRAIN_CUDA_V1.name == "smoke_train_cuda_v1"
    assert SMOKE_TRAIN_CUDA_V1.device_requirement == "cuda"
    assert _without_condition_identity(SMOKE_TRAIN_CUDA_V1) == (
        _without_condition_identity(SMOKE_TRAIN_V1)
    )
    assert SMOKE_TRAIN_CUDA_V1.max_steps == 10
    assert not SMOKE_TRAIN_CUDA_V1.scientific
    assert SMOKE_TRAIN_CUDA_V1.profile_hash not in {
        FULL_TRAIN_V1.profile_hash,
        SMOKE_TRAIN_V1.profile_hash,
        FULL_TRAIN_CUDA_A40_V1.profile_hash,
    }


@pytest.mark.parametrize(
    "profile",
    [FULL_TRAIN_CUDA_A40_V1, SMOKE_TRAIN_CUDA_V1],
)
def test_cuda_profiles_forbid_protocol_shortcuts(profile):
    profile.validate()
    assert profile.dtype == "float32"
    assert profile.deterministic_algorithms
    assert not profile.mixed_precision
    assert not profile.torch_compile
    assert not profile.automatic_batch_size_reduction
    assert not profile.cpu_fallback


def test_cuda_profiles_are_additive_registry_entries():
    assert set(PROFILES) == {
        "full_train_v1",
        "smoke_train_v1",
        "full_train_cuda_a40_v1",
        "smoke_train_cuda_v1",
    }
    assert get_training_profile("full_train_cuda_a40_v1") is FULL_TRAIN_CUDA_A40_V1
    assert get_training_profile("smoke_train_cuda_v1") is SMOKE_TRAIN_CUDA_V1


def test_cuda_smoke_trusted_candidate_identity_is_pinned():
    assert (
        validate_trusted_initial_candidate(TRUSTED_INITIAL_CANDIDATE_PATH)
        == TRUSTED_INITIAL_CANDIDATE_SHA256
    )
