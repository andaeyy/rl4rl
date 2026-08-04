"""Frozen candidate-training profiles and deterministic seed derivation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


SEED_DERIVATION_METHOD = "sha256-v1"
CHECKPOINT_SELECTION_RULE = (
    "higher_development_exact_match_then_lower_development_loss_then_earlier_step"
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def derive_seed(run_seed: int, namespace: str) -> int:
    payload = f"{SEED_DERIVATION_METHOD}|{int(run_seed)}|{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    version: str
    max_steps: int
    global_batch_size: int
    microbatch_size: int | None
    gradient_accumulation_steps: int
    peak_learning_rate: float
    adamw_betas: tuple[float, float]
    weight_decay: float
    warmup_steps: int
    scheduler: str
    gradient_clip_norm: float
    validation_interval: int
    validation_examples: int
    checkpoint_interval: int
    maximum_wall_seconds: int
    dtype: str
    deterministic_algorithms: bool
    device_requirement: str
    mps_memory_fraction: float | None
    scientific: bool
    optimizer: str = "AdamW"
    num_workers: int = 0
    loss: str = "answer_only_cross_entropy"
    min_operand_digits: int = 1
    max_operand_digits: int = 10
    mixed_precision: bool = False
    torch_compile: bool = False
    automatic_batch_size_reduction: bool = False
    cpu_fallback: bool = False
    checkpoint_selection_rule: str = CHECKPOINT_SELECTION_RULE

    @property
    def profile_hash(self) -> str:
        return stable_hash(asdict(self))

    def validate(self) -> None:
        if self.optimizer != "AdamW":
            raise ValueError("only frozen AdamW training is supported")
        if self.scheduler != "cosine_decay_to_zero":
            raise ValueError("only the frozen cosine schedule is supported")
        microbatch = self.microbatch_size or self.global_batch_size
        if microbatch * self.gradient_accumulation_steps != self.global_batch_size:
            raise ValueError(
                "microbatch_size * gradient_accumulation_steps must equal "
                "global_batch_size"
            )
        if self.dtype != "float32" or self.mixed_precision or self.torch_compile:
            raise ValueError("Phase-1 profiles require uncompiled float32 training")
        if self.automatic_batch_size_reduction or self.cpu_fallback:
            raise ValueError("automatic batch reduction and CPU fallback are forbidden")
        if self.device_requirement not in {"mps", "cuda"}:
            raise ValueError(
                "training profiles require an explicit accelerator backend"
            )
        if self.device_requirement == "cuda" and not self.deterministic_algorithms:
            raise ValueError("CUDA profiles require deterministic algorithms")


@dataclass(frozen=True)
class TrainingSeedBundle:
    model_initialization_seed: int
    training_data_seed: int
    development_set_seed: int
    dataloader_seed: int

    @classmethod
    def from_run_seed(cls, run_seed: int) -> "TrainingSeedBundle":
        return cls(
            model_initialization_seed=derive_seed(run_seed, "model_initialization"),
            training_data_seed=derive_seed(run_seed, "training_data"),
            development_set_seed=derive_seed(run_seed, "public_development"),
            dataloader_seed=derive_seed(run_seed, "dataloader"),
        )

    @property
    def bundle_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class TrainingResult:
    success: bool
    failure_stage: str
    error: str
    profile_name: str
    profile_version: str
    profile_hash: str
    candidate_source_hash: str
    initialization_seed: int
    data_seed: int
    development_seed: int
    dataloader_seed: int
    device: str
    dtype: str
    steps_completed: int
    examples_processed: int
    best_development_step: int
    best_development_exact_match_accuracy: float
    best_development_loss: float
    final_training_loss: float
    train_seconds: float
    peak_mps_allocated_bytes: int | None
    current_mps_allocated_bytes: int | None
    driver_mps_allocated_bytes: int | None
    recommended_mps_memory_bytes: int | None
    parameter_count_metadata: int
    checkpoint_path: str
    checkpoint_sha256: str
    event_log_path: str
    unsupported_operation_fallback: bool
    scientific: bool
    hardware_matched: bool
    cleanup_completed: bool
    accelerator_telemetry_schema_version: str = ""
    accelerator_telemetry: dict[str, Any] = field(default_factory=dict)
    timing_synchronized: bool = False
    initial_parameter_sha256: str = ""
    final_parameter_sha256: str = ""
    parameters_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FULL_TRAIN_V1 = TrainingProfile(
    name="full_train_v1",
    version="1",
    max_steps=30_000,
    global_batch_size=512,
    microbatch_size=None,
    gradient_accumulation_steps=1,
    peak_learning_rate=0.001,
    adamw_betas=(0.9, 0.98),
    weight_decay=0.1,
    warmup_steps=300,
    scheduler="cosine_decay_to_zero",
    gradient_clip_norm=1.0,
    validation_interval=1_000,
    validation_examples=2_000,
    checkpoint_interval=1_000,
    maximum_wall_seconds=1_800,
    dtype="float32",
    deterministic_algorithms=True,
    device_requirement="mps",
    mps_memory_fraction=None,
    scientific=True,
)

SMOKE_TRAIN_V1 = TrainingProfile(
    name="smoke_train_v1",
    version="1",
    max_steps=10,
    global_batch_size=16,
    microbatch_size=None,
    gradient_accumulation_steps=1,
    peak_learning_rate=0.001,
    adamw_betas=(0.9, 0.98),
    weight_decay=0.1,
    warmup_steps=2,
    scheduler="cosine_decay_to_zero",
    gradient_clip_norm=1.0,
    validation_interval=10,
    validation_examples=24,
    checkpoint_interval=10,
    maximum_wall_seconds=60,
    dtype="float32",
    deterministic_algorithms=True,
    device_requirement="mps",
    mps_memory_fraction=None,
    scientific=False,
)

FULL_TRAIN_CUDA_A40_V1 = TrainingProfile(
    name="full_train_cuda_a40_v1",
    version="1",
    max_steps=30_000,
    global_batch_size=512,
    microbatch_size=None,
    gradient_accumulation_steps=1,
    peak_learning_rate=0.001,
    adamw_betas=(0.9, 0.98),
    weight_decay=0.1,
    warmup_steps=300,
    scheduler="cosine_decay_to_zero",
    gradient_clip_norm=1.0,
    validation_interval=1_000,
    validation_examples=2_000,
    checkpoint_interval=1_000,
    maximum_wall_seconds=1_800,
    dtype="float32",
    deterministic_algorithms=True,
    device_requirement="cuda",
    mps_memory_fraction=None,
    scientific=True,
)

SMOKE_TRAIN_CUDA_V1 = TrainingProfile(
    name="smoke_train_cuda_v1",
    version="1",
    max_steps=10,
    global_batch_size=16,
    microbatch_size=None,
    gradient_accumulation_steps=1,
    peak_learning_rate=0.001,
    adamw_betas=(0.9, 0.98),
    weight_decay=0.1,
    warmup_steps=2,
    scheduler="cosine_decay_to_zero",
    gradient_clip_norm=1.0,
    validation_interval=10,
    validation_examples=24,
    checkpoint_interval=10,
    maximum_wall_seconds=60,
    dtype="float32",
    deterministic_algorithms=True,
    device_requirement="cuda",
    mps_memory_fraction=None,
    scientific=False,
)

PROFILES = {
    FULL_TRAIN_V1.name: FULL_TRAIN_V1,
    SMOKE_TRAIN_V1.name: SMOKE_TRAIN_V1,
    FULL_TRAIN_CUDA_A40_V1.name: FULL_TRAIN_CUDA_A40_V1,
    SMOKE_TRAIN_CUDA_V1.name: SMOKE_TRAIN_CUDA_V1,
}


def get_training_profile(name: str) -> TrainingProfile:
    try:
        profile = PROFILES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown training profile {name!r}; choose one of {sorted(PROFILES)}"
        ) from error
    profile.validate()
    return profile
