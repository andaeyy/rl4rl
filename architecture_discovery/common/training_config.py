"""Frozen candidate-training profiles and deterministic seed derivation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
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
    accelerator_memory_fraction: float | None
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
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    cublas_workspace_config: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned profile payload used for identity hashing.

        Version-1 MPS profiles retain their exact historical key set. Adding
        CUDA fields to this dataclass must not silently change their hashes.
        """

        payload = asdict(self)
        if self.version == "1":
            payload["mps_memory_fraction"] = payload.pop(
                "accelerator_memory_fraction"
            )
            for field_name in (
                "cudnn_deterministic",
                "cudnn_benchmark",
                "allow_tf32",
                "cublas_workspace_config",
            ):
                payload.pop(field_name)
        return payload

    @property
    def profile_hash(self) -> str:
        return stable_hash(self.to_dict())

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
            raise ValueError("training profiles require either MPS or CUDA")
        if self.accelerator_memory_fraction is not None and not (
            0.0 < self.accelerator_memory_fraction <= 1.0
        ):
            raise ValueError(
                "accelerator_memory_fraction must be in the interval (0, 1]"
            )
        if self.device_requirement == "cuda":
            if not self.deterministic_algorithms:
                raise ValueError("CUDA profiles require deterministic algorithms")
            if not self.cudnn_deterministic or self.cudnn_benchmark:
                raise ValueError(
                    "CUDA profiles require deterministic cuDNN with benchmarking off"
                )
            if self.allow_tf32:
                raise ValueError("the frozen float32 CUDA profile forbids TF32")
            if self.cublas_workspace_config not in {":4096:8", ":16:8"}:
                raise ValueError(
                    "deterministic CUDA requires a supported CUBLAS workspace config"
                )


@dataclass(frozen=True)
class TrainingSeedBundle:
    model_initialization_seed: int
    training_data_seed: int
    development_set_seed: int
    dataloader_seed: int

    @classmethod
    def from_run_seed(cls, run_seed: int) -> TrainingSeedBundle:
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
    accelerator_kind: str
    peak_accelerator_allocated_bytes: int | None
    current_accelerator_allocated_bytes: int | None
    reserved_accelerator_allocated_bytes: int | None
    accelerator_total_memory_bytes: int | None
    accelerator_fingerprint: dict[str, Any]
    parameter_count_metadata: int
    checkpoint_path: str
    checkpoint_sha256: str
    event_log_path: str
    unsupported_operation_fallback: bool
    scientific: bool
    hardware_matched: bool
    cleanup_completed: bool
    schema_name: str = "TrainingResult"
    schema_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.schema_version == "1.0":
            payload.pop("schema_name")
            payload.pop("schema_version")
            payload.pop("accelerator_kind")
            payload.pop("accelerator_fingerprint")
            payload["peak_mps_allocated_bytes"] = payload.pop(
                "peak_accelerator_allocated_bytes"
            )
            payload["current_mps_allocated_bytes"] = payload.pop(
                "current_accelerator_allocated_bytes"
            )
            payload["driver_mps_allocated_bytes"] = payload.pop(
                "reserved_accelerator_allocated_bytes"
            )
            payload["recommended_mps_memory_bytes"] = payload.pop(
                "accelerator_total_memory_bytes"
            )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingResult:
        values = dict(payload)
        if "peak_mps_allocated_bytes" in values:
            values["peak_accelerator_allocated_bytes"] = values.pop(
                "peak_mps_allocated_bytes"
            )
            values["current_accelerator_allocated_bytes"] = values.pop(
                "current_mps_allocated_bytes"
            )
            values["reserved_accelerator_allocated_bytes"] = values.pop(
                "driver_mps_allocated_bytes"
            )
            values["accelerator_total_memory_bytes"] = values.pop(
                "recommended_mps_memory_bytes"
            )
            values["accelerator_kind"] = str(values.get("device", "mps"))
            values["accelerator_fingerprint"] = {}
            values["schema_name"] = "TrainingResult"
            values["schema_version"] = "1.0"
        if values.get("schema_name", "TrainingResult") != "TrainingResult":
            raise ValueError("training result has an unsupported schema name")
        if values.get("schema_version", "2.0") not in {"1.0", "2.0"}:
            raise ValueError("training result has an unsupported schema version")
        return cls(**values)


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
    accelerator_memory_fraction=None,
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
    accelerator_memory_fraction=None,
    scientific=False,
)

FULL_TRAIN_CUDA_V2 = TrainingProfile(
    name="full_train_cuda_v2",
    version="2",
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
    accelerator_memory_fraction=None,
    scientific=True,
    cublas_workspace_config=":4096:8",
)

SMOKE_TRAIN_CUDA_V2 = TrainingProfile(
    name="smoke_train_cuda_v2",
    version="2",
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
    # The bounded Modal resume smoke retains the first nonterminal checkpoint
    # and must prove that a resumed optimizer advances beyond it.
    checkpoint_interval=5,
    maximum_wall_seconds=60,
    dtype="float32",
    deterministic_algorithms=True,
    device_requirement="cuda",
    accelerator_memory_fraction=None,
    scientific=False,
    cublas_workspace_config=":4096:8",
)

# Deliberately tiny CUDA-only profile for the exploratory Modal lane.  This is
# useful for plumbing and hypothesis generation, but it is never a scientific
# ranking profile and must not be used to unlock the formal study gates.
EXPLORATORY_TRAIN_CUDA_V2 = TrainingProfile(
    name="exploratory_train_cuda_v2",
    version="2",
    max_steps=25,
    global_batch_size=16,
    microbatch_size=None,
    gradient_accumulation_steps=1,
    peak_learning_rate=0.001,
    adamw_betas=(0.9, 0.98),
    weight_decay=0.1,
    warmup_steps=2,
    scheduler="cosine_decay_to_zero",
    gradient_clip_norm=1.0,
    validation_interval=25,
    validation_examples=24,
    checkpoint_interval=25,
    maximum_wall_seconds=120,
    dtype="float32",
    deterministic_algorithms=True,
    device_requirement="cuda",
    accelerator_memory_fraction=None,
    scientific=False,
    cublas_workspace_config=":4096:8",
)

PROFILES = {
    FULL_TRAIN_V1.name: FULL_TRAIN_V1,
    SMOKE_TRAIN_V1.name: SMOKE_TRAIN_V1,
    FULL_TRAIN_CUDA_V2.name: FULL_TRAIN_CUDA_V2,
    SMOKE_TRAIN_CUDA_V2.name: SMOKE_TRAIN_CUDA_V2,
    EXPLORATORY_TRAIN_CUDA_V2.name: EXPLORATORY_TRAIN_CUDA_V2,
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
