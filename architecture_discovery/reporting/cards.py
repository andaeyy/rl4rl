"""Model and data cards for exact study artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research_ledger.records import require_identifier, require_sha256, require_text


@dataclass(frozen=True)
class ModelCard:
    model_card_id: str
    candidate_id: str
    architecture_signature_sha256: str
    training_configuration_sha256: str
    checkpoint_sha256: str
    parameter_count_metadata: int
    intended_use: str
    evaluation_scope: str
    limitations: tuple[str, ...]
    parameter_count_role: str = field(default="descriptive_metadata_only", init=False)
    schema_name: str = field(default="ArchitectureDiscoveryModelCard", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_identifier(self.model_card_id, "model_card_id")
        require_identifier(self.candidate_id, "candidate_id")
        for field_name in (
            "architecture_signature_sha256",
            "training_configuration_sha256",
            "checkpoint_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        if (
            not isinstance(self.parameter_count_metadata, int)
            or isinstance(self.parameter_count_metadata, bool)
            or self.parameter_count_metadata < 0
        ):
            raise ValueError("parameter_count_metadata must be a nonnegative integer")
        require_text(self.intended_use, "intended_use")
        require_text(self.evaluation_scope, "evaluation_scope")
        limitations = tuple(self.limitations)
        if not limitations:
            raise ValueError("model cards require explicit limitations")
        for limitation in limitations:
            require_text(limitation, "model limitation")
        object.__setattr__(self, "limitations", limitations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "model_card_id": self.model_card_id,
            "candidate_id": self.candidate_id,
            "architecture_signature_sha256": self.architecture_signature_sha256,
            "training_configuration_sha256": self.training_configuration_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "parameter_count_metadata": self.parameter_count_metadata,
            "parameter_count_role": self.parameter_count_role,
            "intended_use": self.intended_use,
            "evaluation_scope": self.evaluation_scope,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class DataCard:
    data_card_id: str
    dataset_id: str
    generator_code_sha256: str
    split_policy_sha256: str
    seed_manifest_sha256: str
    disjointness_evidence_sha256: str
    data_role: str
    intended_use: str
    limitations: tuple[str, ...]
    schema_name: str = field(default="ArchitectureDiscoveryDataCard", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_identifier(self.data_card_id, "data_card_id")
        require_identifier(self.dataset_id, "dataset_id")
        require_identifier(self.data_role, "data_role")
        for field_name in (
            "generator_code_sha256",
            "split_policy_sha256",
            "seed_manifest_sha256",
            "disjointness_evidence_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        require_text(self.intended_use, "intended_use")
        limitations = tuple(self.limitations)
        if not limitations:
            raise ValueError("data cards require explicit limitations")
        for limitation in limitations:
            require_text(limitation, "data limitation")
        object.__setattr__(self, "limitations", limitations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "data_card_id": self.data_card_id,
            "dataset_id": self.dataset_id,
            "generator_code_sha256": self.generator_code_sha256,
            "split_policy_sha256": self.split_policy_sha256,
            "seed_manifest_sha256": self.seed_manifest_sha256,
            "disjointness_evidence_sha256": self.disjointness_evidence_sha256,
            "data_role": self.data_role,
            "intended_use": self.intended_use,
            "limitations": list(self.limitations),
        }
