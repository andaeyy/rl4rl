"""Structured evaluation result shared by all controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DiscoveryEvaluation:
    execution_ok: bool = False
    transformer_valid: bool = False
    official_accuracy: float = 0.0
    shadow_accuracy: float = 0.0
    edge_accuracy: float = 0.0
    carry_accuracy: float = 0.0
    robustness_score: float = 0.0
    qualifies: bool = False
    combined_score: float = 0.0
    parameter_count_metadata: int = 0
    train_seconds: float = 0.0
    verify_seconds: float = 0.0
    failure_stage: str = ""
    infrastructure_failure: bool = False
    descriptor_vector: dict[str, str] = field(default_factory=dict)
    descriptor_confidence: dict[str, float] = field(default_factory=dict)
    semantic_metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def metrics(self) -> dict[str, float]:
        values = {
            "execution_ok": float(self.execution_ok),
            "transformer_valid": float(self.transformer_valid),
            "official_accuracy": self.official_accuracy,
            "shadow_accuracy": self.shadow_accuracy,
            "edge_accuracy": self.edge_accuracy,
            "carry_accuracy": self.carry_accuracy,
            "robustness_score": self.robustness_score,
            "qualifies": float(self.qualifies),
            "combined_score": self.combined_score,
        }
        values.update(self.semantic_metrics)
        return values
