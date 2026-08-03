"""Frozen, machine-readable statistical analysis plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
import math
from pathlib import Path
from typing import Any

from study.serialization import (
    content_hash,
    create_json_exclusive,
    read_json,
    require_bool,
    require_int,
    require_str,
)
from study.contracts import ConditionId


PRIMARY_OUTCOME = "unique_qualifying_mechanism_clusters_per_assigned_run"
INDEPENDENT_UNIT = "complete_assigned_run"


class CountModel(StrEnum):
    POISSON = "blocked_poisson"
    NEGATIVE_BINOMIAL = "blocked_negative_binomial"


class MultiplicityMethod(StrEnum):
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    BONFERRONI = "bonferroni"


@dataclass(frozen=True)
class AnalysisPlan:
    plan_id: str
    study_id: str
    primary_contrast: tuple[str, str]
    count_model: CountModel
    alpha: float
    target_power: float
    smallest_effect_rate_ratio: float
    multiplicity_method: MultiplicityMethod
    hypothesis_family: tuple[str, ...]
    randomization_draws: int
    simulation_seed: int
    pi_decision_record_hash: str
    scientific: bool
    primary_outcome: str = field(default=PRIMARY_OUTCOME, init=False)
    independent_unit: str = field(default=INDEPENDENT_UNIT, init=False)
    schema_name: str = field(default="AnalysisPlan", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_bool(self.scientific, "scientific")
        require_str(self.plan_id, "plan_id")
        require_str(self.study_id, "study_id")
        require_int(self.randomization_draws, "randomization_draws")
        require_int(self.simulation_seed, "simulation_seed")
        for field_name in (
            "alpha",
            "target_power",
            "smallest_effect_rate_ratio",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field_name} must be a finite number")
        if not self.plan_id or not self.study_id:
            raise ValueError("plan_id and study_id cannot be empty")
        if len(self.primary_contrast) != 2:
            raise ValueError("primary_contrast must be (target, reference)")
        for index, value in enumerate(self.primary_contrast):
            require_str(value, f"primary_contrast[{index}]")
        if not all(value.strip() for value in self.primary_contrast):
            raise ValueError("contrast condition IDs cannot be empty")
        try:
            tuple(ConditionId(value) for value in self.primary_contrast)
        except ValueError as error:
            raise ValueError("primary_contrast must contain C0-C3 condition IDs") from error
        if self.primary_contrast[0] == self.primary_contrast[1]:
            raise ValueError("contrast target and reference must differ")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if not 0.0 < self.target_power < 1.0:
            raise ValueError("target_power must lie strictly between zero and one")
        if self.smallest_effect_rate_ratio <= 0:
            raise ValueError("smallest_effect_rate_ratio must be positive")
        if self.randomization_draws < 1:
            raise ValueError("randomization_draws must be positive")
        for index, hypothesis in enumerate(self.hypothesis_family):
            require_str(hypothesis, f"hypothesis_family[{index}]")
        if (
            not self.hypothesis_family
            or len(set(self.hypothesis_family)) != len(self.hypothesis_family)
            or any(not hypothesis.strip() for hypothesis in self.hypothesis_family)
        ):
            raise ValueError("hypothesis_family must contain unique non-empty IDs")
        if not self.pi_decision_record_hash:
            raise ValueError("PI decision record hash cannot be empty")
        require_str(self.pi_decision_record_hash, "pi_decision_record_hash")
        if self.scientific and self.pi_decision_record_hash.startswith("offline-toy"):
            raise ValueError("scientific plans cannot use toy PI decisions")
        if self.scientific and (
            len(self.pi_decision_record_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.pi_decision_record_hash
            )
        ):
            raise ValueError(
                "scientific pi_decision_record_hash must be a lowercase SHA-256"
            )

    @property
    def plan_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "study_id": self.study_id,
            "primary_contrast": list(self.primary_contrast),
            "count_model": self.count_model.value,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "smallest_effect_rate_ratio": self.smallest_effect_rate_ratio,
            "multiplicity_method": self.multiplicity_method.value,
            "hypothesis_family": list(self.hypothesis_family),
            "randomization_draws": self.randomization_draws,
            "simulation_seed": self.simulation_seed,
            "pi_decision_record_hash": self.pi_decision_record_hash,
            "scientific": self.scientific,
            "primary_outcome": self.primary_outcome,
            "independent_unit": self.independent_unit,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnalysisPlan:
        if payload.get("schema_name") != "AnalysisPlan":
            raise ValueError("expected AnalysisPlan schema")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported AnalysisPlan schema version")
        if payload.get("primary_outcome") != PRIMARY_OUTCOME:
            raise ValueError("primary outcome differs from the run-level contract")
        if payload.get("independent_unit") != INDEPENDENT_UNIT:
            raise ValueError("independent unit must be a complete assigned run")
        return cls(
            plan_id=require_str(payload["plan_id"], "plan_id"),
            study_id=require_str(payload["study_id"], "study_id"),
            primary_contrast=tuple(
                require_str(value, "primary contrast")
                for value in payload["primary_contrast"]
            ),
            count_model=CountModel(
                require_str(payload["count_model"], "count_model")
            ),
            alpha=payload["alpha"],
            target_power=payload["target_power"],
            smallest_effect_rate_ratio=payload["smallest_effect_rate_ratio"],
            multiplicity_method=MultiplicityMethod(
                require_str(payload["multiplicity_method"], "multiplicity_method")
            ),
            hypothesis_family=tuple(
                require_str(value, "hypothesis family member")
                for value in payload["hypothesis_family"]
            ),
            randomization_draws=require_int(
                payload["randomization_draws"], "randomization_draws"
            ),
            simulation_seed=require_int(payload["simulation_seed"], "simulation_seed"),
            pi_decision_record_hash=require_str(
                payload["pi_decision_record_hash"], "pi_decision_record_hash"
            ),
            scientific=require_bool(payload["scientific"], "scientific"),
        )

    @classmethod
    def toy(cls) -> AnalysisPlan:
        """Explicitly non-scientific plan for offline integration tests."""

        return cls(
            plan_id="offline-toy-analysis-v1",
            study_id="offline-toy-study",
            primary_contrast=("C1", "C0"),
            count_model=CountModel.NEGATIVE_BINOMIAL,
            alpha=0.05,
            target_power=0.8,
            smallest_effect_rate_ratio=2.0,
            multiplicity_method=MultiplicityMethod.HOLM,
            hypothesis_family=("toy-primary", "toy-secondary"),
            randomization_draws=999,
            simulation_seed=17,
            pi_decision_record_hash="offline-toy-decisions",
            scientific=False,
        )


@dataclass(frozen=True)
class FrozenAnalysisPlan:
    plan: AnalysisPlan
    frozen_at_utc: str
    plan_hash: str
    schema_name: str = field(default="FrozenAnalysisPlan", init=False)
    schema_version: str = field(default="1.0", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "frozen_at_utc": self.frozen_at_utc,
            "plan_hash": self.plan_hash,
            "plan": self.plan.to_dict(),
        }


def freeze_analysis_plan(path: str | Path, plan: AnalysisPlan) -> FrozenAnalysisPlan:
    """Create an immutable-once plan file and refuse every overwrite attempt."""

    frozen = FrozenAnalysisPlan(
        plan=plan,
        frozen_at_utc=datetime.now(UTC).isoformat(),
        plan_hash=plan.plan_hash,
    )
    create_json_exclusive(path, frozen.to_dict())
    return frozen


def load_frozen_analysis_plan(path: str | Path) -> FrozenAnalysisPlan:
    payload = read_json(path)
    if payload.get("schema_name") != "FrozenAnalysisPlan":
        raise ValueError("expected FrozenAnalysisPlan schema")
    plan = AnalysisPlan.from_dict(payload["plan"])
    stored_hash = require_str(payload["plan_hash"], "plan_hash")
    if stored_hash != plan.plan_hash:
        raise ValueError("frozen analysis plan hash mismatch; file may be mutated")
    frozen_at = require_str(payload["frozen_at_utc"], "frozen_at_utc")
    if not frozen_at:
        raise ValueError("frozen plan lacks its freeze time")
    return FrozenAnalysisPlan(
        plan=plan,
        frozen_at_utc=frozen_at,
        plan_hash=stored_hash,
    )
