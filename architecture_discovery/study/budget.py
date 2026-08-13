"""Frozen ceilings and reconstructed resource accounting for one study run."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from study.serialization import require_int, require_str


class BudgetExceeded(RuntimeError):
    """Raised before an action would exceed a frozen scientific ceiling."""


class OpportunityStateError(RuntimeError):
    """Raised when proposal opportunities are advanced out of order."""


class OpportunityOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INVALID = "invalid"
    SCIENTIFIC_FAILURE = "scientific_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


def _require_accelerator_kind(value: object, field_name: str = "accelerator_kind") -> str:
    resolved = require_str(value, field_name)
    if not resolved or resolved != resolved.strip() or resolved != resolved.lower():
        raise ValueError(
            f"{field_name} must be a non-empty lowercase string without surrounding whitespace"
        )
    return resolved


def _require_nonnegative_number(value: object, field_name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


@dataclass(frozen=True, init=False)
class BudgetSpec:
    """Preregistered hard ceilings. No field is a model-size objective."""

    proposal_opportunities: int
    provider_attempts_per_opportunity: int
    prompt_tokens: int
    completion_tokens: int
    repairs: int
    candidate_training_attempts: int
    training_steps: int
    training_examples: int
    accelerator_kind: str
    accelerator_seconds: float
    evaluation_cases: int
    infrastructure_retries: int
    repair_attempts_per_opportunity: int = 1
    seed_evaluations: int = 1
    schema_name: str = field(default="BudgetSpec", init=False)
    schema_version: str = field(default="2.0", init=False)

    def __init__(
        self,
        proposal_opportunities: int,
        provider_attempts_per_opportunity: int,
        prompt_tokens: int,
        completion_tokens: int,
        repairs: int,
        candidate_training_attempts: int,
        training_steps: int,
        training_examples: int,
        evaluation_cases: int,
        infrastructure_retries: int,
        repair_attempts_per_opportunity: int = 1,
        seed_evaluations: int = 1,
        *,
        accelerator_kind: str | None = None,
        accelerator_seconds: float | None = None,
        mps_seconds: float | None = None,
    ) -> None:
        """Create a v2 budget, accepting ``mps_seconds`` as a source-code alias.

        Only ``from_dict`` creates a legacy v1 serialization view. Consequently,
        newly constructed records never emit the deprecated field.
        """

        if mps_seconds is not None:
            _require_nonnegative_number(mps_seconds, "mps_seconds")
            if accelerator_kind is not None and accelerator_kind != "mps":
                raise ValueError(
                    "mps_seconds cannot be combined with a non-MPS accelerator_kind"
                )
            if (
                accelerator_seconds is not None
                and accelerator_seconds != mps_seconds
            ):
                raise ValueError(
                    "mps_seconds and accelerator_seconds must agree when both are supplied"
                )
            accelerator_kind = "mps"
            accelerator_seconds = mps_seconds
        if accelerator_kind is None or accelerator_seconds is None:
            raise ValueError(
                "accelerator_kind and accelerator_seconds are required"
            )

        values = {
            "proposal_opportunities": proposal_opportunities,
            "provider_attempts_per_opportunity": provider_attempts_per_opportunity,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "repairs": repairs,
            "candidate_training_attempts": candidate_training_attempts,
            "training_steps": training_steps,
            "training_examples": training_examples,
            "accelerator_kind": accelerator_kind,
            "accelerator_seconds": accelerator_seconds,
            "evaluation_cases": evaluation_cases,
            "infrastructure_retries": infrastructure_retries,
            "repair_attempts_per_opportunity": repair_attempts_per_opportunity,
            "seed_evaluations": seed_evaluations,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "schema_name", "BudgetSpec")
        object.__setattr__(self, "schema_version", "2.0")
        self.__post_init__()

    def __post_init__(self) -> None:
        integer_fields = (
            "proposal_opportunities",
            "provider_attempts_per_opportunity",
            "prompt_tokens",
            "completion_tokens",
            "repairs",
            "candidate_training_attempts",
            "training_steps",
            "training_examples",
            "evaluation_cases",
            "infrastructure_retries",
            "repair_attempts_per_opportunity",
            "seed_evaluations",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.provider_attempts_per_opportunity < 1:
            raise ValueError("provider_attempts_per_opportunity must be at least one")
        if self.seed_evaluations != 1:
            raise ValueError("primary C0-C3 runs require exactly one seed evaluation")
        _require_accelerator_kind(self.accelerator_kind)
        _require_nonnegative_number(
            self.accelerator_seconds, "accelerator_seconds"
        )

    @property
    def mps_seconds(self) -> float:
        """Deprecated source-code alias for pre-v2 consumers."""

        return self.accelerator_seconds

    @classmethod
    def toy(cls, proposal_opportunities: int = 3) -> BudgetSpec:
        """Small offline-only fixture. These values are not scientific defaults."""

        return cls(
            proposal_opportunities=proposal_opportunities,
            provider_attempts_per_opportunity=2,
            prompt_tokens=max(1, proposal_opportunities) * 100,
            completion_tokens=max(1, proposal_opportunities) * 100,
            repairs=proposal_opportunities,
            candidate_training_attempts=proposal_opportunities + 1,
            training_steps=(proposal_opportunities + 1) * 4,
            training_examples=(proposal_opportunities + 1) * 16,
            accelerator_kind="cpu",
            accelerator_seconds=60.0,
            evaluation_cases=(proposal_opportunities + 1) * 8,
            infrastructure_retries=proposal_opportunities,
            repair_attempts_per_opportunity=1,
        )

    def to_dict(self) -> dict[str, Any]:
        common = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "proposal_opportunities": self.proposal_opportunities,
            "provider_attempts_per_opportunity": self.provider_attempts_per_opportunity,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "repairs": self.repairs,
            "candidate_training_attempts": self.candidate_training_attempts,
            "training_steps": self.training_steps,
            "training_examples": self.training_examples,
        }
        if self.schema_version == "1.0":
            compute = {"mps_seconds": self.accelerator_seconds}
        elif self.schema_version == "2.0":
            compute = {
                "accelerator_kind": self.accelerator_kind,
                "accelerator_seconds": self.accelerator_seconds,
            }
        else:  # pragma: no cover - guarded at construction/loading boundaries
            raise ValueError("unsupported BudgetSpec schema version")
        return {
            **common,
            **compute,
            "evaluation_cases": self.evaluation_cases,
            "infrastructure_retries": self.infrastructure_retries,
            "repair_attempts_per_opportunity": self.repair_attempts_per_opportunity,
            "seed_evaluations": self.seed_evaluations,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BudgetSpec:
        if payload.get("schema_name") != "BudgetSpec":
            raise ValueError("expected BudgetSpec schema")
        version = payload.get("schema_version")
        common_fields = {
            "schema_name",
            "schema_version",
            "proposal_opportunities",
            "provider_attempts_per_opportunity",
            "prompt_tokens",
            "completion_tokens",
            "repairs",
            "candidate_training_attempts",
            "training_steps",
            "training_examples",
            "evaluation_cases",
            "infrastructure_retries",
            "repair_attempts_per_opportunity",
            "seed_evaluations",
        }
        if version == "1.0":
            expected = common_fields | {"mps_seconds"}
            if set(payload) != expected:
                raise ValueError("BudgetSpec v1 has invalid fields")
            result = cls(
                **{
                    key: value
                    for key, value in payload.items()
                    if key not in {"schema_name", "schema_version", "mps_seconds"}
                },
                mps_seconds=payload["mps_seconds"],
            )
            object.__setattr__(result, "schema_version", "1.0")
            return result
        if version == "2.0":
            expected = common_fields | {
                "accelerator_kind",
                "accelerator_seconds",
            }
            if set(payload) != expected:
                raise ValueError("BudgetSpec v2 has invalid fields")
            return cls(
                **{
                    key: value
                    for key, value in payload.items()
                    if key not in {"schema_name", "schema_version"}
                }
            )
        raise ValueError("unsupported BudgetSpec schema version")


@dataclass
class BudgetLedger:
    """Mutable actuals with state-machine guards around proposal opportunities."""

    spec: BudgetSpec
    accelerator_kind: str = ""
    seed_evaluations: int = 0
    proposal_opportunities: int = 0
    provider_attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    unknown_provider_usage: int = 0
    parse_failures: int = 0
    candidate_training_attempts: int = 0
    training_steps: int = 0
    training_examples: int = 0
    accelerator_seconds: float = 0.0
    evaluation_cases: int = 0
    repairs: int = 0
    infrastructure_retries: int = 0
    accepted: int = 0
    rejected: int = 0
    invalid: int = 0
    scientific_failures: int = 0
    infrastructure_failures: int = 0
    active_opportunity: int | None = None
    schema_version: str = field(default="2.0", init=False)
    _provider_attempts_by_opportunity: dict[int, int] = field(default_factory=dict)
    _repairs_by_opportunity: dict[int, int] = field(default_factory=dict)
    _candidate_source_hashes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.accelerator_kind:
            self.accelerator_kind = self.spec.accelerator_kind
        _require_accelerator_kind(self.accelerator_kind)
        if self.accelerator_kind != self.spec.accelerator_kind:
            raise ValueError("ledger accelerator_kind differs from its frozen budget")

    @property
    def mps_seconds(self) -> float:
        """Deprecated source-code alias for pre-v2 consumers."""

        return self.accelerator_seconds

    @property
    def unique_candidate_sources(self) -> int:
        return len(self._candidate_source_hashes)

    @property
    def terminal_opportunities(self) -> int:
        return (
            self.accepted
            + self.rejected
            + self.invalid
            + self.scientific_failures
            + self.infrastructure_failures
        )

    def _add(self, field_name: str, amount: int | float, ceiling: int | float) -> None:
        if isinstance(ceiling, int) and not isinstance(ceiling, bool):
            require_int(amount, field_name)
        elif (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(amount)
        ):
            raise ValueError(f"{field_name} must be a finite number")
        if isinstance(ceiling, float) and not math.isfinite(ceiling):
            raise ValueError(f"{field_name} ceiling must be finite")
        if amount < 0:
            raise ValueError(f"cannot subtract from {field_name}")
        updated = getattr(self, field_name) + amount
        if updated > ceiling:
            raise BudgetExceeded(
                f"{field_name} ceiling exceeded: requested {updated}, frozen {ceiling}"
            )
        setattr(self, field_name, updated)

    def _resolve_accelerator_usage(
        self,
        *,
        accelerator_kind: str | None,
        accelerator_seconds: float | None,
        mps_seconds: float | None,
    ) -> tuple[str, int | float]:
        if mps_seconds is not None:
            _require_nonnegative_number(mps_seconds, "mps_seconds")
            if accelerator_kind is not None or accelerator_seconds is not None:
                raise ValueError(
                    "mps_seconds cannot be combined with accelerator fields"
                )
            # The old argument was the sole compute counter. Preserve source
            # compatibility while charging it to the frozen v2 accelerator kind.
            return self.spec.accelerator_kind, mps_seconds
        if accelerator_seconds is None:
            raise ValueError("accelerator_seconds is required")
        kind = (
            self.spec.accelerator_kind
            if accelerator_kind is None
            else _require_accelerator_kind(accelerator_kind)
        )
        seconds = _require_nonnegative_number(
            accelerator_seconds, "accelerator_seconds"
        )
        if kind != self.spec.accelerator_kind:
            raise ValueError("accelerator_kind differs from the frozen budget")
        return kind, seconds

    def record_seed_evaluation(
        self,
        *,
        training_attempts: int,
        training_steps: int,
        training_examples: int,
        evaluation_cases: int,
        accelerator_kind: str | None = None,
        accelerator_seconds: float | None = None,
        mps_seconds: float | None = None,
    ) -> None:
        if self.active_opportunity is not None or self.proposal_opportunities:
            raise OpportunityStateError("seed evaluation must precede all proposals")
        self._add("seed_evaluations", 1, self.spec.seed_evaluations)
        self.record_training(
            attempts=training_attempts,
            steps=training_steps,
            examples=training_examples,
            accelerator_kind=accelerator_kind,
            accelerator_seconds=accelerator_seconds,
            mps_seconds=mps_seconds,
            require_active=False,
        )
        self.record_evaluation(evaluation_cases, require_active=False)

    def begin_opportunity(self, index: int) -> None:
        if self.seed_evaluations != self.spec.seed_evaluations:
            raise OpportunityStateError("the seed must be evaluated exactly once first")
        if self.active_opportunity is not None:
            raise OpportunityStateError(
                f"opportunity {self.active_opportunity} is not terminal"
            )
        expected = self.proposal_opportunities + 1
        if index != expected:
            raise OpportunityStateError(
                f"expected opportunity {expected}, received {index}"
            )
        self._add("proposal_opportunities", 1, self.spec.proposal_opportunities)
        self.active_opportunity = index
        self._provider_attempts_by_opportunity.setdefault(index, 0)
        self._repairs_by_opportunity.setdefault(index, 0)

    def start_provider_attempt(self) -> int:
        index = self._require_active()
        count = self._provider_attempts_by_opportunity[index] + 1
        if count > self.spec.provider_attempts_per_opportunity:
            raise BudgetExceeded(
                f"provider attempts for opportunity {index} exceed the per-opportunity ceiling"
            )
        self._provider_attempts_by_opportunity[index] = count
        self.provider_attempts += 1
        return count

    def record_provider_usage(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        self._require_active()
        if prompt_tokens is None or completion_tokens is None:
            self.unknown_provider_usage += 1
        if prompt_tokens is not None:
            self._add("prompt_tokens", prompt_tokens, self.spec.prompt_tokens)
        if completion_tokens is not None:
            self._add(
                "completion_tokens", completion_tokens, self.spec.completion_tokens
            )

    def record_parse_failure(self) -> None:
        self._require_active()
        self.parse_failures += 1

    def record_candidate_source(self, source_hash: str) -> None:
        self._require_active()
        if not source_hash:
            raise ValueError("candidate source hash cannot be empty")
        self._candidate_source_hashes.add(source_hash)

    def record_training(
        self,
        *,
        attempts: int,
        steps: int,
        examples: int,
        accelerator_kind: str | None = None,
        accelerator_seconds: float | None = None,
        mps_seconds: float | None = None,
        require_active: bool = True,
    ) -> None:
        if require_active:
            self._require_active()
        kind, seconds = self._resolve_accelerator_usage(
            accelerator_kind=accelerator_kind,
            accelerator_seconds=accelerator_seconds,
            mps_seconds=mps_seconds,
        )
        if kind != self.accelerator_kind:
            raise ValueError("recorded accelerator differs from the ledger")
        self._add(
            "candidate_training_attempts",
            attempts,
            self.spec.candidate_training_attempts,
        )
        self._add("training_steps", steps, self.spec.training_steps)
        self._add("training_examples", examples, self.spec.training_examples)
        self._add(
            "accelerator_seconds", seconds, self.spec.accelerator_seconds
        )

    def record_evaluation(self, cases: int, *, require_active: bool = True) -> None:
        if require_active:
            self._require_active()
        self._add("evaluation_cases", cases, self.spec.evaluation_cases)

    def record_repair(self) -> None:
        index = self._require_active()
        count = self._repairs_by_opportunity.setdefault(index, 0) + 1
        if count > self.spec.repair_attempts_per_opportunity:
            raise BudgetExceeded(
                f"repairs for opportunity {index} exceed the per-opportunity ceiling"
            )
        self._add("repairs", 1, self.spec.repairs)
        self._repairs_by_opportunity[index] = count

    def record_infrastructure_retry(self, *, require_active: bool = True) -> None:
        if require_active:
            self._require_active()
        self._add("infrastructure_retries", 1, self.spec.infrastructure_retries)

    def finish_opportunity(self, outcome: OpportunityOutcome) -> None:
        self._require_active()
        counter = {
            OpportunityOutcome.ACCEPTED: "accepted",
            OpportunityOutcome.REJECTED: "rejected",
            OpportunityOutcome.INVALID: "invalid",
            OpportunityOutcome.SCIENTIFIC_FAILURE: "scientific_failures",
            OpportunityOutcome.INFRASTRUCTURE_FAILURE: "infrastructure_failures",
        }[outcome]
        setattr(self, counter, getattr(self, counter) + 1)
        self.active_opportunity = None

    def _require_active(self) -> int:
        if self.active_opportunity is None:
            raise OpportunityStateError("no active proposal opportunity")
        return self.active_opportunity

    def to_dict(self) -> dict[str, Any]:
        common = {
            "schema_name": "BudgetLedger",
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "seed_evaluations": self.seed_evaluations,
            "proposal_opportunities": self.proposal_opportunities,
            "provider_attempts": self.provider_attempts,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "unknown_provider_usage": self.unknown_provider_usage,
            "parse_failures": self.parse_failures,
            "unique_candidate_sources": self.unique_candidate_sources,
            "candidate_source_hashes": sorted(self._candidate_source_hashes),
            "candidate_training_attempts": self.candidate_training_attempts,
            "training_steps": self.training_steps,
            "training_examples": self.training_examples,
        }
        if self.schema_version == "1.0":
            compute = {"mps_seconds": self.accelerator_seconds}
        elif self.schema_version == "2.0":
            compute = {
                "accelerator_kind": self.accelerator_kind,
                "accelerator_seconds": self.accelerator_seconds,
            }
        else:  # pragma: no cover - guarded at loading boundaries
            raise ValueError("unsupported BudgetLedger schema version")
        return {
            **common,
            **compute,
            "evaluation_cases": self.evaluation_cases,
            "repairs": self.repairs,
            "infrastructure_retries": self.infrastructure_retries,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "invalid": self.invalid,
            "scientific_failures": self.scientific_failures,
            "infrastructure_failures": self.infrastructure_failures,
            "terminal_opportunities": self.terminal_opportunities,
            "active_opportunity": self.active_opportunity,
            "provider_attempts_by_opportunity": {
                str(key): value
                for key, value in sorted(
                    self._provider_attempts_by_opportunity.items()
                )
            },
            "repairs_by_opportunity": {
                str(key): value
                for key, value in sorted(self._repairs_by_opportunity.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BudgetLedger:
        if payload.get("schema_name") != "BudgetLedger":
            raise ValueError("expected BudgetLedger schema")
        version = payload.get("schema_version")
        if version not in {"1.0", "2.0"}:
            raise ValueError("unsupported BudgetLedger schema version")
        spec = BudgetSpec.from_dict(payload["spec"])
        if version == "1.0":
            if spec.schema_version != "1.0" or spec.accelerator_kind != "mps":
                raise ValueError("BudgetLedger v1 requires a BudgetSpec v1 MPS budget")
            compute_fields = {"mps_seconds"}
            accelerator_kind = "mps"
            accelerator_seconds = payload.get("mps_seconds", 0.0)
        else:
            compute_fields = {"accelerator_kind", "accelerator_seconds"}
            accelerator_kind = payload.get("accelerator_kind")
            accelerator_seconds = payload.get("accelerator_seconds", 0.0)
        expected_fields = {
            "schema_name",
            "schema_version",
            "spec",
            "seed_evaluations",
            "proposal_opportunities",
            "provider_attempts",
            "prompt_tokens",
            "completion_tokens",
            "unknown_provider_usage",
            "parse_failures",
            "unique_candidate_sources",
            "candidate_source_hashes",
            "candidate_training_attempts",
            "training_steps",
            "training_examples",
            *compute_fields,
            "evaluation_cases",
            "repairs",
            "infrastructure_retries",
            "accepted",
            "rejected",
            "invalid",
            "scientific_failures",
            "infrastructure_failures",
            "terminal_opportunities",
            "active_opportunity",
            "provider_attempts_by_opportunity",
            "repairs_by_opportunity",
        }
        if set(payload) != expected_fields:
            raise ValueError(f"BudgetLedger v{version[0]} has invalid fields")
        ledger = cls(spec=spec, accelerator_kind=accelerator_kind)
        ledger.schema_version = version
        scalar_fields = (
            "seed_evaluations",
            "proposal_opportunities",
            "provider_attempts",
            "prompt_tokens",
            "completion_tokens",
            "unknown_provider_usage",
            "parse_failures",
            "candidate_training_attempts",
            "training_steps",
            "training_examples",
            "evaluation_cases",
            "repairs",
            "infrastructure_retries",
            "accepted",
            "rejected",
            "invalid",
            "scientific_failures",
            "infrastructure_failures",
            "active_opportunity",
        )
        for name in scalar_fields:
            setattr(ledger, name, payload[name])
        ledger.accelerator_seconds = accelerator_seconds
        ledger._provider_attempts_by_opportunity = {
            int(key): require_int(value, "provider attempt count")
            for key, value in payload["provider_attempts_by_opportunity"].items()
        }
        ledger._repairs_by_opportunity = {
            int(key): require_int(value, "repair count")
            for key, value in payload["repairs_by_opportunity"].items()
        }
        candidate_hashes = payload["candidate_source_hashes"]
        if not isinstance(candidate_hashes, list) or any(
            not isinstance(value, str) or not value for value in candidate_hashes
        ):
            raise ValueError("candidate_source_hashes must be non-empty strings")
        ledger._candidate_source_hashes = set(candidate_hashes)
        ledger.validate()
        if payload["unique_candidate_sources"] != ledger.unique_candidate_sources:
            raise ValueError("unique candidate-source count does not reconstruct")
        if payload["terminal_opportunities"] != ledger.terminal_opportunities:
            raise ValueError("terminal opportunity count does not reconstruct")
        return ledger

    def validate(self) -> None:
        integer_fields = (
            "seed_evaluations",
            "proposal_opportunities",
            "provider_attempts",
            "prompt_tokens",
            "completion_tokens",
            "unknown_provider_usage",
            "parse_failures",
            "candidate_training_attempts",
            "training_steps",
            "training_examples",
            "evaluation_cases",
            "repairs",
            "infrastructure_retries",
            "accepted",
            "rejected",
            "invalid",
            "scientific_failures",
            "infrastructure_failures",
        )
        for name in integer_fields:
            value = require_int(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"stored {name} cannot be negative")
        if self.active_opportunity is not None:
            require_int(self.active_opportunity, "active_opportunity")
        _require_accelerator_kind(self.accelerator_kind)
        if self.accelerator_kind != self.spec.accelerator_kind:
            raise ValueError("stored accelerator_kind differs from its budget")
        _require_nonnegative_number(
            self.accelerator_seconds, "stored accelerator_seconds"
        )
        if self.seed_evaluations > self.spec.seed_evaluations:
            raise ValueError("stored seed-evaluation count exceeds its ceiling")
        if self.proposal_opportunities > self.spec.proposal_opportunities:
            raise ValueError("stored proposal count exceeds its ceiling")
        if self.terminal_opportunities > self.proposal_opportunities:
            raise ValueError("more terminal outcomes than proposal opportunities")
        if self.active_opportunity is not None:
            if self.active_opportunity != self.terminal_opportunities + 1:
                raise ValueError("stored active opportunity is out of sequence")
        elif self.terminal_opportunities != self.proposal_opportunities:
            raise ValueError("nonterminal opportunity missing from stored ledger")
        ceilings = {
            "prompt_tokens": self.spec.prompt_tokens,
            "completion_tokens": self.spec.completion_tokens,
            "candidate_training_attempts": self.spec.candidate_training_attempts,
            "training_steps": self.spec.training_steps,
            "training_examples": self.spec.training_examples,
            "accelerator_seconds": self.spec.accelerator_seconds,
            "evaluation_cases": self.spec.evaluation_cases,
            "repairs": self.spec.repairs,
            "infrastructure_retries": self.spec.infrastructure_retries,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"stored {name} must be finite")
            if value < 0 or value > ceiling:
                raise ValueError(f"stored {name} lies outside its frozen ceiling")
        if self.provider_attempts != sum(
            self._provider_attempts_by_opportunity.values()
        ):
            raise ValueError("provider-attempt total does not reconstruct")
        if any(
            count < 0 or count > self.spec.provider_attempts_per_opportunity
            for count in self._provider_attempts_by_opportunity.values()
        ):
            raise ValueError("stored provider attempts exceed an opportunity ceiling")
        if self.repairs != sum(self._repairs_by_opportunity.values()):
            raise ValueError("repair total does not reconstruct")
        if any(
            count < 0 or count > self.spec.repair_attempts_per_opportunity
            for count in self._repairs_by_opportunity.values()
        ):
            raise ValueError("stored repairs exceed an opportunity ceiling")
