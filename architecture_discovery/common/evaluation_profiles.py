"""Versioned evaluation profiles and fail-closed scientific validation.

Training profiles govern optimizer work.  These profiles govern evaluation
exposure.  Unit, smoke, and development profiles are synthetic engineering
profiles and can target any layer in tests.  Scientific profiles are bound to
one layer and deliberately omit a default case count so that a study cannot
silently inherit a smoke-sized evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable


SCIENTIFIC_CASE_FLOOR = 10_000


class EvaluationLayer(StrEnum):
    SEARCH = "layer_a"
    QUALIFICATION = "layer_b"
    CONFIRMATION = "layer_c"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    version: str
    scientific: bool
    fixed_layer: EvaluationLayer | None
    default_case_count: int | None
    minimum_case_count: int
    synthetic_only: bool
    controller_visible: bool
    sealed: bool

    @property
    def profile_hash(self) -> str:
        payload = asdict(self)
        payload["fixed_layer"] = (
            self.fixed_layer.value if self.fixed_layer is not None else None
        )
        return _stable_hash(payload)

    def validate_definition(self) -> None:
        if not self.name or not self.version:
            raise ValueError("evaluation profiles require a name and version")
        if self.minimum_case_count <= 0:
            raise ValueError("minimum_case_count must be positive")
        if self.default_case_count is not None:
            if self.default_case_count < self.minimum_case_count:
                raise ValueError("default_case_count is below the profile minimum")
        if self.scientific:
            if self.fixed_layer is None:
                raise ValueError("scientific profiles must be bound to one layer")
            if self.default_case_count is not None:
                raise ValueError(
                    "scientific profiles cannot supply an implicit case count"
                )
            if self.minimum_case_count < SCIENTIFIC_CASE_FLOOR:
                raise ValueError("scientific profile minimum is below the safety floor")
            if self.synthetic_only:
                raise ValueError("scientific profiles cannot be synthetic-only")
            if self.controller_visible != (
                self.fixed_layer is EvaluationLayer.SEARCH
            ):
                raise ValueError(
                    "only a scientific Layer A profile may be controller-visible"
                )
            if self.sealed != (
                self.fixed_layer
                in {EvaluationLayer.QUALIFICATION, EvaluationLayer.CONFIRMATION}
            ):
                raise ValueError("scientific Layer B and C profiles must be sealed")
        else:
            if not self.synthetic_only:
                raise ValueError("engineering profiles must be synthetic-only")
            if self.fixed_layer is not None:
                raise ValueError(
                    "engineering profiles are layer-neutral synthetic fixtures"
                )
            if self.sealed or self.controller_visible:
                raise ValueError(
                    "visibility is selected explicitly when resolving a synthetic plan"
                )


@dataclass(frozen=True)
class EvaluationPlan:
    profile_name: str
    profile_version: str
    profile_hash: str
    layer: EvaluationLayer
    case_count: int
    case_source_id: str
    case_source_sha256: str
    scientific: bool
    synthetic: bool
    controller_visible: bool
    sealed: bool
    pi_decision_record_id: str | None

    @property
    def plan_hash(self) -> str:
        payload = asdict(self)
        payload["layer"] = self.layer.value
        return _stable_hash(payload)

    def validate(self) -> None:
        if self.case_count <= 0:
            raise ValueError("case_count must be positive")
        if not self.case_source_id:
            raise ValueError("case_source_id is required")
        _require_sha256(self.case_source_sha256, "case_source_sha256")
        profile = get_evaluation_profile(self.profile_name)
        if self.profile_version != profile.version:
            raise ValueError("evaluation plan profile version mismatch")
        if self.profile_hash != profile.profile_hash:
            raise ValueError("evaluation plan profile hash mismatch")
        if self.scientific != profile.scientific:
            raise ValueError("evaluation plan scientific flag mismatch")
        if self.case_count < profile.minimum_case_count:
            raise ValueError("evaluation case count is below the profile minimum")
        if profile.fixed_layer is not None and self.layer is not profile.fixed_layer:
            raise ValueError("evaluation plan uses a profile for the wrong layer")
        if self.scientific:
            if self.synthetic:
                raise ValueError("scientific evaluation cannot use synthetic cases")
            if not self.pi_decision_record_id:
                raise ValueError(
                    "scientific evaluation requires a frozen PI decision record"
                )
            expected_visible = self.layer is EvaluationLayer.SEARCH
            if self.controller_visible != expected_visible:
                raise ValueError("scientific evaluation visibility is invalid")
            expected_sealed = self.layer in {
                EvaluationLayer.QUALIFICATION,
                EvaluationLayer.CONFIRMATION,
            }
            if self.sealed != expected_sealed:
                raise ValueError("scientific evaluation sealing is invalid")
        else:
            if not self.synthetic:
                raise ValueError("engineering plans must use synthetic cases")
            if self.pi_decision_record_id is not None:
                raise ValueError("synthetic plans must not claim PI approval")


UNIT_EVAL_V1 = EvaluationProfile(
    name="unit_eval_v1",
    version="1",
    scientific=False,
    fixed_layer=None,
    default_case_count=4,
    minimum_case_count=1,
    synthetic_only=True,
    controller_visible=False,
    sealed=False,
)

SMOKE_EVAL_V1 = EvaluationProfile(
    name="smoke_eval_v1",
    version="1",
    scientific=False,
    fixed_layer=None,
    default_case_count=64,
    minimum_case_count=8,
    synthetic_only=True,
    controller_visible=False,
    sealed=False,
)

DEVELOPMENT_EVAL_V1 = EvaluationProfile(
    name="development_eval_v1",
    version="1",
    scientific=False,
    fixed_layer=None,
    default_case_count=512,
    minimum_case_count=64,
    synthetic_only=True,
    controller_visible=False,
    sealed=False,
)

SCIENTIFIC_LAYER_A_V1 = EvaluationProfile(
    name="scientific_layer_a_v1",
    version="1",
    scientific=True,
    fixed_layer=EvaluationLayer.SEARCH,
    default_case_count=None,
    minimum_case_count=SCIENTIFIC_CASE_FLOOR,
    synthetic_only=False,
    controller_visible=True,
    sealed=False,
)

SCIENTIFIC_LAYER_B_V1 = EvaluationProfile(
    name="scientific_layer_b_v1",
    version="1",
    scientific=True,
    fixed_layer=EvaluationLayer.QUALIFICATION,
    default_case_count=None,
    minimum_case_count=SCIENTIFIC_CASE_FLOOR,
    synthetic_only=False,
    controller_visible=False,
    sealed=True,
)

SCIENTIFIC_LAYER_C_V1 = EvaluationProfile(
    name="scientific_layer_c_v1",
    version="1",
    scientific=True,
    fixed_layer=EvaluationLayer.CONFIRMATION,
    default_case_count=None,
    minimum_case_count=SCIENTIFIC_CASE_FLOOR,
    synthetic_only=False,
    controller_visible=False,
    sealed=True,
)


EVALUATION_PROFILES = {
    profile.name: profile
    for profile in (
        UNIT_EVAL_V1,
        SMOKE_EVAL_V1,
        DEVELOPMENT_EVAL_V1,
        SCIENTIFIC_LAYER_A_V1,
        SCIENTIFIC_LAYER_B_V1,
        SCIENTIFIC_LAYER_C_V1,
    )
}


def get_evaluation_profile(name: str) -> EvaluationProfile:
    try:
        profile = EVALUATION_PROFILES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown evaluation profile {name!r}; choose one of "
            f"{sorted(EVALUATION_PROFILES)}"
        ) from error
    profile.validate_definition()
    return profile


def resolve_evaluation_plan(
    profile_name: str,
    *,
    layer: EvaluationLayer,
    case_source_id: str,
    case_source_sha256: str,
    case_count: int | None = None,
    pi_decision_record_id: str | None = None,
) -> EvaluationPlan:
    """Resolve a profile without permitting an implicit scientific case count."""

    profile = get_evaluation_profile(profile_name)
    if profile.fixed_layer is not None and layer is not profile.fixed_layer:
        raise ValueError(f"{profile.name} is fixed to {profile.fixed_layer.value}")
    resolved_count = profile.default_case_count if case_count is None else case_count
    if resolved_count is None:
        raise ValueError(
            "scientific case_count must be supplied by the frozen study decision"
        )
    if resolved_count < profile.minimum_case_count:
        raise ValueError(
            f"{profile.name} requires at least {profile.minimum_case_count} cases"
        )
    plan = EvaluationPlan(
        profile_name=profile.name,
        profile_version=profile.version,
        profile_hash=profile.profile_hash,
        layer=layer,
        case_count=int(resolved_count),
        case_source_id=case_source_id,
        case_source_sha256=case_source_sha256,
        scientific=profile.scientific,
        synthetic=profile.synthetic_only,
        controller_visible=(
            layer is EvaluationLayer.SEARCH if not profile.scientific else profile.controller_visible
        ),
        sealed=(
            layer in {EvaluationLayer.QUALIFICATION, EvaluationLayer.CONFIRMATION}
            if not profile.scientific
            else profile.sealed
        ),
        pi_decision_record_id=pi_decision_record_id,
    )
    plan.validate()
    return plan


def validate_disjoint_scientific_plans(plans: Iterable[EvaluationPlan]) -> None:
    """Require exactly one disjoint, scientific A/B/C plan."""

    values = tuple(plans)
    layers = [plan.layer for plan in values]
    expected = {
        EvaluationLayer.SEARCH,
        EvaluationLayer.QUALIFICATION,
        EvaluationLayer.CONFIRMATION,
    }
    if len(values) != 3 or set(layers) != expected:
        raise ValueError("scientific evaluation requires exactly one A, B, and C plan")
    for plan in values:
        plan.validate()
        if not plan.scientific:
            raise ValueError("engineering profiles cannot enter a scientific plan set")
    source_ids = {plan.case_source_id for plan in values}
    source_hashes = {plan.case_source_sha256 for plan in values}
    if len(source_ids) != 3 or len(source_hashes) != 3:
        raise ValueError("Layer A, B, and C case sources must be disjoint")


def evaluation_plan_from_dict(payload: dict[str, Any]) -> EvaluationPlan:
    """Reconstruct and validate a serialized evaluation plan."""

    values = dict(payload)
    values["layer"] = EvaluationLayer(values["layer"])
    plan = EvaluationPlan(**values)
    plan.validate()
    return plan
