"""Typed, versioned records for the three evaluation layers."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from common.evaluation_profiles import EvaluationLayer


SCHEMA_VERSION = "1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _require_identifier(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed identifier")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise ValueError(f"{field_name} must not contain path separators")


def require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def require_bool(value: object, field_name: str) -> bool:
    """Reject JSON type confusion instead of applying Python truthiness."""

    if type(value) is not bool:
        raise ValueError(f"{field_name} must be boolean")
    return value


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _probability(value: object, field_name: str) -> float:
    converted = _finite_number(value, field_name)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return converted


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return value


def _metric_pairs(
    values: tuple[tuple[str, float], ...], field_name: str
) -> None:
    names: set[str] = set()
    for name, value in values:
        _require_identifier(name, f"{field_name} name")
        if name in names:
            raise ValueError(f"duplicate {field_name} name {name!r}")
        names.add(name)
        _finite_number(value, f"{field_name} value")


@dataclass(frozen=True)
class RecordEnvelope:
    schema_name: str
    schema_version: str
    record_id: str
    study_id: str
    block_id: str
    run_id: str
    condition_id: str
    created_at_utc: str
    writer_component: str
    code_sha256: str
    config_sha256: str
    environment_sha256: str

    @classmethod
    def create(
        cls,
        *,
        schema_name: str,
        study_id: str,
        block_id: str,
        run_id: str,
        condition_id: str,
        writer_component: str,
        code_sha256: str,
        config_sha256: str,
        environment_sha256: str,
        record_id: str | None = None,
    ) -> "RecordEnvelope":
        envelope = cls(
            schema_name=schema_name,
            schema_version=SCHEMA_VERSION,
            record_id=record_id or f"{schema_name}-{uuid.uuid4().hex}",
            study_id=study_id,
            block_id=block_id,
            run_id=run_id,
            condition_id=condition_id,
            created_at_utc=utc_now(),
            writer_component=writer_component,
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            environment_sha256=environment_sha256,
        )
        envelope.validate(expected_schema=schema_name)
        return envelope

    def validate(self, *, expected_schema: str) -> None:
        if self.schema_name != expected_schema:
            raise ValueError(
                f"expected schema {expected_schema!r}, got {self.schema_name!r}"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported evaluation record schema version")
        for name in (
            "record_id",
            "study_id",
            "block_id",
            "run_id",
            "condition_id",
            "writer_component",
        ):
            _require_identifier(getattr(self, name), name)
        for name in ("code_sha256", "config_sha256", "environment_sha256"):
            require_sha256(getattr(self, name), name)
        if not self.created_at_utc.endswith("Z"):
            raise ValueError("created_at_utc must be an explicit UTC timestamp")
        try:
            datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("created_at_utc is not a valid timestamp") from error


@dataclass(frozen=True)
class ArtifactReference:
    layer: EvaluationLayer
    relative_path: str
    sha256: str

    def validate(self, *, expected_layer: EvaluationLayer) -> None:
        if self.layer is not expected_layer:
            raise ValueError(
                f"artifact belongs to {self.layer.value}, expected {expected_layer.value}"
            )
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or not self.relative_path or ".." in path.parts:
            raise ValueError("artifact references must be safe relative paths")
        require_sha256(self.sha256, "artifact sha256")


CONTROLLER_SEARCH_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "record_id",
        "run_id",
        "condition_id",
        "candidate_id",
        "execution_ok",
        "transformer_valid",
        "public_accuracy",
        "search_score",
        "eligible_for_parent",
        "failure_stage",
        "infrastructure_failure",
        "online_descriptor_codes",
    }
)


@dataclass(frozen=True)
class ControllerSearchView:
    """The complete and only evaluation object allowed into a controller."""

    schema_name: str
    schema_version: str
    record_id: str
    run_id: str
    condition_id: str
    candidate_id: str
    execution_ok: bool
    transformer_valid: bool
    public_accuracy: float
    search_score: float
    eligible_for_parent: bool
    failure_stage: str
    infrastructure_failure: bool
    online_descriptor_codes: tuple[tuple[str, float], ...]

    def as_dict(self) -> Mapping[str, Any]:
        payload = asdict(self)
        if set(payload) != CONTROLLER_SEARCH_FIELDS:
            raise RuntimeError("controller view no longer matches its field allowlist")
        return MappingProxyType(payload)


@dataclass(frozen=True)
class SearchEvaluationRecord:
    envelope: RecordEnvelope
    candidate_id: str
    training_record_id: str
    execution_ok: bool
    transformer_valid: bool
    public_accuracy: float
    search_score: float
    eligible_for_parent: bool
    failure_stage: str = ""
    infrastructure_failure: bool = False
    parameter_count_metadata: int = 0
    online_descriptor_codes: tuple[tuple[str, float], ...] = ()
    public_artifacts: tuple[ArtifactReference, ...] = ()
    runtime_validity_artifact: ArtifactReference | None = None

    def __post_init__(self) -> None:
        self.envelope.validate(expected_schema="search_evaluation")
        _require_identifier(self.candidate_id, "candidate_id")
        _require_identifier(self.training_record_id, "training_record_id")
        _probability(self.public_accuracy, "public_accuracy")
        _probability(self.search_score, "search_score")
        _nonnegative_integer(
            self.parameter_count_metadata,
            "parameter_count_metadata",
        )
        if self.eligible_for_parent and not (
            self.execution_ok and self.transformer_valid
        ):
            raise ValueError(
                "only successfully executed, transformer-valid candidates are eligible"
            )
        _metric_pairs(self.online_descriptor_codes, "online descriptor")
        for artifact in self.public_artifacts:
            artifact.validate(expected_layer=EvaluationLayer.SEARCH)
        if self.runtime_validity_artifact is not None:
            self.runtime_validity_artifact.validate(
                expected_layer=EvaluationLayer.SEARCH
            )

    def controller_view(self) -> ControllerSearchView:
        return ControllerSearchView(
            schema_name=self.envelope.schema_name,
            schema_version=self.envelope.schema_version,
            record_id=self.envelope.record_id,
            run_id=self.envelope.run_id,
            condition_id=self.envelope.condition_id,
            candidate_id=self.candidate_id,
            execution_ok=self.execution_ok,
            transformer_valid=self.transformer_valid,
            public_accuracy=self.public_accuracy,
            search_score=self.search_score,
            eligible_for_parent=self.eligible_for_parent,
            failure_stage=self.failure_stage,
            infrastructure_failure=self.infrastructure_failure,
            online_descriptor_codes=self.online_descriptor_codes,
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @property
    def record_hash(self) -> str:
        return content_sha256(self.to_dict())


def search_evaluation_from_dict(payload: Mapping[str, Any]) -> SearchEvaluationRecord:
    """Reconstruct an exact Layer A record from an untrusted JSON object."""

    required = {
        "envelope",
        "candidate_id",
        "training_record_id",
        "execution_ok",
        "transformer_valid",
        "public_accuracy",
        "search_score",
        "eligible_for_parent",
        "failure_stage",
        "infrastructure_failure",
        "parameter_count_metadata",
        "online_descriptor_codes",
        "public_artifacts",
    }
    optional = {"runtime_validity_artifact"}
    if not required.issubset(payload) or set(payload).difference(required | optional):
        raise ValueError("Layer A worker response has unexpected or missing fields")
    envelope_value = payload["envelope"]
    if not isinstance(envelope_value, Mapping):
        raise ValueError("Layer A worker response envelope must be an object")
    envelope = RecordEnvelope(**dict(envelope_value))
    descriptor_value = payload["online_descriptor_codes"]
    if not isinstance(descriptor_value, (list, tuple)):
        raise ValueError("online descriptor codes must be a sequence")
    artifact_value = payload["public_artifacts"]
    if not isinstance(artifact_value, (list, tuple)):
        raise ValueError("public artifacts must be a sequence")
    artifacts: list[ArtifactReference] = []
    for item in artifact_value:
        if not isinstance(item, Mapping):
            raise ValueError("public artifact references must be objects")
        values = dict(item)
        values["layer"] = EvaluationLayer(values["layer"])
        artifacts.append(ArtifactReference(**values))
    # ``runtime_validity_artifact`` was introduced as an optional additive v1
    # field.  Accept records written before it existed instead of silently
    # making old schema-v1 records unparsable.
    runtime_artifact_value = payload.get("runtime_validity_artifact")
    runtime_artifact: ArtifactReference | None = None
    if runtime_artifact_value is not None:
        if not isinstance(runtime_artifact_value, Mapping):
            raise ValueError("runtime validity artifact must be an object or null")
        runtime_values = dict(runtime_artifact_value)
        runtime_values["layer"] = EvaluationLayer(runtime_values["layer"])
        runtime_artifact = ArtifactReference(**runtime_values)
    return SearchEvaluationRecord(
        envelope=envelope,
        candidate_id=str(payload["candidate_id"]),
        training_record_id=str(payload["training_record_id"]),
        execution_ok=require_bool(payload["execution_ok"], "execution_ok"),
        transformer_valid=require_bool(
            payload["transformer_valid"], "transformer_valid"
        ),
        public_accuracy=_probability(payload["public_accuracy"], "public_accuracy"),
        search_score=_probability(payload["search_score"], "search_score"),
        eligible_for_parent=require_bool(
            payload["eligible_for_parent"], "eligible_for_parent"
        ),
        failure_stage=str(payload["failure_stage"]),
        infrastructure_failure=require_bool(
            payload["infrastructure_failure"], "infrastructure_failure"
        ),
        parameter_count_metadata=_nonnegative_integer(
            payload["parameter_count_metadata"],
            "parameter_count_metadata",
        ),
        online_descriptor_codes=tuple(
            (
                str(item[0]),
                _finite_number(item[1], "online descriptor value"),
            )
            for item in descriptor_value
        ),
        public_artifacts=tuple(artifacts),
        runtime_validity_artifact=runtime_artifact,
    )


@dataclass(frozen=True)
class QualificationEvaluationRecord:
    envelope: RecordEnvelope
    candidate_id: str
    frozen_snapshot_id: str
    frozen_snapshot_sha256: str
    evaluation_plan_sha256: str
    exact_match_accuracy: float
    qualifies: bool
    evaluation_complete: bool
    sealed_metrics: tuple[tuple[str, float], ...] = ()
    sealed_artifacts: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        self.envelope.validate(expected_schema="qualification_evaluation")
        _require_identifier(self.candidate_id, "candidate_id")
        _require_identifier(self.frozen_snapshot_id, "frozen_snapshot_id")
        require_sha256(self.frozen_snapshot_sha256, "frozen_snapshot_sha256")
        require_sha256(self.evaluation_plan_sha256, "evaluation_plan_sha256")
        _probability(self.exact_match_accuracy, "exact_match_accuracy")
        if self.qualifies and not self.evaluation_complete:
            raise ValueError("an incomplete Layer B evaluation cannot qualify")
        _metric_pairs(self.sealed_metrics, "sealed metric")
        for artifact in self.sealed_artifacts:
            artifact.validate(expected_layer=EvaluationLayer.QUALIFICATION)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @property
    def record_hash(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class ConfirmationEvaluationRecord:
    envelope: RecordEnvelope
    candidate_id: str
    frozen_snapshot_id: str
    frozen_candidate_sha256: str
    qualification_record_id: str
    release_authorization_id: str
    evaluation_plan_sha256: str
    exact_match_accuracy: float
    confirmed: bool
    evaluation_complete: bool
    sealed_metrics: tuple[tuple[str, float], ...] = ()
    sealed_artifacts: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        self.envelope.validate(expected_schema="confirmation_evaluation")
        for name in (
            "candidate_id",
            "frozen_snapshot_id",
            "qualification_record_id",
            "release_authorization_id",
        ):
            _require_identifier(getattr(self, name), name)
        require_sha256(self.frozen_candidate_sha256, "frozen_candidate_sha256")
        require_sha256(self.evaluation_plan_sha256, "evaluation_plan_sha256")
        _probability(self.exact_match_accuracy, "exact_match_accuracy")
        if self.confirmed and not self.evaluation_complete:
            raise ValueError("an incomplete Layer C evaluation cannot confirm")
        _metric_pairs(self.sealed_metrics, "sealed metric")
        for artifact in self.sealed_artifacts:
            artifact.validate(expected_layer=EvaluationLayer.CONFIRMATION)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @property
    def record_hash(self) -> str:
        return content_sha256(self.to_dict())
